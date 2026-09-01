//  Copyright (c) 2026-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/hot_memtable.h"
#include <cstdlib>
#include <cstring>
#include <thread>
#include "memory/arena.h"

namespace ROCKSDB_NAMESPACE {

class HotMemTableIterator : public InternalIterator {
 public:
  HotMemTableIterator(const HotMemTable* table)
      : table_(table),
        user_cmp_(table->internal_comparator_.user_comparator()),
        status_(Status::OK()) {
    std::lock_guard<std::mutex> lock(table_->index_mutex_);
    entries_.reserve(table_->index_.size());
    for (const auto& kv : table_->index_) {
      entries_.push_back(kv.second);
    }
  }

  bool Valid() const override {
    return idx_ >= 0 && idx_ < static_cast<int>(entries_.size());
  }

  void SeekToFirst() override {
    idx_ = entries_.empty() ? -1 : 0;
    UpdateCurrent();
  }

  void SeekToLast() override {
    idx_ = entries_.empty() ? -1 : static_cast<int>(entries_.size()) - 1;
    UpdateCurrent();
  }

  void Seek(const Slice& target) override {
    ParsedInternalKey pikey;
    Status s = ParseInternalKey(target, &pikey, false /* log_err_key */);
    Slice user_key = s.ok() ? pikey.user_key : target;

    int left = 0;
    int right = static_cast<int>(entries_.size()) - 1;
    int best = -1;

    while (left <= right) {
      int mid = left + (right - left) / 2;
      HotNode* node = entries_[mid];
      Slice node_key(node->UserKey(), node->user_key_len);
      int c = user_cmp_->Compare(node_key, user_key);
      if (c >= 0) {
        best = mid;
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    idx_ = best;
    UpdateCurrent();
  }

  void SeekForPrev(const Slice& target) override {
    ParsedInternalKey pikey;
    Status s = ParseInternalKey(target, &pikey, false /* log_err_key */);
    Slice user_key = s.ok() ? pikey.user_key : target;

    int left = 0;
    int right = static_cast<int>(entries_.size()) - 1;
    int best = -1;

    while (left <= right) {
      int mid = left + (right - left) / 2;
      HotNode* node = entries_[mid];
      Slice node_key(node->UserKey(), node->user_key_len);
      int c = user_cmp_->Compare(node_key, user_key);
      if (c <= 0) {
        best = mid;
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }
    idx_ = best;
    UpdateCurrent();
  }

  void Next() override {
    if (idx_ >= 0 && idx_ < static_cast<int>(entries_.size())) {
      idx_++;
      UpdateCurrent();
    }
  }

  void Prev() override {
    if (idx_ >= 0) {
      idx_--;
      UpdateCurrent();
    }
  }

  Slice key() const override {
    assert(Valid());
    return Slice(current_ikey_);
  }

  Slice value() const override {
    assert(Valid());
    return Slice(current_val_);
  }

  Status status() const override { return status_; }

 private:
  void UpdateCurrent() {
    if (!Valid()) {
      current_ikey_.clear();
      current_val_.clear();
      return;
    }
    HotNode* node = entries_[idx_];

    uint32_t v1, v2;
    do {
      v1 = node->seq_version.load(std::memory_order_acquire);
      while (v1 & 1) {
        std::this_thread::yield();
        v1 = node->seq_version.load(std::memory_order_acquire);
      }

      current_ikey_.clear();
      AppendInternalKey(&current_ikey_,
                        ParsedInternalKey(Slice(node->UserKey(), node->user_key_len),
                                          node->seq, node->value_type));
      current_val_.assign(node->ValBuf(), node->val_len);

      v2 = node->seq_version.load(std::memory_order_acquire);
    } while (v1 != v2);
  }

  const HotMemTable* table_;
  const Comparator* user_cmp_;
  Status status_;
  std::vector<HotNode*> entries_;
  int idx_{-1};
  std::string current_ikey_;
  std::string current_val_;
};

HotMemTable::HotMemTable(const InternalKeyComparator& cmp, size_t write_buffer_size,
                         uint32_t max_val_size)
    : internal_comparator_(cmp),
      write_buffer_size_(write_buffer_size > 0 ? write_buffer_size : 64 * 1024 * 1024),
      max_val_size_(max_val_size > 0 ? max_val_size : 1024),
      index_(KeyComparatorWrapper{cmp.user_comparator()}) {}

HotMemTable::~HotMemTable() {
  std::lock_guard<std::mutex> lock(index_mutex_);
  for (void* ptr : allocated_node_ptrs_) {
    free(ptr);
  }
}

bool HotMemTable::UpdateInPlace(const Slice& user_key, const Slice& value,
                               ValueType type, SequenceNumber seq,
                               uint64_t log_num) {
  HotNode* node = nullptr;
  {
    std::lock_guard<std::mutex> lock(index_mutex_);
    auto it = index_.find(user_key.ToString());
    if (it == index_.end()) {
      return false;
    }
    node = it->second;
  }

  uint32_t clamped_val_size = static_cast<uint32_t>(std::min<size_t>(value.size(), max_val_size_));

  uint32_t v = node->seq_version.load(std::memory_order_relaxed);
  node->seq_version.store(v + 1, std::memory_order_release);

  node->val_len = clamped_val_size;
  node->value_type = type;
  node->seq = seq;
  if (clamped_val_size > 0) {
    memcpy(node->ValBuf(), value.data(), clamped_val_size);
  }

  node->seq_version.store(v + 2, std::memory_order_release);
  node->hit_count.fetch_add(1, std::memory_order_relaxed);

  if (log_num > 0) {
    uint64_t cur_earliest_log = earliest_log_num_.load(std::memory_order_relaxed);
    while (log_num < cur_earliest_log &&
           !earliest_log_num_.compare_exchange_weak(cur_earliest_log, log_num)) {}
  }

  return true;
}

bool HotMemTable::Add(const Slice& user_key, const Slice& value, ValueType type,
                      SequenceNumber seq, uint64_t log_num) {
  std::lock_guard<std::mutex> lock(index_mutex_);
  std::string key_str = user_key.ToString();
  auto it = index_.find(key_str);
  if (it != index_.end()) {
    HotNode* node = it->second;
    uint32_t clamped_val_size = static_cast<uint32_t>(std::min<size_t>(value.size(), max_val_size_));
    uint32_t v = node->seq_version.load(std::memory_order_relaxed);
    node->seq_version.store(v + 1, std::memory_order_release);
    node->val_len = clamped_val_size;
    node->value_type = type;
    node->seq = seq;
    if (clamped_val_size > 0) {
      memcpy(node->ValBuf(), value.data(), clamped_val_size);
    }
    node->seq_version.store(v + 2, std::memory_order_release);
    if (log_num > 0) {
      uint64_t cur_earliest_log = earliest_log_num_.load(std::memory_order_relaxed);
      while (log_num < cur_earliest_log &&
             !earliest_log_num_.compare_exchange_weak(cur_earliest_log, log_num)) {}
    }
    return true;
  }

  size_t alloc_size = sizeof(HotNode) + user_key.size() + max_val_size_;
  void* raw = malloc(alloc_size);
  if (!raw) return false;

  HotNode* node = new (raw) HotNode();
  node->user_key_len = static_cast<uint32_t>(user_key.size());
  node->val_len = static_cast<uint32_t>(std::min<size_t>(value.size(), max_val_size_));
  node->value_type = type;
  node->seq = seq;

  memcpy(const_cast<char*>(node->UserKey()), user_key.data(), user_key.size());
  if (node->val_len > 0) {
    memcpy(node->ValBuf(), value.data(), node->val_len);
  }

  index_[key_str] = node;
  allocated_node_ptrs_.push_back(raw);
  allocated_bytes_.fetch_add(alloc_size, std::memory_order_relaxed);

  SequenceNumber cur_earliest = earliest_seq_.load(std::memory_order_relaxed);
  while (seq < cur_earliest && !earliest_seq_.compare_exchange_weak(cur_earliest, seq)) {}

  if (log_num > 0) {
    uint64_t cur_earliest_log = earliest_log_num_.load(std::memory_order_relaxed);
    while (log_num < cur_earliest_log &&
           !earliest_log_num_.compare_exchange_weak(cur_earliest_log, log_num)) {}
  }

  return true;
}

bool HotMemTable::Get(const Slice& user_key, std::string* value, Status* status,
                      SequenceNumber* seq_found) {
  HotNode* node = nullptr;
  {
    std::lock_guard<std::mutex> lock(index_mutex_);
    auto it = index_.find(user_key.ToString());
    if (it == index_.end()) {
      return false;
    }
    node = it->second;
  }

  uint32_t v1, v2;
  do {
    v1 = node->seq_version.load(std::memory_order_acquire);
    while (v1 & 1) {
      std::this_thread::yield();
      v1 = node->seq_version.load(std::memory_order_acquire);
    }

    if (node->value_type == kTypeDeletion) {
      if (status) *status = Status::NotFound();
      if (seq_found) *seq_found = node->seq;
      v2 = node->seq_version.load(std::memory_order_acquire);
      if (v1 == v2) return true;
      continue;
    }

    if (value) {
      value->assign(node->ValBuf(), node->val_len);
    }
    if (status) *status = Status::OK();
    if (seq_found) *seq_found = node->seq;

    v2 = node->seq_version.load(std::memory_order_acquire);
  } while (v1 != v2);

  node->hit_count.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void HotMemTable::SweepHits(std::unordered_map<std::string, uint32_t>* hit_map) {
  if (!hit_map) return;
  std::lock_guard<std::mutex> lock(index_mutex_);
  hit_map->reserve(index_.size());
  for (const auto& kv : index_) {
    HotNode* node = kv.second;
    uint32_t hits = node->hit_count.exchange(0, std::memory_order_relaxed);
    (*hit_map)[kv.first] = hits;
  }
}

InternalIterator* HotMemTable::NewIterator(Arena* arena) {
  if (arena) {
    void* mem = arena->AllocateAligned(sizeof(HotMemTableIterator));
    return new (mem) HotMemTableIterator(this);
  } else {
    return new HotMemTableIterator(this);
  }
}

size_t HotMemTable::KeyCount() const {
  std::lock_guard<std::mutex> lock(index_mutex_);
  return index_.size();
}

}  // namespace ROCKSDB_NAMESPACE
