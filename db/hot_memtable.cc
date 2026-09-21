//  Copyright (c) 2026-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/hot_memtable.h"
#include <cstdlib>
#include <cstring>
#include <thread>
#include "memory/arena.h"
#include "test_util/sync_point.h"

namespace ROCKSDB_NAMESPACE {

class HotMemTableIterator : public InternalIterator {
 public:
  HotMemTableIterator(const HotMemTable* table)
      : table_(table),
        user_cmp_(table->internal_comparator_.user_comparator()),
        status_(Status::OK()) {
    std::shared_lock<std::shared_mutex> lock(table_->index_rwlock_);
    entries_.reserve(table_->index_.size());
    for (const auto& kv : table_->index_) {
      if (kv.second->seq > 0) {
        entries_.push_back(kv.second);
      }
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

    int left = 0;
    int right = static_cast<int>(entries_.size()) - 1;
    int best = -1;

    while (left <= right) {
      int mid = left + (right - left) / 2;
      HotNode* node = entries_[mid];
      Slice node_key(node->UserKey(), node->user_key_len);
      if (s.ok()) {
        int c = user_cmp_->Compare(node_key, pikey.user_key);
        if (c < 0) {
          left = mid + 1;
        } else if (c > 0) {
          best = mid;
          right = mid - 1;
        } else {
          // user_key matches. Check sequence number: higher seq is smaller in
          // internal order.
          if (node->seq > pikey.sequence) {
            // node < target, so node is before target
            left = mid + 1;
          } else {
            best = mid;
            right = mid - 1;
          }
        }
      } else {
        int c = user_cmp_->Compare(node_key, target);
        if (c >= 0) {
          best = mid;
          right = mid - 1;
        } else {
          left = mid + 1;
        }
      }
    }
    idx_ = best;
    UpdateCurrent();
  }

  void SeekForPrev(const Slice& target) override {
    ParsedInternalKey pikey;
    Status s = ParseInternalKey(target, &pikey, false /* log_err_key */);

    int left = 0;
    int right = static_cast<int>(entries_.size()) - 1;
    int best = -1;

    while (left <= right) {
      int mid = left + (right - left) / 2;
      HotNode* node = entries_[mid];
      Slice node_key(node->UserKey(), node->user_key_len);
      if (s.ok()) {
        int c = user_cmp_->Compare(node_key, pikey.user_key);
        if (c < 0) {
          best = mid;
          left = mid + 1;
        } else if (c > 0) {
          right = mid - 1;
        } else {
          // user_key matches. If node->seq >= pikey.sequence, node <= target in
          // internal key order.
          if (node->seq >= pikey.sequence) {
            best = mid;
            left = mid + 1;
          } else {
            right = mid - 1;
          }
        }
      } else {
        int c = user_cmp_->Compare(node_key, target);
        if (c <= 0) {
          best = mid;
          left = mid + 1;
        } else {
          right = mid - 1;
        }
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
  size_t EntryCount() const { return entries_.size(); }

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
  std::unique_lock<std::shared_mutex> lock(index_rwlock_);
  for (void* ptr : allocated_node_ptrs_) {
    free(ptr);
  }
}

void HotMemTable::Close() {
  closed_.store(true, std::memory_order_release);
  // Barrier: cannot be granted while any in-flight Add()/UpdateInPlace()
  // still holds index_rwlock_ (shared or exclusive), and every such call
  // rechecks closed_ immediately after acquiring it. See the closed_ comment
  // in hot_memtable.h and the fast-path lock-scope comment in
  // UpdateInPlace() for the full argument.
  std::unique_lock<std::shared_mutex> drain(index_rwlock_);
}

bool HotMemTable::UpdateInPlace(const Slice& user_key, const Slice& value,
                               ValueType type, SequenceNumber seq,
                               uint64_t log_num) {
  if (closed_.load(std::memory_order_acquire)) {
    return false;
  }

  // Fast path: look up the node and mutate it under a single continuous
  // shared_lock scope (rather than releasing the lock between lookup and
  // mutation). This is required for Close() to be a correct barrier: Close()
  // drains by briefly taking index_rwlock_ in exclusive mode, which cannot be
  // granted while any shared_lock holder (i.e. any in-flight fast-path
  // writer, from the closed_ recheck below through the end of this scope) is
  // still active. Holding the shared lock this long does not add contention
  // between concurrent fast-path writers to different keys -- shared holders
  // never block each other -- it only makes Close()/Add() (which need the
  // exclusive lock) wait for in-flight fast-path writers to finish, which is
  // exactly the drain semantics Close() needs.
  {
    std::shared_lock<std::shared_mutex> lock(index_rwlock_);
    if (closed_.load(std::memory_order_acquire)) {
      return false;
    }
    auto it = index_.find(user_key.ToString());
    if (it == index_.end()) {
      return false;
    }
    HotNode* node = it->second;
    TEST_SYNC_POINT(
        "HotMemTable::UpdateInPlace:FastPath:HoldingSharedLock");

    std::lock_guard<SpinMutex> node_lock(node->write_lock);
    if (seq <= node->seq && node->seq > 0) {
      return true;
    }

    if (value.size() <= node->capacity) {
      uint32_t v = node->seq_version.load(std::memory_order_relaxed);
      node->seq_version.store(v + 1, std::memory_order_release);

      node->val_len = static_cast<uint32_t>(value.size());
      node->value_type = type;
      node->seq = seq;
      if (node->val_len > 0) {
        memcpy(node->ValBuf(), value.data(), node->val_len);
      }

      node->seq_version.store(v + 2, std::memory_order_release);
      node->hit_count.fetch_add(1, std::memory_order_relaxed);

      if (seq > 0) {
        SequenceNumber cur_earliest =
            earliest_seq_.load(std::memory_order_relaxed);
        while (seq < cur_earliest &&
               !earliest_seq_.compare_exchange_weak(cur_earliest, seq)) {
        }
      }

      if (log_num > 0) {
        uint64_t cur_earliest_log =
            earliest_log_num_.load(std::memory_order_relaxed);
        while (log_num < cur_earliest_log &&
               !earliest_log_num_.compare_exchange_weak(cur_earliest_log,
                                                        log_num)) {
        }
      }

      return true;
    }
  }

  // Slow path: value.size() > node->capacity. Dynamically reallocate a larger
  // HotNode.
  std::unique_lock<std::shared_mutex> lock(index_rwlock_);
  if (closed_.load(std::memory_order_acquire)) {
    return false;
  }
  std::string key_str = user_key.ToString();
  auto it = index_.find(key_str);
  if (it == index_.end()) {
    return false;
  }
  HotNode* node = it->second;

  std::lock_guard<SpinMutex> node_lock(node->write_lock);
  if (seq <= node->seq && node->seq > 0) {
    return true;
  }

  if (value.size() <= node->capacity) {
    uint32_t v = node->seq_version.load(std::memory_order_relaxed);
    node->seq_version.store(v + 1, std::memory_order_release);

    node->val_len = static_cast<uint32_t>(value.size());
    node->value_type = type;
    node->seq = seq;
    if (node->val_len > 0) {
      memcpy(node->ValBuf(), value.data(), node->val_len);
    }

    node->seq_version.store(v + 2, std::memory_order_release);
    node->hit_count.fetch_add(1, std::memory_order_relaxed);
  } else {
    uint32_t new_cap =
        std::max(static_cast<uint32_t>(value.size()),
                 node->capacity > 0 ? node->capacity * 2 : max_val_size_);
    size_t new_alloc_size = sizeof(HotNode) + node->user_key_len + new_cap;
    void* raw = malloc(new_alloc_size);
    if (!raw) {
      return false;
    }

    HotNode* new_node = new (raw) HotNode();
    new_node->user_key_len = node->user_key_len;
    new_node->val_len = static_cast<uint32_t>(value.size());
    new_node->capacity = new_cap;
    new_node->value_type = type;
    new_node->seq = seq;
    new_node->hit_count.store(
        node->hit_count.load(std::memory_order_relaxed) + 1,
        std::memory_order_relaxed);

    memcpy(const_cast<char*>(new_node->UserKey()), node->UserKey(),
           node->user_key_len);
    if (new_node->val_len > 0) {
      memcpy(new_node->ValBuf(), value.data(), new_node->val_len);
    }

    it->second = new_node;
    allocated_node_ptrs_.push_back(raw);
    allocated_bytes_.fetch_add(new_alloc_size, std::memory_order_relaxed);
    // Do not touch `node` (the old, now-detached node) beyond this point: any
    // reader that captured this pointer before the swap above may still be
    // mid-seqlock-read on it. new_node already carries the correct seq (set
    // above at construction); mutating the old node's seq/seq_version here
    // would let such a reader observe a self-consistent seqlock snapshot
    // pairing the *new* seq with the *stale* (pre-resize) value bytes.
  }

  if (seq > 0) {
    SequenceNumber cur_earliest = earliest_seq_.load(std::memory_order_relaxed);
    while (seq < cur_earliest &&
           !earliest_seq_.compare_exchange_weak(cur_earliest, seq)) {
    }
  }

  if (log_num > 0) {
    uint64_t cur_earliest_log = earliest_log_num_.load(std::memory_order_relaxed);
    while (log_num < cur_earliest_log &&
           !earliest_log_num_.compare_exchange_weak(cur_earliest_log, log_num)) {}
  }

  return true;
}

bool HotMemTable::Add(const Slice& user_key, const Slice& value, ValueType type,
                      SequenceNumber seq, uint64_t log_num) {
  if (closed_.load(std::memory_order_acquire)) {
    return false;
  }
  std::unique_lock<std::shared_mutex> lock(index_rwlock_);
  if (closed_.load(std::memory_order_acquire)) {
    return false;
  }
  std::string key_str = user_key.ToString();
  auto it = index_.find(key_str);
  if (it != index_.end()) {
    HotNode* node = it->second;
    std::lock_guard<SpinMutex> node_lock(node->write_lock);
    if (seq <= node->seq && node->seq > 0) {
      return true;
    }
    if (value.size() <= node->capacity) {
      uint32_t v = node->seq_version.load(std::memory_order_relaxed);
      node->seq_version.store(v + 1, std::memory_order_release);
      node->val_len = static_cast<uint32_t>(value.size());
      node->value_type = type;
      node->seq = seq;
      if (node->val_len > 0) {
        memcpy(node->ValBuf(), value.data(), node->val_len);
      }
      node->seq_version.store(v + 2, std::memory_order_release);
    } else {
      uint32_t new_cap =
          std::max(static_cast<uint32_t>(value.size()),
                   node->capacity > 0 ? node->capacity * 2 : max_val_size_);
      size_t new_alloc_size = sizeof(HotNode) + node->user_key_len + new_cap;
      void* raw = malloc(new_alloc_size);
      if (!raw) return false;

      HotNode* new_node = new (raw) HotNode();
      new_node->user_key_len = node->user_key_len;
      new_node->val_len = static_cast<uint32_t>(value.size());
      new_node->capacity = new_cap;
      new_node->value_type = type;
      new_node->seq = seq;
      memcpy(const_cast<char*>(new_node->UserKey()), node->UserKey(),
             node->user_key_len);
      if (new_node->val_len > 0) {
        memcpy(new_node->ValBuf(), value.data(), new_node->val_len);
      }
      it->second = new_node;
      allocated_node_ptrs_.push_back(raw);
      allocated_bytes_.fetch_add(new_alloc_size, std::memory_order_relaxed);
      // See UpdateInPlace(): do not mutate the old, now-detached node.
    }

    if (seq > 0) {
      SequenceNumber cur_earliest =
          earliest_seq_.load(std::memory_order_relaxed);
      while (seq < cur_earliest &&
             !earliest_seq_.compare_exchange_weak(cur_earliest, seq)) {
      }
    }
    if (log_num > 0) {
      uint64_t cur_earliest_log = earliest_log_num_.load(std::memory_order_relaxed);
      while (log_num < cur_earliest_log &&
             !earliest_log_num_.compare_exchange_weak(cur_earliest_log, log_num)) {}
    }
    return true;
  }

  uint32_t initial_cap =
      std::max(max_val_size_, static_cast<uint32_t>(value.size()));
  size_t alloc_size = sizeof(HotNode) + user_key.size() + initial_cap;
  void* raw = malloc(alloc_size);
  if (!raw) return false;

  HotNode* node = new (raw) HotNode();
  node->user_key_len = static_cast<uint32_t>(user_key.size());
  node->val_len = static_cast<uint32_t>(value.size());
  node->capacity = initial_cap;
  node->value_type = type;
  node->seq = seq;

  memcpy(const_cast<char*>(node->UserKey()), user_key.data(), user_key.size());
  if (node->val_len > 0) {
    memcpy(node->ValBuf(), value.data(), node->val_len);
  }

  index_[key_str] = node;
  allocated_node_ptrs_.push_back(raw);
  allocated_bytes_.fetch_add(alloc_size, std::memory_order_relaxed);

  if (seq > 0) {
    SequenceNumber cur_earliest = earliest_seq_.load(std::memory_order_relaxed);
    while (seq < cur_earliest &&
           !earliest_seq_.compare_exchange_weak(cur_earliest, seq)) {
    }
  }

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
    std::shared_lock<std::shared_mutex> lock(index_rwlock_);
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

    if (node->seq == 0) {
      // Pre-populated key that has not yet been written to in HotMemTable
      v2 = node->seq_version.load(std::memory_order_acquire);
      if (v1 == v2) return false;
      continue;
    }

    if (node->value_type == kTypeDeletion ||
        node->value_type == kTypeSingleDeletion) {
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
  std::shared_lock<std::shared_mutex> lock(index_rwlock_);
  hit_map->reserve(index_.size());
  for (const auto& kv : index_) {
    HotNode* node = kv.second;
    uint32_t hits = node->hit_count.exchange(0, std::memory_order_relaxed);
    (*hit_map)[kv.first] = hits;
  }
}

InternalIterator* HotMemTable::NewIterator(Arena* arena, size_t* out_key_count) {
  HotMemTableIterator* iter = nullptr;
  if (arena) {
    void* mem = arena->AllocateAligned(sizeof(HotMemTableIterator));
    iter = new (mem) HotMemTableIterator(this);
  } else {
    iter = new HotMemTableIterator(this);
  }
  if (out_key_count) {
    *out_key_count = iter->EntryCount();
  }
  return iter;
}

size_t HotMemTable::KeyCount() const {
  std::shared_lock<std::shared_mutex> lock(index_rwlock_);
  size_t count = 0;
  for (const auto& kv : index_) {
    if (kv.second->seq > 0) {
      count++;
    }
  }
  return count;
}

}  // namespace ROCKSDB_NAMESPACE
