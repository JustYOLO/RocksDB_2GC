//  Copyright (c) 2026-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <map>
#include <mutex>
#include <vector>

#include "db/dbformat.h"
#include "rocksdb/comparator.h"
#include "rocksdb/slice.h"
#include "rocksdb/status.h"
#include "table/internal_iterator.h"

namespace ROCKSDB_NAMESPACE {

struct HotNode {
  std::atomic<uint32_t> hit_count{0};
  std::atomic<uint32_t> seq_version{0}; // Seqlock: Even = stable, Odd = writing
  uint32_t user_key_len{0};
  uint32_t val_len{0};
  ValueType value_type{kTypeValue};
  SequenceNumber seq{0};

  const char* UserKey() const {
    return reinterpret_cast<const char*>(this + 1);
  }
  char* ValBuf() {
    return const_cast<char*>(reinterpret_cast<const char*>(this + 1)) + user_key_len;
  }
  const char* ValBuf() const {
    return reinterpret_cast<const char*>(this + 1) + user_key_len;
  }
};

class HotMemTable {
 public:
  HotMemTable(const InternalKeyComparator& cmp, size_t write_buffer_size,
              uint32_t max_val_size);
  ~HotMemTable();

  // In-place update for an existing hot key.
  // Returns true if key was present and updated in-place; false if key was not found.
  bool UpdateInPlace(const Slice& user_key, const Slice& value, ValueType type,
                     SequenceNumber seq, uint64_t log_num = 0);

  // Add a new hot key node with over-provisioned value padding.
  bool Add(const Slice& user_key, const Slice& value, ValueType type,
           SequenceNumber seq, uint64_t log_num = 0);

  // Lock-free untorn read for a hot key.
  // Returns true if key was found in HotTable.
  bool Get(const Slice& user_key, std::string* value, Status* status,
           SequenceNumber* seq_found = nullptr);

  // Sweep and reset all atomic hit counters for Virtual Flush
  void SweepHits(std::unordered_map<std::string, uint32_t>* hit_map);

  // Create an InternalIterator for reading/flushing all entries
  InternalIterator* NewIterator(Arena* arena = nullptr);

  size_t ApproximateMemoryUsage() const {
    return allocated_bytes_.load(std::memory_order_relaxed);
  }

  size_t WriteBufferSize() const { return write_buffer_size_; }
  uint32_t MaxValueSize() const { return max_val_size_; }
  bool IsFull() const { return ApproximateMemoryUsage() >= write_buffer_size_; }
  size_t KeyCount() const;

  SequenceNumber GetEarliestSequenceNumber() const {
    return earliest_seq_.load(std::memory_order_relaxed);
  }

  uint64_t GetEarliestLogNumber() const {
    return earliest_log_num_.load(std::memory_order_relaxed);
  }

  void SetEarliestLogNumber(uint64_t log_num) {
    earliest_log_num_.store(log_num, std::memory_order_relaxed);
  }

  void ResetEarliestLogNumber(uint64_t log_num) {
    earliest_log_num_.store(log_num, std::memory_order_relaxed);
  }

 private:
  friend class HotMemTableIterator;

  struct KeyComparatorWrapper {
    const Comparator* user_cmp;
    bool operator()(const std::string& a, const std::string& b) const {
      return user_cmp->Compare(a, b) < 0;
    }
  };

  const InternalKeyComparator internal_comparator_;
  const size_t write_buffer_size_;
  const uint32_t max_val_size_;

  std::atomic<size_t> allocated_bytes_{0};
  std::atomic<SequenceNumber> earliest_seq_{kMaxSequenceNumber};
  std::atomic<uint64_t> earliest_log_num_{kMaxSequenceNumber};

  mutable std::mutex index_mutex_;
  std::map<std::string, HotNode*, KeyComparatorWrapper> index_;
  std::vector<void*> allocated_node_ptrs_;
};

}  // namespace ROCKSDB_NAMESPACE
