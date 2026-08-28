#pragma once

#include <memory>
#include <atomic>
#include "rocksdb/slice.h"
#include "memory/arena.h"
#include "util/dynamic_bloom.h"

namespace ROCKSDB_NAMESPACE {

class HotTableRouter {
 public:
  explicit HotTableRouter(size_t initial_capacity = 65536, uint32_t bits_per_key = 10);
  ~HotTableRouter() = default;

  // O(1) lock-free check for fast rejection
  inline bool MayContain(const Slice& key) const {
    DynamicBloom* bloom = bloom_.load(std::memory_order_acquire);
    if (!bloom) return false;
    return bloom->MayContain(key);
  }

  // Add key to bloom filter
  void Add(const Slice& key);
  void AddConcurrently(const Slice& key);

  // Reset and rebuild the bloom filter for the next HotTable epoch
  void Rebuild(size_t capacity);

  // Disable bloom router and release memory
  void Disable();

  // Check if router has active bloom filter
  bool IsActive() const {
    return bloom_.load(std::memory_order_relaxed) != nullptr;
  }

  size_t KeyCount() const {
    return key_count_.load(std::memory_order_relaxed);
  }

 private:
  const uint32_t bits_per_key_;
  std::atomic<DynamicBloom*> bloom_{nullptr};
  std::atomic<size_t> key_count_{0};
  std::unique_ptr<Arena> arena_;
  std::unique_ptr<DynamicBloom> bloom_holder_;
};

}  // namespace ROCKSDB_NAMESPACE
