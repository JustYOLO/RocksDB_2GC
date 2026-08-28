#include "db/hot_table_router.h"
#include <algorithm>

namespace ROCKSDB_NAMESPACE {

HotTableRouter::HotTableRouter(size_t initial_capacity, uint32_t bits_per_key)
    : bits_per_key_(bits_per_key > 0 ? bits_per_key : 10) {
  Rebuild(initial_capacity);
}

void HotTableRouter::Add(const Slice& key) {
  DynamicBloom* bloom = bloom_.load(std::memory_order_acquire);
  if (bloom) {
    bloom->Add(key);
    key_count_.fetch_add(1, std::memory_order_relaxed);
  }
}

void HotTableRouter::AddConcurrently(const Slice& key) {
  DynamicBloom* bloom = bloom_.load(std::memory_order_acquire);
  if (bloom) {
    bloom->AddConcurrently(key);
    key_count_.fetch_add(1, std::memory_order_relaxed);
  }
}

void HotTableRouter::Rebuild(size_t capacity) {
  size_t cap = std::max<size_t>(capacity, 64);
  uint32_t total_bits = static_cast<uint32_t>(cap * bits_per_key_);
  if (total_bits < 64) total_bits = 64;

  key_count_.store(0, std::memory_order_relaxed);
  arena_ = std::make_unique<Arena>();
  auto new_bloom = std::make_unique<DynamicBloom>(arena_.get(), total_bits);
  bloom_.store(new_bloom.get(), std::memory_order_release);
  bloom_holder_ = std::move(new_bloom);
}

void HotTableRouter::Disable() {
  bloom_.store(nullptr, std::memory_order_release);
  bloom_holder_.reset();
  arena_.reset();
  key_count_.store(0, std::memory_order_relaxed);
}

}  // namespace ROCKSDB_NAMESPACE
