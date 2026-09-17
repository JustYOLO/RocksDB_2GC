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

  // Build the new generation without touching the currently-published one.
  auto new_arena = std::make_unique<Arena>();
  auto new_bloom = std::make_unique<DynamicBloom>(new_arena.get(), total_bits);
  DynamicBloom* new_bloom_ptr = new_bloom.get();

  key_count_.store(0, std::memory_order_relaxed);
  // Publish the new bloom before releasing anything backing the old one.
  bloom_.store(new_bloom_ptr, std::memory_order_release);

  // Retire (never free while this router is alive) the previous generation;
  // see header comment.
  if (arena_) {
    retired_arenas_.push_back(std::move(arena_));
  }
  if (bloom_holder_) {
    retired_bloom_holders_.push_back(std::move(bloom_holder_));
  }
  arena_ = std::move(new_arena);
  bloom_holder_ = std::move(new_bloom);
}

void HotTableRouter::Disable() {
  bloom_.store(nullptr, std::memory_order_release);
  key_count_.store(0, std::memory_order_relaxed);
  // Retire (never free while this router is alive) the previous generation;
  // see header comment.
  if (arena_) {
    retired_arenas_.push_back(std::move(arena_));
  }
  if (bloom_holder_) {
    retired_bloom_holders_.push_back(std::move(bloom_holder_));
  }
}

}  // namespace ROCKSDB_NAMESPACE
