#include "db/spatial_cms.h"

#include <algorithm>
#include <cmath>

#include "util/hash.h"

namespace ROCKSDB_NAMESPACE {

namespace {
constexpr uint64_t kSeeds[SpatialCountMinSketch::kDepth] = {
    0xbc9f1d34ull, 0x517cc1b7ull, 0x9e3779b9ull, 0x85ebca6bull};
}  // namespace

SpatialCountMinSketch::SpatialCountMinSketch(size_t prefix_len)
    : prefix_len_(prefix_len),
      table_(new std::atomic<uint16_t>[kDepth * kWidth]) {
  Reset();
}

void SpatialCountMinSketch::Reset() {
  for (size_t i = 0; i < kDepth * kWidth; ++i) {
    table_[i].store(0, std::memory_order_relaxed);
  }
  total_evictions_.store(0, std::memory_order_relaxed);
  is_skewed_.store(false, std::memory_order_release);
  last_skew_ratio_.store(0.0, std::memory_order_relaxed);
}

void SpatialCountMinSketch::AddPrefix(const Slice& key) {
  if (key.empty()) {
    return;
  }
  Slice prefix(key.data(), std::min(key.size(), prefix_len_));
  for (size_t d = 0; d < kDepth; ++d) {
    uint64_t hash = GetSliceNPHash64(prefix, kSeeds[d]);
    size_t col = static_cast<size_t>(hash & width_mask_);
    size_t idx = d * kWidth + col;
    table_[idx].fetch_add(1, std::memory_order_relaxed);
  }
  total_evictions_.fetch_add(1, std::memory_order_relaxed);
}

uint16_t SpatialCountMinSketch::Estimate(const Slice& key) const {
  if (key.empty()) {
    return 0;
  }
  Slice prefix(key.data(), std::min(key.size(), prefix_len_));
  uint16_t min_count = std::numeric_limits<uint16_t>::max();
  for (size_t d = 0; d < kDepth; ++d) {
    uint64_t hash = GetSliceNPHash64(prefix, kSeeds[d]);
    size_t col = static_cast<size_t>(hash & width_mask_);
    size_t idx = d * kWidth + col;
    uint16_t val = table_[idx].load(std::memory_order_relaxed);
    min_count = std::min(min_count, val);
  }
  return min_count;
}

void SpatialCountMinSketch::CoolPrefix(const Slice& key, uint16_t amount) {
  if (key.empty()) {
    return;
  }
  Slice prefix(key.data(), std::min(key.size(), prefix_len_));
  for (size_t d = 0; d < kDepth; ++d) {
    uint64_t hash = GetSliceNPHash64(prefix, kSeeds[d]);
    size_t col = static_cast<size_t>(hash & width_mask_);
    size_t idx = d * kWidth + col;
    uint16_t val = table_[idx].load(std::memory_order_relaxed);
    while (val > 0) {
      uint16_t new_val = (val >= amount) ? (val - amount) : 0;
      if (table_[idx].compare_exchange_weak(val, new_val,
                                            std::memory_order_relaxed)) {
        break;
      }
    }
  }
}

bool SpatialCountMinSketch::HasHotKeyInRange(const Slice& smallest,
                                             const Slice& largest,
                                             uint16_t threshold) const {
  if (threshold == 0) {
    return true;
  }
  if (!smallest.empty() && Estimate(smallest) >= threshold) {
    return true;
  }
  if (!largest.empty() && Estimate(largest) >= threshold) {
    return true;
  }
  return false;
}

void SpatialCountMinSketch::DecayAndEvaluateSkew(double min_skew_threshold) {
  uint64_t sum_val = 0;
  uint16_t max_val = 0;
  size_t total_buckets = kDepth * kWidth;

  for (size_t i = 0; i < total_buckets; ++i) {
    uint16_t current = table_[i].load(std::memory_order_relaxed);
    uint16_t decayed = current >> 1;
    table_[i].store(decayed, std::memory_order_relaxed);

    sum_val += decayed;
    if (decayed > max_val) {
      max_val = decayed;
    }
  }

  double avg_val = static_cast<double>(sum_val) / static_cast<double>(total_buckets);
  double skew_ratio = (avg_val > 0.0) ? (static_cast<double>(max_val) / avg_val) : 0.0;

  last_skew_ratio_.store(skew_ratio, std::memory_order_relaxed);
  is_skewed_.store(min_skew_threshold <= 0.0 || skew_ratio >= min_skew_threshold,
                   std::memory_order_release);
}

}  // namespace ROCKSDB_NAMESPACE
