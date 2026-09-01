#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rocksdb/rocksdb_namespace.h"
#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

// Lock-Free Spatial Count-Min Sketch for tracking cache block eviction frequencies
// over spatial key ranges/prefixes.
class SpatialCountMinSketch {
 public:
  static constexpr size_t kDepth = 4;
  static constexpr size_t kWidth = 2048;  // Power of 2 (2048) for bitwise index masking

  explicit SpatialCountMinSketch(size_t prefix_len = 8);
  ~SpatialCountMinSketch() = default;

  // Non-copyable, non-movable
  SpatialCountMinSketch(const SpatialCountMinSketch&) = delete;
  SpatialCountMinSketch& operator=(const SpatialCountMinSketch&) = delete;

  // Add an eviction event for the given key/prefix (Lock-free O(1))
  void AddPrefix(const Slice& key);

  // Estimate the eviction frequency for the given key/prefix (Lock-free O(1))
  uint16_t Estimate(const Slice& key) const;

  // Cool down / decrement the eviction heat for a key/prefix upon retention
  void CoolPrefix(const Slice& key, uint16_t amount = 1);

  // Check if smallest or largest key prefix in a file has eviction heat >= threshold
  bool HasHotKeyInRange(const Slice& smallest, const Slice& largest,
                        uint16_t threshold) const;

  // Age out historical evictions by halving all counters and evaluate workload skew
  void DecayAndEvaluateSkew(double min_skew_threshold);

  // Check if current workload is evaluated as skewed (Relaxed atomic load)
  bool IsWorkloadSkewed() const {
    return is_skewed_.load(std::memory_order_acquire);
  }

  // Force enable/disable skew override (useful for testing or manual control)
  void SetWorkloadSkewed(bool skewed) {
    is_skewed_.store(skewed, std::memory_order_release);
  }

  double GetLastSkewRatio() const {
    return last_skew_ratio_.load(std::memory_order_relaxed);
  }

  uint64_t GetTotalEvictions() const {
    return total_evictions_.load(std::memory_order_relaxed);
  }

  size_t prefix_len() const { return prefix_len_; }
  void set_prefix_len(size_t prefix_len) { prefix_len_ = prefix_len; }

  // Reset all counters to 0
  void Reset();

 private:
  const size_t width_mask_{kWidth - 1};
  size_t prefix_len_{8};

  std::unique_ptr<std::atomic<uint16_t>[]> table_;
  std::atomic<uint64_t> total_evictions_{0};
  std::atomic<bool> is_skewed_{false};
  std::atomic<double> last_skew_ratio_{0.0};
};

}  // namespace ROCKSDB_NAMESPACE
