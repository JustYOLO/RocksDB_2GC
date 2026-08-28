#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <mutex>
#include <memory>
#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

struct SpaceSavingEntry {
  std::string key;
  uint64_t count{0};
  uint64_t error{0};
};

class SpaceSavingTopK {
 public:
  explicit SpaceSavingTopK(size_t max_capacity, double decay_factor = 0.5,
                           double zero_hit_penalty = 0.25);
  ~SpaceSavingTopK() = default;

  // Record an observation/hit for a key
  void Update(const Slice& key, uint64_t count = 1);

  // Apply decay to all existing keys and zero-hit penalty to inactive hot keys
  void ApplyDecayAndPenalties(const std::unordered_map<std::string, uint32_t>& hot_key_hits);

  // Return the top keys sorted by estimated frequency with minimum appearance threshold
  std::vector<SpaceSavingEntry> GetTopK(size_t k, uint64_t min_count = 2) const;

  // Count how many keys meet the minimum appearance threshold
  size_t QualifiedHeavyHittersCount(uint64_t min_count = 2) const;

  // Record stats from cold memtable flush and HotTable activity
  void RecordFlushWindow(uint64_t total_cold_entries, uint64_t duplicate_cold_entries,
                         uint64_t hot_hits = 0, uint64_t hot_misses = 0);
  double GetRecentDuplicateRatio() const;
  double GetRecentAbsorptionRatio() const;
  uint32_t GetConsecutiveFlatWindows() const;
  uint32_t GetConsecutiveSkewedWindows() const;

  // Evaluate whether workload is skewed based on consecutive flush windows
  bool IsWorkloadSkewed(bool currently_active, double min_dup_ratio = 0.20,
                        double min_abs_ratio = 0.20,
                        uint32_t consecutive_threshold_windows = 2);

  // Clear tracker state
  void Clear();

  size_t Size() const;
  size_t Capacity() const { return max_capacity_; }

 private:
  const size_t max_capacity_;
  const double decay_factor_;
  const double zero_hit_penalty_;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, SpaceSavingEntry> entries_;
  std::set<std::pair<uint64_t, std::string>> count_set_;
  double recent_duplicate_ratio_{0.0};
  double recent_absorption_ratio_{0.0};
  uint32_t consecutive_flat_windows_{0};
  uint32_t consecutive_skewed_windows_{0};
};

}  // namespace ROCKSDB_NAMESPACE
