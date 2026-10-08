#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rocksdb/slice.h"

namespace ROCKSDB_NAMESPACE {

struct SpaceSavingEntry {
  std::string key;
  uint64_t count{0};
  uint64_t error{0};
  // Activity observed in the current decision window only (never inherited
  // on eviction). Reset by PredictAbsorptionAndResetWindow().
  uint64_t window_count{0};
};

class SpaceSavingTopK {
 public:
  explicit SpaceSavingTopK(size_t max_capacity, double decay_factor = 0.5,
                           double zero_hit_penalty = 0.25);
  ~SpaceSavingTopK() = default;

  // Records `count` occurrences of `key`. `in_window`: whether they belong to
  // the current decision window (also added to window_count).
  void Update(const Slice& key, uint64_t count = 1, bool in_window = true);
  // Update() for each (key, count) run.
  void UpdateBatch(const std::vector<std::pair<std::string, uint64_t>>& runs,
                   bool in_window);
  // Adds swept HotTable write hits. Keys not tracked are inserted only while
  // there is room (hits never evict).
  void AddHits(const std::unordered_map<std::string, uint32_t>& hits,
               bool in_window);
  // Predicted absorption of a HotTable built now: the window_count of the
  // keys GetTopK(max_keys) would return, cut off where a HotTable of
  // `budget_bytes` would stop seeding (each key costs
  // `per_key_bytes + key.size()`, and seeding stops at HotTableSeedLimit()
  // with `reserve_per_key` and `keep_reserve`), divided by `window_writes`
  // (capped at 1). Resets window_count of every entry.
  double PredictAbsorptionAndResetWindow(size_t max_keys, size_t budget_bytes,
                                         size_t per_key_bytes,
                                         uint64_t reserve_per_key,
                                         bool keep_reserve,
                                         uint64_t window_writes);
  // Ages every entry once: keys in `swept_hits` with zero hits get the
  // zero-hit penalty, all others the decay factor. `swept_hits` may be null.
  void Age(const std::unordered_map<std::string, uint32_t>* swept_hits);

  // Return the top keys sorted by estimated frequency with minimum appearance threshold
  std::vector<SpaceSavingEntry> GetTopK(size_t k, uint64_t min_count = 2) const;

  // Count how many keys meet the minimum appearance threshold
  size_t QualifiedHeavyHittersCount(uint64_t min_count = 2) const;

  // Number of flushes between duplicate-key probes once the cost model has
  // decided to stay disabled for `stay_disabled_windows` consecutive
  // windows: 1 while that is at most `threshold_windows`, else
  // min(stay_disabled_windows, max_backoff_flushes), at least 1. Used by
  // FlushJob::WriteLevel0Table() to gate its (fused) duplicate-key scan.
  static uint32_t ComputeScanProbeInterval(uint32_t stay_disabled_windows,
                                           uint32_t threshold_windows,
                                           uint32_t max_backoff_flushes);

  // Clear tracker state
  void Clear();

  size_t Size() const;
  size_t Capacity() const { return max_capacity_; }

 private:
  // REQUIRES: mutex_ held.
  void UpdateLocked(const std::string& key, uint64_t count, bool in_window);
  // REQUIRES: mutex_ held.
  void RebuildCountSetLocked();

  const size_t max_capacity_;
  const double decay_factor_;
  const double zero_hit_penalty_;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, SpaceSavingEntry> entries_;
  std::set<std::pair<uint64_t, std::string>> count_set_;
};

}  // namespace ROCKSDB_NAMESPACE
