#include "db/space_saving_topk.h"

#include <algorithm>

#include "db/hot_table_policy.h"

namespace ROCKSDB_NAMESPACE {

SpaceSavingTopK::SpaceSavingTopK(size_t max_capacity, double decay_factor,
                                 double zero_hit_penalty)
    : max_capacity_(max_capacity > 0 ? max_capacity : 1024),
      decay_factor_(decay_factor > 0.0 && decay_factor < 1.0 ? decay_factor : 0.5),
      zero_hit_penalty_(zero_hit_penalty > 0.0 && zero_hit_penalty < 1.0 ? zero_hit_penalty : 0.25) {}

void SpaceSavingTopK::UpdateLocked(const std::string& key, uint64_t count,
                                   bool in_window) {
  const uint64_t observed = in_window ? count : 0;
  auto it = entries_.find(key);
  if (it != entries_.end()) {
    count_set_.erase({it->second.count, key});
    it->second.count += count;
    it->second.window_count += observed;
    count_set_.insert({it->second.count, key});
    return;
  }
  if (entries_.size() < max_capacity_) {
    entries_[key] = {key, count, 0, observed};
    count_set_.insert({count, key});
    return;
  }
  // Replace the minimum. The new key inherits its count as error, but not
  // its window activity.
  auto min_it = count_set_.begin();
  const uint64_t min_count = min_it->first;
  entries_.erase(min_it->second);
  count_set_.erase(min_it);
  entries_[key] = {key, min_count + count, min_count, observed};
  count_set_.insert({min_count + count, key});
}

void SpaceSavingTopK::Update(const Slice& key, uint64_t count, bool in_window) {
  if (count == 0) return;
  std::lock_guard<std::mutex> lock(mutex_);
  UpdateLocked(key.ToString(), count, in_window);
}

void SpaceSavingTopK::UpdateBatch(
    const std::vector<std::pair<std::string, uint64_t>>& runs, bool in_window) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& run : runs) {
    if (run.second > 0) {
      UpdateLocked(run.first, run.second, in_window);
    }
  }
}

void SpaceSavingTopK::AddHits(
    const std::unordered_map<std::string, uint32_t>& hits, bool in_window) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& kv : hits) {
    if (kv.second == 0) {
      continue;
    }
    const uint64_t observed = in_window ? kv.second : 0;
    auto it = entries_.find(kv.first);
    if (it != entries_.end()) {
      count_set_.erase({it->second.count, kv.first});
      it->second.count += kv.second;
      it->second.window_count += observed;
      count_set_.insert({it->second.count, kv.first});
    } else if (entries_.size() < max_capacity_) {
      entries_[kv.first] = {kv.first, kv.second, 0, observed};
      count_set_.insert({kv.second, kv.first});
    }
  }
}

double SpaceSavingTopK::PredictAbsorptionAndResetWindow(
    size_t max_keys, size_t budget_bytes, size_t per_key_bytes,
    uint64_t reserve_per_key, bool keep_reserve, uint64_t window_writes) {
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t covered = 0;
  size_t keys = 0;
  size_t used_bytes = 0;
  for (auto it = count_set_.rbegin(); it != count_set_.rend(); ++it) {
    if (keys >= max_keys || it->first < 2) {
      break;
    }
    // Same cutoff as the seeding loop in ColumnFamilyData::RebuildHotTable().
    const size_t node_bytes = per_key_bytes + it->second.size();
    if (used_bytes + node_bytes >= HotTableSeedLimit(budget_bytes, keys + 1,
                                                     reserve_per_key,
                                                     keep_reserve)) {
      break;
    }
    used_bytes += node_bytes;
    ++keys;
    auto entry_it = entries_.find(it->second);
    if (entry_it != entries_.end()) {
      covered += entry_it->second.window_count;
    }
  }
  for (auto& kv : entries_) {
    kv.second.window_count = 0;
  }
  if (window_writes == 0) {
    return 0.0;
  }
  return std::min(
      1.0, static_cast<double>(covered) / static_cast<double>(window_writes));
}

void SpaceSavingTopK::Age(
    const std::unordered_map<std::string, uint32_t>* swept_hits) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = entries_.begin(); it != entries_.end();) {
    bool idle = false;
    if (swept_hits != nullptr) {
      auto hit_it = swept_hits->find(it->first);
      idle = hit_it != swept_hits->end() && hit_it->second == 0;
    }
    const double factor = idle ? zero_hit_penalty_ : decay_factor_;
    it->second.count = static_cast<uint64_t>(it->second.count * factor);
    it->second.error = static_cast<uint64_t>(it->second.error * factor);
    if (it->second.count == 0) {
      it = entries_.erase(it);
    } else {
      ++it;
    }
  }
  RebuildCountSetLocked();
}

void SpaceSavingTopK::RebuildCountSetLocked() {
  count_set_.clear();
  for (const auto& kv : entries_) {
    count_set_.insert({kv.second.count, kv.first});
  }
}

std::vector<SpaceSavingEntry> SpaceSavingTopK::GetTopK(size_t k, uint64_t min_count) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<SpaceSavingEntry> list;
  size_t target_size = std::min(k, entries_.size());
  list.reserve(target_size);

  for (auto it = count_set_.rbegin(); it != count_set_.rend() && list.size() < target_size; ++it) {
    if (it->first < min_count) {
      break;  // Ordered descending: remaining entries all have count < min_count
    }
    auto map_it = entries_.find(it->second);
    if (map_it != entries_.end()) {
      list.push_back(map_it->second);
    }
  }
  return list;
}

size_t SpaceSavingTopK::QualifiedHeavyHittersCount(uint64_t min_count) const {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t count = 0;
  for (auto it = count_set_.rbegin(); it != count_set_.rend(); ++it) {
    if (it->first < min_count) {
      break;
    }
    count++;
  }
  return count;
}

uint32_t SpaceSavingTopK::ComputeScanProbeInterval(
    uint32_t stay_disabled_windows, uint32_t threshold_windows,
    uint32_t max_backoff_flushes) {
  if (stay_disabled_windows <= threshold_windows) {
    return 1;
  }
  uint32_t max_backoff = std::max<uint32_t>(1, max_backoff_flushes);
  return std::min(stay_disabled_windows, max_backoff);
}

void SpaceSavingTopK::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.clear();
  count_set_.clear();
}

size_t SpaceSavingTopK::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}

}  // namespace ROCKSDB_NAMESPACE
