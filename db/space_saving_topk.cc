#include "db/space_saving_topk.h"
#include <algorithm>

namespace ROCKSDB_NAMESPACE {

SpaceSavingTopK::SpaceSavingTopK(size_t max_capacity, double decay_factor,
                                 double zero_hit_penalty)
    : max_capacity_(max_capacity > 0 ? max_capacity : 1024),
      decay_factor_(decay_factor > 0.0 && decay_factor < 1.0 ? decay_factor : 0.5),
      zero_hit_penalty_(zero_hit_penalty > 0.0 && zero_hit_penalty < 1.0 ? zero_hit_penalty : 0.25) {}

void SpaceSavingTopK::Update(const Slice& key, uint64_t count) {
  if (count == 0) return;
  std::lock_guard<std::mutex> lock(mutex_);
  std::string key_str = key.ToString();
  auto it = entries_.find(key_str);
  if (it != entries_.end()) {
    count_set_.erase({it->second.count, key_str});
    it->second.count += count;
    count_set_.insert({it->second.count, key_str});
  } else {
    if (entries_.size() < max_capacity_) {
      entries_[key_str] = {key_str, count, 0};
      count_set_.insert({count, key_str});
    } else {
      // Find element with minimum count in O(1) from count_set_
      auto min_it = count_set_.begin();
      uint64_t min_count = min_it->first;
      std::string evicted_key = min_it->second;
      count_set_.erase(min_it);
      entries_.erase(evicted_key);

      entries_[key_str] = {key_str, min_count + count, min_count};
      count_set_.insert({min_count + count, key_str});
    }
  }
}

void SpaceSavingTopK::ApplyDecayAndPenalties(
    const std::unordered_map<std::string, uint32_t>& hot_key_hits) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Apply hit updates from sweep first
  for (const auto& kv : hot_key_hits) {
    if (kv.second > 0) {
      auto it = entries_.find(kv.first);
      if (it != entries_.end()) {
        it->second.count += kv.second;
      } else {
        if (entries_.size() < max_capacity_) {
          entries_[kv.first] = {kv.first, kv.second, 0};
        }
      }
    }
  }

  // Apply global aging decay to all entries and zero-hit penalty
  for (auto it = entries_.begin(); it != entries_.end();) {
    auto hit_it = hot_key_hits.find(it->first);
    if (hit_it != hot_key_hits.end() && hit_it->second == 0) {
      // Zero hits in Hot Table during this epoch -> apply aggressive penalty
      it->second.count = static_cast<uint64_t>(it->second.count * zero_hit_penalty_);
      it->second.error = static_cast<uint64_t>(it->second.error * zero_hit_penalty_);
    } else {
      // Standard global aging
      it->second.count = static_cast<uint64_t>(it->second.count * decay_factor_);
      it->second.error = static_cast<uint64_t>(it->second.error * decay_factor_);
    }

    if (it->second.count == 0) {
      it = entries_.erase(it);
    } else {
      ++it;
    }
  }

  // Rebuild count_set_
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

void SpaceSavingTopK::RecordFlushWindow(uint64_t total_cold_entries,
                                       uint64_t duplicate_cold_entries,
                                       uint64_t hot_hits,
                                       uint64_t hot_misses) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (total_cold_entries > 0) {
    recent_duplicate_ratio_ =
        static_cast<double>(duplicate_cold_entries) / static_cast<double>(total_cold_entries);
  } else {
    recent_duplicate_ratio_ = 0.0;
  }

  uint64_t total_hot_ops = hot_hits + hot_misses;
  if (total_hot_ops > 0) {
    recent_absorption_ratio_ =
        static_cast<double>(hot_hits) / static_cast<double>(total_hot_ops);
  } else {
    recent_absorption_ratio_ = 0.0;
  }
}

double SpaceSavingTopK::GetRecentDuplicateRatio() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return recent_duplicate_ratio_;
}

double SpaceSavingTopK::GetRecentAbsorptionRatio() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return recent_absorption_ratio_;
}

uint32_t SpaceSavingTopK::GetConsecutiveFlatWindows() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return consecutive_flat_windows_;
}

uint32_t SpaceSavingTopK::GetConsecutiveSkewedWindows() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return consecutive_skewed_windows_;
}

bool SpaceSavingTopK::IsWorkloadSkewed(bool currently_active, double min_dup_ratio,
                                      double min_abs_ratio,
                                      uint32_t consecutive_threshold_windows) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (consecutive_threshold_windows == 0) {
    consecutive_threshold_windows = 1;
  }

  if (currently_active) {
    // When HotTable is active, check if either HotTable absorption ratio or cold duplicate ratio meets threshold
    bool healthy_skew = (recent_absorption_ratio_ >= min_abs_ratio) ||
                        (recent_duplicate_ratio_ >= min_dup_ratio);
    if (!healthy_skew) {
      consecutive_flat_windows_++;
      consecutive_skewed_windows_ = 0;
      if (consecutive_flat_windows_ >= consecutive_threshold_windows) {
        return false;  // Transition to disabled
      }
    } else {
      consecutive_flat_windows_ = 0;
      consecutive_skewed_windows_++;
    }
    return true;  // Remain active
  } else {
    // When HotTable is disabled, check if cold memtable exhibits skew (duplicates >= min_dup_ratio)
    bool detected_skew = (recent_duplicate_ratio_ >= min_dup_ratio);
    if (detected_skew) {
      consecutive_skewed_windows_++;
      consecutive_flat_windows_ = 0;
      if (consecutive_skewed_windows_ >= consecutive_threshold_windows) {
        return true;  // Transition to enabled
      }
    } else {
      consecutive_skewed_windows_ = 0;
      consecutive_flat_windows_++;
    }
    return false;  // Remain disabled
  }
}

void SpaceSavingTopK::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  entries_.clear();
  count_set_.clear();
  recent_duplicate_ratio_ = 0.0;
  recent_absorption_ratio_ = 0.0;
}

size_t SpaceSavingTopK::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return entries_.size();
}

}  // namespace ROCKSDB_NAMESPACE
