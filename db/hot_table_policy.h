//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <algorithm>
#include <cstdint>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

// --- HotTable seeding with a growth reserve -------------------------------
//
// A seeded HotTable node's value slot is sized for hot_table_max_value_size.
// A larger value reallocates the node, and a HotTable that runs out of room
// is sealed early. So seeding keeps `reserve_per_key` bytes of the budget free
// for each seeded key, and the seeded share of the budget stays within
// [kHotTableMinSeedPercent, kHotTableMaxSeedPercent] percent.

constexpr uint64_t kHotTableMinSeedPercent = 30;
constexpr uint64_t kHotTableMaxSeedPercent = 95;
// A HotTable that filled up only shows a lower bound of the growth it needed;
// the learned reserve is raised by this factor over that bound.
constexpr double kHotGrowthReserveFullBackoff = 1.5;

// floor(value * num / den) without overflowing for any uint64_t value.
inline uint64_t HotTableFractionOf(uint64_t value, uint64_t num, uint64_t den) {
  return value / den * num + value % den * num / den;
}

inline uint64_t HotTableMinSeedBytes(uint64_t budget) {
  return HotTableFractionOf(budget, kHotTableMinSeedPercent, 100);
}

inline uint64_t HotTableMaxSeedBytes(uint64_t budget) {
  return HotTableFractionOf(budget, kHotTableMaxSeedPercent, 100);
}

// Bytes the seeded nodes of a HotTable with `budget` bytes may take once
// `num_keys` keys are seeded, each keeping `reserve_per_key` bytes for value
// growth. With `keep_reserve` false: the whole budget (no reserve, no clamp).
inline uint64_t HotTableSeedLimit(uint64_t budget, uint64_t num_keys,
                                  uint64_t reserve_per_key, bool keep_reserve) {
  if (!keep_reserve) {
    return budget;
  }
  uint64_t room = 0;
  if (reserve_per_key == 0 || num_keys <= budget / reserve_per_key) {
    room = budget - num_keys * reserve_per_key;
  }
  return std::min(std::max(room, HotTableMinSeedBytes(budget)),
                  HotTableMaxSeedBytes(budget));
}

// Most keys HotTableSeedLimit() can admit when every seeded node costs at
// least `min_node_bytes`. Sizes the router and the top-K extraction, so it
// must not be below the number of keys seeding actually admits. At least 1.
inline uint64_t HotTableKeyCapacity(uint64_t budget, uint64_t min_node_bytes,
                                    uint64_t reserve_per_key,
                                    bool keep_reserve) {
  min_node_bytes = std::max<uint64_t>(min_node_bytes, 1);
  uint64_t keys = budget / min_node_bytes;
  if (keep_reserve) {
    const uint64_t with_reserve = budget / (min_node_bytes + reserve_per_key);
    const uint64_t at_least = HotTableMinSeedBytes(budget) / min_node_bytes;
    const uint64_t at_most = HotTableMaxSeedBytes(budget) / min_node_bytes;
    keys = std::min(std::max(with_reserve, at_least), at_most);
  }
  return std::max<uint64_t>(keys, 1);
}

// Outcome of one HotTable decision window (see DecideHotTable()).
enum class HotTableDecision {
  kEnable,        // disabled -> build a HotTable
  kStayDisabled,  // disabled -> keep it off
  kRebuild,       // active -> seal it and build a successor from current keys
  kDisable,       // active -> seal it with no successor
  kContinue,      // active -> keep it
};

// Cost model of the HotTable decision engine. All quantities are in
// memtable flushes or fractions of writes.
struct HotTableCostModel {
  uint32_t epoch_flushes = 1;    // E
  double rebuild_cost = 1.0;     // kappa
  uint64_t memtable_bytes = 0;   // C
  uint64_t wal_limit_bytes = 0;  // W; 0 = unknown
  // Whether an active HotTable may be rebuilt or disabled. false never
  // returns that decision (see hot_table_rebuild_on_prediction and
  // hot_table_disable_on_low_absorption).
  bool allow_rebuild = true;
  bool allow_disable = true;

  double EnableThreshold() const {
    const double e = std::max<uint32_t>(epoch_flushes, 1);
    return rebuild_cost / (e + rebuild_cost);
  }
  // kappa * C / W, or 0 when W is unknown.
  double RawDisableThreshold() const {
    if (wal_limit_bytes == 0) {
      return 0.0;
    }
    return rebuild_cost * static_cast<double>(memtable_bytes) /
           static_cast<double>(wal_limit_bytes);
  }
  // RawDisableThreshold() clamped to EnableThreshold(), so that a disable
  // never immediately follows an enable.
  double DisableThreshold() const {
    return std::min(RawDisableThreshold(), EnableThreshold());
  }
  bool DisableThresholdClamped() const {
    return RawDisableThreshold() > EnableThreshold();
  }
  // Flushes saved over the next window by rebuilding now, rather than
  // keeping a HotTable that absorbs `a`, if a rebuilt one absorbs `a_hat`.
  double RebuildGain(double a, double a_hat) const {
    if (a_hat <= a) {
      return 0.0;
    }
    if (a_hat >= 1.0) {
      return 1e18;
    }
    const double e = std::max<uint32_t>(epoch_flushes, 1);
    return e * (a_hat - a) / (1.0 - a_hat);
  }
};

// `a`: measured absorption of the active HotTable over the window (ignored
// when disabled). `a_hat`: predicted absorption of a HotTable built now.
inline HotTableDecision DecideHotTable(const HotTableCostModel& m,
                                       bool enabled, double a, double a_hat) {
  if (!enabled) {
    return a_hat >= m.EnableThreshold() ? HotTableDecision::kEnable
                                        : HotTableDecision::kStayDisabled;
  }
  if (m.allow_rebuild && m.RebuildGain(a, a_hat) >= m.rebuild_cost) {
    return HotTableDecision::kRebuild;
  }
  if (m.allow_disable && a < m.DisableThreshold()) {
    return HotTableDecision::kDisable;
  }
  return HotTableDecision::kContinue;
}

inline const char* HotTableDecisionName(HotTableDecision d) {
  switch (d) {
    case HotTableDecision::kEnable:
      return "enable";
    case HotTableDecision::kStayDisabled:
      return "stay_disabled";
    case HotTableDecision::kRebuild:
      return "rebuild";
    case HotTableDecision::kDisable:
      return "disable";
    case HotTableDecision::kContinue:
      return "continue";
  }
  return "unknown";
}

}  // namespace ROCKSDB_NAMESPACE
