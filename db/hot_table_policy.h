//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <algorithm>
#include <cstdint>

#include "rocksdb/rocksdb_namespace.h"

namespace ROCKSDB_NAMESPACE {

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
  if (m.RebuildGain(a, a_hat) >= m.rebuild_cost) {
    return HotTableDecision::kRebuild;
  }
  if (a < m.DisableThreshold()) {
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
