//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/db_impl/db_impl.h"
#include "db/hot_memtable.h"
#include "db/hot_table_flush_job.h"
#include "db/hot_table_router.h"
#include "db/job_context.h"
#include "logging/logging.h"
#include "monitoring/iostats_context_imp.h"
#include "test_util/sync_point.h"

namespace ROCKSDB_NAMESPACE {

void DBImpl::MaybeScheduleHotTableRebuild(ColumnFamilyData* cfd) {
  mutex_.AssertHeld();
  if (reject_new_background_jobs_ || shutting_down_.load(std::memory_order_acquire)) {
    return;
  }
  if (!cfd->ioptions().enable_hot_table || cfd->IsDropped()) {
    return;
  }
  std::shared_ptr<HotMemTable> hot_mem = cfd->hot_mem_shared();
  if (!hot_mem || !hot_mem->IsFull() || hot_mem->IsClosed()) {
    // Not actually full (a stale MarkHotRebuildNeeded()/periodic sweep), or
    // already closed by a rebuild that's in flight/just finished.
    return;
  }
  // Throttle: require at least one decay cycle (ExecuteVirtualFlush(), tied
  // to virtual_flush_interval_flushes cold flushes) since the last
  // dispatched physical rebuild for this CF. Without this, a capacity-only
  // trigger can refire almost immediately after a rebuild: reseeding the
  // (unchanged, since no decay ran) top-K candidate set alone can consume
  // most of hot_table_write_buffer_size, so hot_mem_->IsFull() often goes
  // true again within milliseconds -- a "reseed storm" of back-to-back
  // rebuilds that all reseed the same stale key set, wasting I/O and
  // (worse) repeatedly reopening the redirect-then-capture cutover window.
  // Waiting for a fresh decay cycle bounds physical-rebuild frequency to
  // roughly the cold-flush cadence, matching what the original synchronous
  // design implicitly had (physical flushes were only ever checked at cold
  // flush time).
  if (cfd->GetHotDecayEpoch() <= cfd->GetHotRebuildDecayEpoch()) {
    return;
  }
  if (!cfd->TryBeginHotTableRebuild()) {
    return;  // Another dispatch already has this CF's rebuild in flight.
  }
  cfd->SetHotRebuildDecayEpoch(cfd->GetHotDecayEpoch());

  cfd->Ref();
  bg_hot_table_rebuild_scheduled_++;
  auto* arg = new HotTableRebuildArg{this, cfd};
  // Same tag (`this`, i.e. TaskType::kDefault) as BGWorkFlush/BGWorkCompaction
  // so the blanket per-TaskType env_->UnSchedule() loop in
  // CancelAllBackgroundWork()/~DBImpl() already covers this job type with no
  // further changes needed there.
  env_->Schedule(&DBImpl::BGWorkHotTableRebuild, arg, Env::Priority::LOW,
                this, &DBImpl::UnscheduleHotTableRebuildCallback);
}

void DBImpl::HotTableRebuildCheck() {
  InstrumentedMutexLock l(&mutex_);
  for (auto cfd : *versions_->GetColumnFamilySet()) {
    if (cfd->IsDropped() || !cfd->ioptions().enable_hot_table) {
      continue;
    }
    // Cheap common case: nothing to do unless either the event-driven flag
    // was set by a write, or (belt-and-suspenders) the table is observed
    // full without the flag having been set.
    bool needs_check = cfd->ConsumeHotRebuildNeeded();
    if (!needs_check) {
      HotMemTable* hot_mem = cfd->hot_mem();
      needs_check = hot_mem && hot_mem->IsFull();
    }
    if (needs_check) {
      MaybeScheduleHotTableRebuild(cfd);
    }
  }
}

void DBImpl::BGWorkHotTableRebuild(void* arg) {
  std::unique_ptr<HotTableRebuildArg> ha(static_cast<HotTableRebuildArg*>(arg));
  IOSTATS_SET_THREAD_POOL_ID(Env::Priority::LOW);
  TEST_SYNC_POINT("DBImpl::BGWorkHotTableRebuild");
  ha->db_->BackgroundCallHotTableRebuild(ha->cfd_);
  TEST_SYNC_POINT("DBImpl::BGWorkHotTableRebuild:done");
}

void DBImpl::UnscheduleHotTableRebuildCallback(void* arg) {
  std::unique_ptr<HotTableRebuildArg> ha(static_cast<HotTableRebuildArg*>(arg));
  DBImpl* db = ha->db_;
  ColumnFamilyData* cfd = ha->cfd_;
  {
    InstrumentedMutexLock l(&db->mutex_);
    db->bg_hot_table_rebuild_scheduled_--;
    cfd->EndHotTableRebuild();
    if (cfd->UnrefAndTryDelete()) {
      // cfd was the last reference; already deleted -- do not touch it again.
    }
    db->bg_cv_.SignalAll();
  }
  TEST_SYNC_POINT("DBImpl::UnscheduleHotTableRebuildCallback");
}

void DBImpl::BackgroundCallHotTableRebuild(ColumnFamilyData* cfd) {
  JobContext job_context(next_job_id_.fetch_add(1), true);
  TEST_SYNC_POINT("DBImpl::BackgroundCallHotTableRebuild:start");

  InstrumentedMutexLock l(&mutex_);
  assert(bg_hot_table_rebuild_scheduled_);

  auto cleanup_and_return = [&]() {
    if (job_context.HaveSomethingToClean() ||
        job_context.HaveSomethingToDelete()) {
      job_context.Clean();
    }
    cfd->EndHotTableRebuild();
    bg_hot_table_rebuild_scheduled_--;
    if (cfd->UnrefAndTryDelete()) {
      // cfd was the last reference; already deleted -- do not touch it again.
    }
    bg_cv_.SignalAll();
  };

  if (shutting_down_.load(std::memory_order_acquire) || cfd->IsDropped()) {
    cleanup_and_return();
    return;
  }

  InitSnapshotContext(&job_context);

  std::shared_ptr<HotMemTable> hot_mem = cfd->hot_mem_shared();
  std::shared_ptr<HotTableRouter> hot_router = cfd->hot_router_shared();
  if (!hot_mem || !hot_mem->IsFull()) {
    // Raced with something else that already handled this CF's rebuild
    // (e.g. its own cold flush ran RebuildHotTable() first and reset
    // hot_mem_ to a fresh, not-full instance). Nothing to do.
    cleanup_and_return();
    return;
  }

  // Redirect-then-capture cutover: stop routing new writes to this
  // generation, then close it so any write already in flight against it is
  // guaranteed to finish before Close() returns. After this point no write
  // can land on hot_mem, so it is safe to iterate/flush from here on with no
  // further synchronization. See HotMemTable::Close()'s comment and the
  // design doc for the full correctness argument this replaces (the old
  // DB-wide hot_stall_token stop-token approach).
  if (hot_router) {
    hot_router->Disable();
  }
  hot_mem->Close();

  const MutableCFOptions mutable_cf_options_copy =
      cfd->GetLatestMutableCFOptions();
  CompressionType output_compression =
      GetCompressionFlush(cfd->ioptions(), mutable_cf_options_copy);

  Status s;
  {
    HotTableFlushJob job(dbname_, cfd, immutable_db_options_,
                         mutable_cf_options_copy, file_options_for_compaction_,
                         versions_.get(), &mutex_, hot_mem,
                         directories_.GetDbDir(), GetDataDir(cfd, 0U),
                         output_compression, stats_, &event_logger_,
                         io_tracer_, db_id_, db_session_id_,
                         job_context.job_id, &job_context);
    s = job.Run();
  }

  if (s.ok()) {
    // Publishes a fresh, empty HotMemTable + reseeded HotTableRouter for
    // this CF and decides whether the cold path keeps its bonus memtable
    // slot -- entirely unchanged, existing logic (see column_family.cc).
    // flush_log_number=0 is deliberate: this job's VersionEdit never calls
    // SetLogNumber(), so cfd_->GetLogNumber() is untouched by it, and
    // RebuildHotTable()'s own fallback (seed from GetLogNumber() when
    // flush_log_number == 0) is exactly correct here.
    //
    // Hand off the single-flight guard we've held since dispatch: release
    // it right here (still holding mutex_ continuously, so no other thread
    // can observe it free until RebuildHotTable() re-acquires it a moment
    // later) so RebuildHotTable()'s own TryBeginHotTableRebuild() succeeds
    // instead of seeing "already held by myself" and bailing out. Passing
    // &mutex_ lets it unlock around its own expensive reseed section
    // (GetTopK() + up to ~800K HotMemTable::Add() calls) instead of that
    // running while every other write/flush/compaction in the DB is
    // blocked on mutex_ -- see the comment on RebuildHotTable() itself.
    cfd->EndHotTableRebuild();
    cfd->RebuildHotTable(/*was_physically_flushed=*/true,
                         /*flush_log_number=*/0, &mutex_);
  } else if (!s.IsShutdownInProgress() && !s.IsColumnFamilyDropped()) {
    // Not treated as a DB-wide background error: the data this job was
    // flushing is still safely durable in the WAL (HotTable writes go
    // through WAL like any other write), so nothing is lost. The safe
    // fallback is simply to leave this CF's HotTable closed/disabled
    // (writes keep falling through to the cold path, which is always
    // correct, just not "hot") rather than retrying or halting the DB.
    ROCKS_LOG_ERROR(immutable_db_options_.info_log,
                    "[%s] [JOB %d] Standalone HotTable flush failed, "
                    "leaving HotTable disabled for this column family: %s",
                    cfd->GetName().c_str(), job_context.job_id,
                    s.ToString().c_str());
  }

  cleanup_and_return();
}

}  // namespace ROCKSDB_NAMESPACE
