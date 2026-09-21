//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/hot_table_flush_job.h"

#include <vector>

#include "db/builder.h"
#include "db/column_family.h"
#include "db/hot_memtable.h"
#include "db/job_context.h"
#include "db/version_edit.h"
#include "db/version_set.h"
#include "logging/event_logger.h"
#include "logging/logging.h"
#include "memory/arena.h"
#include "monitoring/instrumented_mutex.h"
#include "monitoring/statistics_impl.h"
#include "table/table_builder.h"

namespace ROCKSDB_NAMESPACE {

HotTableFlushJob::HotTableFlushJob(
    const std::string& dbname, ColumnFamilyData* cfd,
    const ImmutableDBOptions& db_options,
    const MutableCFOptions& mutable_cf_options,
    const FileOptions& file_options, VersionSet* versions,
    InstrumentedMutex* db_mutex, std::shared_ptr<HotMemTable> hot_mem,
    FSDirectory* db_directory, FSDirectory* output_file_directory,
    CompressionType output_compression, Statistics* stats,
    EventLogger* event_logger,
    const std::shared_ptr<IOTracer>& io_tracer, const std::string& db_id,
    const std::string& db_session_id, int job_id, JobContext* job_context)
    : dbname_(dbname),
      cfd_(cfd),
      db_options_(db_options),
      mutable_cf_options_(mutable_cf_options),
      file_options_(file_options),
      versions_(versions),
      db_mutex_(db_mutex),
      hot_mem_(std::move(hot_mem)),
      db_directory_(db_directory),
      output_file_directory_(output_file_directory),
      output_compression_(output_compression),
      stats_(stats),
      event_logger_(event_logger),
      io_tracer_(io_tracer),
      db_id_(db_id),
      db_session_id_(db_session_id),
      job_id_(job_id),
      job_context_(job_context) {}

Status HotTableFlushJob::Run() {
  db_mutex_->AssertHeld();
  assert(hot_mem_);
  assert(hot_mem_->IsClosed());
  assert(job_context_);

  FileMetaData hot_meta;
  hot_meta.fd = FileDescriptor(versions_->NewFileNumber(), 0, 0);
  hot_meta.epoch_number = cfd_->NewEpochNumber();
  hot_meta.temperature = mutable_cf_options_.default_write_temperature;

  Status s;
  IOStatus io_s;
  TableProperties table_properties;
  uint64_t memtable_payload_bytes = 0;
  uint64_t memtable_garbage_bytes = 0;
  InternalStats::CompactionStats flush_stats(CompactionReason::kFlush,
                                             1 /* count */);
  std::vector<BlobFileAddition> blob_file_additions;
  size_t hot_key_count = 0;

  {
    db_mutex_->Unlock();

    SystemClock* clock = db_options_.clock;
    int64_t current_time_raw = 0;
    Status time_status = clock->GetCurrentTime(&current_time_raw);
    if (!time_status.ok()) {
      ROCKS_LOG_WARN(db_options_.info_log,
                     "Failed to get current time to populate creation_time "
                     "property for standalone HotTable flush. Status: %s",
                     time_status.ToString().c_str());
    }
    const uint64_t current_time = static_cast<uint64_t>(current_time_raw);
    hot_meta.oldest_ancester_time = current_time;
    hot_meta.file_creation_time = current_time;

    Arena arena;
    ScopedArenaPtr<InternalIterator> hot_iter(
        hot_mem_->NewIterator(&arena, &hot_key_count));

    const std::string full_history_ts_low = cfd_->GetFullHistoryTsLow();
    const std::string* const full_history_ts_low_ptr =
        full_history_ts_low.empty() ? nullptr : &full_history_ts_low;

    ReadOptions read_options(Env::IOActivity::kFlush);
    const WriteOptions write_options(Env::IOActivity::kFlush);
    TableBuilderOptions tboptions(
        cfd_->ioptions(), mutable_cf_options_, read_options, write_options,
        cfd_->internal_comparator(), cfd_->internal_tbl_prop_coll_factories(),
        output_compression_, mutable_cf_options_.compression_opts,
        cfd_->GetID(), cfd_->GetName(), 0 /* level */, current_time, false,
        TableFileCreationReason::kFlush,
        static_cast<int64_t>(hot_meta.oldest_ancester_time), current_time,
        db_id_, db_session_id_, 0 /* target_file_size */,
        hot_meta.fd.GetNumber(),
        kMaxSequenceNumber /* preclude_last_level_min_seqno: tiering's
                              preclude-last-level feature does not apply to
                              this standalone job */);

    s = BuildTable(
        dbname_, versions_, db_options_, tboptions, file_options_,
        cfd_->table_cache(), hot_iter.get(),
        std::vector<std::unique_ptr<FragmentedRangeTombstoneIterator>>(),
        &hot_meta, &blob_file_additions, job_context_->snapshot_seqs,
        job_context_->GetEarliestSnapshotSequence(),
        job_context_->earliest_write_conflict_snapshot,
        job_context_->GetJobSnapshotSequence(),
        job_context_->snapshot_checker,
        mutable_cf_options_.paranoid_file_checks, cfd_->internal_stats(),
        &io_s, io_tracer_, BlobFileCreationReason::kFlush,
        nullptr /* seqno_to_time_mapping */, event_logger_, job_id_,
        &table_properties, Env::WLTH_NOT_SET, full_history_ts_low_ptr,
        nullptr /* blob_callback */, nullptr /* version */,
        &memtable_payload_bytes, &memtable_garbage_bytes, &flush_stats,
        nullptr /* blob_file_garbages */, false /* fast_sst_open */,
        nullptr /* compaction_iteration_stats */);

    assert(!s.ok() || io_s.ok());
    io_s.PermitUncheckedError();

    if (s.ok() && hot_key_count != flush_stats.num_input_records) {
      std::string msg = "Expected " + std::to_string(hot_key_count) +
                        " entries in HotTable, but read " +
                        std::to_string(flush_stats.num_input_records);
      ROCKS_LOG_WARN(db_options_.info_log,
                     "[%s] [JOB %d] Standalone HotTable flush %s",
                     cfd_->GetName().c_str(), job_id_, msg.c_str());
      if (db_options_.flush_verify_memtable_count) {
        s = Status::Corruption(msg);
      }
    }
    if (s.ok() &&
        (mutable_cf_options_.table_factory->IsInstanceOf(
             TableFactory::kBlockBasedTableName()) ||
         mutable_cf_options_.table_factory->IsInstanceOf(
             TableFactory::kPlainTableName())) &&
        flush_stats.num_output_records != table_properties.num_entries) {
      s = Status::Corruption(
          "Number of keys in standalone HotTable flush output SST does not "
          "match number of keys added to the table.");
    }

    if (s.ok() && output_file_directory_ != nullptr) {
      s = output_file_directory_->FsyncWithDirOptions(
          IOOptions(), nullptr,
          DirFsyncOptions(DirFsyncOptions::FsyncReason::kNewFileSynced));
    }

    db_mutex_->Lock();
  }

  const bool has_output = s.ok() && hot_meta.fd.GetFileSize() > 0;
  if (s.ok() && has_output) {
    VersionEdit edit;
    edit.SetColumnFamily(cfd_->GetID());
    edit.AddFile(0 /* level */, hot_meta);
    edit.SetBlobFileAdditions(std::move(blob_file_additions));

    const ReadOptions read_options(Env::IOActivity::kFlush);
    const WriteOptions write_options(Env::IOActivity::kFlush);
    s = versions_->LogAndApply(cfd_, read_options, write_options, &edit,
                               db_mutex_, db_directory_);
  }

  RecordTick(stats_, MEMTABLE_PAYLOAD_BYTES_AT_FLUSH, memtable_payload_bytes);
  RecordTick(stats_, MEMTABLE_GARBAGE_BYTES_AT_FLUSH, memtable_garbage_bytes);
  if (s.ok()) {
    flush_stats.bytes_written = hot_meta.fd.GetFileSize();
    flush_stats.num_output_files = has_output ? 1 : 0;
    RecordTick(stats_, MEMTABLE_FLUSH_COUNT);
    cfd_->internal_stats()->AddCompactionStats(0 /* level */,
                                               Env::Priority::LOW,
                                               flush_stats);
    cfd_->internal_stats()->AddCFStats(
        InternalStats::BYTES_FLUSHED,
        flush_stats.bytes_written + flush_stats.bytes_written_blob);
  }

  ROCKS_LOG_INFO(db_options_.info_log,
                 "[%s] [JOB %d] Standalone HotTable flush: %" PRIu64
                 " bytes, %" PRIu64 " keys, status: %s",
                 cfd_->GetName().c_str(), job_id_, hot_meta.fd.GetFileSize(),
                 static_cast<uint64_t>(hot_key_count), s.ToString().c_str());

  return s;
}

}  // namespace ROCKSDB_NAMESPACE
