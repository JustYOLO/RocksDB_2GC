//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <memory>
#include <string>

#include "options/db_options.h"
#include "rocksdb/env.h"
#include "rocksdb/status.h"
#include "rocksdb/types.h"

namespace ROCKSDB_NAMESPACE {

class ColumnFamilyData;
class HotMemTable;
class VersionSet;
class InstrumentedMutex;
class FSDirectory;
class EventLogger;
class Statistics;
class IOTracer;
struct JobContext;
struct FileOptions;
struct MutableCFOptions;

// Standalone physical flush of a frozen HotMemTable into its own SST file,
// committed via its own VersionEdit/LogAndApply() call. Entirely independent
// of any cold-memtable FlushJob -- this is what lets HotTable's physical
// flush run on a background thread, off the write-stalling critical path.
// See docs/plans (or the "Zero-Stall HotTable Physical Flush" design) for
// the full rationale.
//
// REQUIRES: `hot_mem` has already been Close()d by the caller (this job does
// not close it -- closing is the write-redirection cutover point and must
// happen before this job is even constructed, not as part of running it).
class HotTableFlushJob {
 public:
  HotTableFlushJob(const std::string& dbname, ColumnFamilyData* cfd,
                   const ImmutableDBOptions& db_options,
                   const MutableCFOptions& mutable_cf_options,
                   const FileOptions& file_options, VersionSet* versions,
                   InstrumentedMutex* db_mutex,
                   std::shared_ptr<HotMemTable> hot_mem,
                   FSDirectory* db_directory, FSDirectory* output_file_directory,
                   CompressionType output_compression, Statistics* stats,
                   EventLogger* event_logger,
                   const std::shared_ptr<IOTracer>& io_tracer,
                   const std::string& db_id, const std::string& db_session_id,
                   int job_id, JobContext* job_context);

  // Requires: db_mutex held on entry (released internally around the actual
  // BuildTable() I/O, exactly like FlushJob::WriteLevel0Table(), and
  // reacquired before returning).
  Status Run();

 private:
  const std::string& dbname_;
  ColumnFamilyData* cfd_;
  const ImmutableDBOptions& db_options_;
  const MutableCFOptions& mutable_cf_options_;
  const FileOptions& file_options_;
  VersionSet* versions_;
  InstrumentedMutex* db_mutex_;
  std::shared_ptr<HotMemTable> hot_mem_;
  FSDirectory* db_directory_;
  FSDirectory* output_file_directory_;
  CompressionType output_compression_;
  Statistics* stats_;
  EventLogger* event_logger_;
  const std::shared_ptr<IOTracer>& io_tracer_;
  const std::string& db_id_;
  const std::string& db_session_id_;
  int job_id_;
  JobContext* job_context_;
};

}  // namespace ROCKSDB_NAMESPACE
