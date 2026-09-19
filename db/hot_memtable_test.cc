#include <atomic>
#include <cinttypes>
#include <string>
#include <thread>
#include <vector>

#include "db/hot_memtable.h"
#include "db/hot_table_router.h"
#include "db/space_saving_topk.h"
#include "test_util/testharness.h"

namespace ROCKSDB_NAMESPACE {

class HotMemTableTest : public testing::Test {};

TEST_F(HotMemTableTest, BasicInPlaceUpdateAndGet) {
  InternalKeyComparator cmp(BytewiseComparator());
  HotMemTable hot_table(cmp, 1024 * 1024, 256);

  ASSERT_TRUE(hot_table.Add("user_key1", "val1", kTypeValue, 100));
  ASSERT_EQ(hot_table.KeyCount(), 1);

  std::string val;
  Status s;
  SequenceNumber seq = 0;
  ASSERT_TRUE(hot_table.Get("user_key1", &val, &s, &seq));
  ASSERT_OK(s);
  ASSERT_EQ(val, "val1");
  ASSERT_EQ(seq, 100);

  // In-place update with longer string within max_val_size
  ASSERT_TRUE(hot_table.UpdateInPlace("user_key1", "val1_updated_longer", kTypeValue, 101));
  ASSERT_TRUE(hot_table.Get("user_key1", &val, &s, &seq));
  ASSERT_OK(s);
  ASSERT_EQ(val, "val1_updated_longer");
  ASSERT_EQ(seq, 101);

  // In-place tombstone overwrite
  ASSERT_TRUE(hot_table.UpdateInPlace("user_key1", "", kTypeDeletion, 102));
  ASSERT_TRUE(hot_table.Get("user_key1", &val, &s, &seq));
  ASSERT_TRUE(s.IsNotFound());
  ASSERT_EQ(seq, 102);

  // Non-existent key should return false
  ASSERT_FALSE(hot_table.UpdateInPlace("absent_key", "val", kTypeValue, 103));
  ASSERT_FALSE(hot_table.Get("absent_key", &val, &s, &seq));
}

TEST_F(HotMemTableTest, SeqlockConcurrentNoTornReads) {
  InternalKeyComparator cmp(BytewiseComparator());
  HotMemTable hot_table(cmp, 1024 * 1024, 256);

  std::string key = "hot_counter_key";
  hot_table.Add(key, "0000000000", kTypeValue, 1);

  std::atomic<bool> stop{false};
  std::atomic<uint64_t> write_ops{0};
  std::atomic<uint64_t> read_ops{0};

  // Writer thread constantly updates value in-place with formatted 10-digit number
  std::thread writer([&]() {
    char buf[16];
    uint64_t count = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      count++;
      snprintf(buf, sizeof(buf), "%010" PRIu64, count);
      hot_table.UpdateInPlace(key, Slice(buf, 10), kTypeValue, count);
      write_ops.fetch_add(1, std::memory_order_relaxed);
    }
  });

  // Reader threads verify that every read is untorn and exactly 10 matching digits
  auto reader_func = [&]() {
    std::string val;
    Status s;
    SequenceNumber seq = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      if (hot_table.Get(key, &val, &s, &seq)) {
        ASSERT_OK(s);
        ASSERT_EQ(val.size(), 10);
        // Verify format integrity
        for (char c : val) {
          ASSERT_TRUE(c >= '0' && c <= '9');
        }
        read_ops.fetch_add(1, std::memory_order_relaxed);
      }
    }
  };

  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back(reader_func);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  stop.store(true);

  writer.join();
  for (auto& r : readers) {
    r.join();
  }

  ASSERT_GT(write_ops.load(), 1000);
  ASSERT_GT(read_ops.load(), 1000);
}

TEST_F(HotMemTableTest, HotTableRouterBloomRejection) {
  HotTableRouter router(100);
  router.Add("hot1");
  router.Add("hot2");

  ASSERT_TRUE(router.MayContain("hot1"));
  ASSERT_TRUE(router.MayContain("hot2"));
  ASSERT_FALSE(router.MayContain("definitely_absent_cold_key_xyz123"));

  router.Rebuild(100);
  // After rebuild, previous keys are cleared
  ASSERT_FALSE(router.MayContain("hot1"));
}

TEST_F(HotMemTableTest, SpaceSavingTopKDecayAndPenalties) {
  SpaceSavingTopK tracker(10, 0.5 /* decay_factor */, 0.25 /* zero_hit_penalty */);

  tracker.Update("key_hot", 100);
  tracker.Update("key_warm", 40);
  tracker.Update("key_cold", 10);

  auto top = tracker.GetTopK(3);
  ASSERT_EQ(top.size(), 3);
  ASSERT_EQ(top[0].key, "key_hot");
  ASSERT_EQ(top[0].count, 100);

  // Sweep simulation: key_hot had 50 hits, key_warm had 0 hits (false positive)
  std::unordered_map<std::string, uint32_t> sweep_hits;
  sweep_hits["key_hot"] = 50;
  sweep_hits["key_warm"] = 0;

  tracker.ApplyDecayAndPenalties(sweep_hits);

  // key_hot: (100 + 50) * 0.5 = 75
  // key_warm: 40 * 0.25 = 10
  // key_cold: 10 * 0.5 = 5
  top = tracker.GetTopK(3);
  ASSERT_EQ(top[0].key, "key_hot");
  ASSERT_EQ(top[0].count, 75);
  ASSERT_EQ(top[1].key, "key_warm");
  ASSERT_EQ(top[1].count, 10);
  ASSERT_EQ(top[2].key, "key_cold");
  ASSERT_EQ(top[2].count, 5);

  tracker.Update("single_hit_key", 1);
  top = tracker.GetTopK(10, 2);
  for (const auto& entry : top) {
    ASSERT_NE(entry.key, "single_hit_key");
    ASSERT_GE(entry.count, 2);
  }
}

TEST_F(HotMemTableTest, IteratorSeekAndSortedOrder) {
  InternalKeyComparator cmp(BytewiseComparator());
  HotMemTable hot_table(cmp, 1024 * 1024, 256);

  hot_table.Add("c", "val_c", kTypeValue, 10);
  hot_table.Add("a", "val_a", kTypeValue, 20);
  hot_table.Add("b", "val_b", kTypeValue, 30);

  std::unique_ptr<InternalIterator> iter(hot_table.NewIterator());
  iter->SeekToFirst();
  ASSERT_TRUE(iter->Valid());

  ParsedInternalKey pikey;
  ASSERT_OK(ParseInternalKey(iter->key(), &pikey, false));
  ASSERT_EQ(pikey.user_key.ToString(), "a");
  ASSERT_EQ(iter->value().ToString(), "val_a");

  iter->Next();
  ASSERT_TRUE(iter->Valid());
  ASSERT_OK(ParseInternalKey(iter->key(), &pikey, false));
  ASSERT_EQ(pikey.user_key.ToString(), "b");
  ASSERT_EQ(iter->value().ToString(), "val_b");

  iter->Next();
  ASSERT_TRUE(iter->Valid());
  ASSERT_OK(ParseInternalKey(iter->key(), &pikey, false));
  ASSERT_EQ(pikey.user_key.ToString(), "c");
  ASSERT_EQ(iter->value().ToString(), "val_c");

  iter->Next();
  ASSERT_FALSE(iter->Valid());
}

TEST_F(HotMemTableTest, ConsecutiveWindowSkewnessTracking) {
  SpaceSavingTopK tracker(1024);

  // Initial state: HotTable active
  bool active = true;

  // Window 1: Flat flush (0.1% cold duplicates, 0.1% hot absorption)
  tracker.RecordFlushWindow(/*total_cold=*/1000, /*dup_cold=*/1, /*hot_hits=*/1, /*hot_misses=*/1000);
  bool skewed = tracker.IsWorkloadSkewed(active, /*min_dup=*/0.20, /*min_abs=*/0.20, /*consecutive_windows=*/2);
  ASSERT_TRUE(skewed); // 1st flat window: stays active

  // Window 2: Second flat flush
  tracker.RecordFlushWindow(1000, 1, 0, 1000);
  skewed = tracker.IsWorkloadSkewed(active, 0.20, 0.20, 2);
  ASSERT_FALSE(skewed); // 2nd flat window: transitions to disabled!

  // Now HotTable is disabled
  active = false;
  tracker.Clear();

  // Window 3: 1st skewed flush (30% cold duplicates)
  tracker.RecordFlushWindow(1000, 300, 0, 1000);
  skewed = tracker.IsWorkloadSkewed(active, 0.20, 0.20, 2);
  ASSERT_FALSE(skewed); // 1st skewed window: stays disabled

  // Window 4: 2nd skewed flush (40% cold duplicates)
  tracker.RecordFlushWindow(1000, 400, 0, 1000);
  skewed = tracker.IsWorkloadSkewed(active, 0.20, 0.20, 2);
  ASSERT_TRUE(skewed); // 2nd skewed window: transitions to enabled!
}

TEST_F(HotMemTableTest, RouterDisableAndAdaptiveRebuild) {
  HotTableRouter router(1024);
  ASSERT_TRUE(router.IsActive());
  router.Add("test_key");
  ASSERT_TRUE(router.MayContain("test_key"));

  // Disabling router
  router.Disable();
  ASSERT_FALSE(router.IsActive());
  ASSERT_FALSE(router.MayContain("test_key"));
  ASSERT_EQ(router.KeyCount(), 0);

  // Rebuild router
  router.Rebuild(1024);
  ASSERT_TRUE(router.IsActive());
  router.Add("test_key2");
  ASSERT_TRUE(router.MayContain("test_key2"));

  // Test tracker duplicate ratio and heavy hitter counts
  SpaceSavingTopK tracker(1024);
  tracker.RecordFlushWindow(1000, 2); // 0.2% duplicate ratio (uniform)
  ASSERT_LT(tracker.GetRecentDuplicateRatio(), 0.005);
  ASSERT_EQ(tracker.QualifiedHeavyHittersCount(2), 0);

  tracker.Update("hot_key1", 10);
  tracker.Update("hot_key2", 5);
  tracker.RecordFlushWindow(100, 15); // 15% duplicate ratio (skewed)
  ASSERT_GE(tracker.GetRecentDuplicateRatio(), 0.005);
  ASSERT_EQ(tracker.QualifiedHeavyHittersCount(2), 2);
}

TEST_F(HotMemTableTest, ScanProbeIntervalBacksOffOnceConfidentlyFlat) {
  struct TestCase {
    std::string name;
    uint32_t flat_windows;
    uint32_t threshold_windows;
    uint32_t max_backoff_flushes;
    uint32_t expected_interval;
  };
  TestCase cases[] = {
      // Still within (or at) the threshold: probe every flush.
      {"zero_flat_windows", 0, 2, 16, 1},
      {"at_threshold", 2, 2, 16, 1},
      // Past the threshold: interval grows with flat_windows, capped by
      // max_backoff_flushes.
      {"just_past_threshold", 3, 2, 16, 3},
      {"grows_with_flatness", 10, 2, 16, 10},
      {"capped_by_max_backoff", 100, 2, 16, 16},
      // Defensive clamp: a misconfigured max_backoff_flushes of 0 must not
      // produce a modulo-by-zero interval at the call site.
      {"zero_max_backoff_clamped_to_one", 100, 2, 0, 1},
  };
  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.name);
    ASSERT_EQ(
        SpaceSavingTopK::ComputeScanProbeInterval(
            tc.flat_windows, tc.threshold_windows, tc.max_backoff_flushes),
        tc.expected_interval);
  }
}

TEST_F(HotMemTableTest, ConcurrentWritersHighestSeqWinsAndNoTornWrites) {
  InternalKeyComparator cmp(BytewiseComparator());
  HotMemTable hot_table(cmp, 1024 * 1024, 256);

  std::string key = "contended_hot_key";
  hot_table.Add(key, "init_val_0000000000", kTypeValue, 1);

  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> global_seq{1};
  std::atomic<uint64_t> total_writes{0};

  const int num_writers = 12;
  const int num_readers = 4;
  std::vector<std::thread> writers;
  std::vector<std::thread> readers;

  // 12 Concurrent writers hammering the same key
  for (int w = 0; w < num_writers; ++w) {
    writers.emplace_back([&, w]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      char buf[64];
      while (!stop.load(std::memory_order_relaxed)) {
        uint64_t my_seq = global_seq.fetch_add(1, std::memory_order_relaxed);
        // Payload consists of repeated formatted sequence number to detect torn writes
        snprintf(buf, sizeof(buf), "%016" PRIu64 ":%016" PRIu64, my_seq, my_seq);
        hot_table.UpdateInPlace(key, Slice(buf, 33), kTypeValue, my_seq);
        total_writes.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // 4 Concurrent readers checking for untorn values and valid sequence numbers
  std::atomic<uint64_t> total_reads{0};
  for (int r = 0; r < num_readers; ++r) {
    readers.emplace_back([&, r]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      std::string val;
      Status s;
      SequenceNumber seq = 0;
      uint64_t last_seq = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        if (hot_table.Get(key, &val, &s, &seq)) {
          ASSERT_OK(s);
          if (seq == 1) {
            ASSERT_EQ(val, "init_val_0000000000");
            continue;
          }
          ASSERT_EQ(val.size(), 33);
          uint64_t s1 = 0, s2 = 0;
          ASSERT_EQ(sscanf(val.c_str(), "%016" PRIu64 ":%016" PRIu64, &s1, &s2), 2);
          // Both parts must match (proves no torn write occurred)
          ASSERT_EQ(s1, s2);
          ASSERT_EQ(s1, seq);
          ASSERT_GE(seq, last_seq);
          last_seq = seq;
          total_reads.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  // Start all threads simultaneously
  start.store(true, std::memory_order_release);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  stop.store(true, std::memory_order_release);

  for (auto& t : writers) {
    t.join();
  }
  for (auto& t : readers) {
    t.join();
  }

  // Final validation: HotTable must hold the HIGHEST sequence number written
  std::string final_val;
  Status final_status;
  SequenceNumber final_seq = 0;
  ASSERT_TRUE(hot_table.Get(key, &final_val, &final_status, &final_seq));
  ASSERT_OK(final_status);

  uint64_t highest_assigned_seq = global_seq.load(std::memory_order_relaxed) - 1;
  ASSERT_EQ(final_seq, highest_assigned_seq);

  uint64_t val_s1 = 0, val_s2 = 0;
  ASSERT_EQ(sscanf(final_val.c_str(), "%016" PRIu64 ":%016" PRIu64, &val_s1, &val_s2), 2);
  ASSERT_EQ(val_s1, highest_assigned_seq);
  ASSERT_EQ(val_s2, highest_assigned_seq);
}

TEST_F(HotMemTableTest, DynamicValueGrowthBeyondMaxValSize) {
  InternalKeyComparator cmp(BytewiseComparator());
  // max_val_size initialized to 64 bytes
  HotMemTable hot_table(cmp, 1024 * 1024, 64);

  std::string small_val(40, 'a');
  ASSERT_TRUE(hot_table.Add("resize_key", small_val, kTypeValue, 10));

  std::string read_val;
  Status s;
  SequenceNumber seq = 0;
  ASSERT_TRUE(hot_table.Get("resize_key", &read_val, &s, &seq));
  ASSERT_OK(s);
  ASSERT_EQ(read_val, small_val);

  // Exceed initial max_val_size (150 bytes > 64 bytes)
  std::string med_val(150, 'b');
  ASSERT_TRUE(hot_table.UpdateInPlace("resize_key", med_val, kTypeValue, 11));
  ASSERT_TRUE(hot_table.Get("resize_key", &read_val, &s, &seq));
  ASSERT_OK(s);
  ASSERT_EQ(read_val, med_val);
  ASSERT_EQ(read_val.size(), 150);

  // Exceed further with 1200 bytes
  std::string large_val(1200, 'c');
  ASSERT_TRUE(hot_table.UpdateInPlace("resize_key", large_val, kTypeValue, 12));
  ASSERT_TRUE(hot_table.Get("resize_key", &read_val, &s, &seq));
  ASSERT_OK(s);
  ASSERT_EQ(read_val, large_val);
  ASSERT_EQ(read_val.size(), 1200);
}

TEST_F(HotMemTableTest, ConcurrentResizeNoTornSeqValuePairs) {
  InternalKeyComparator cmp(BytewiseComparator());
  // Small max_val_size so the growth (resize) path triggers frequently.
  HotMemTable hot_table(cmp, 1024 * 1024, 32);

  std::string key = "resize_race_key";
  hot_table.Add(key, std::string(16, 'a'), kTypeValue, 1);

  std::atomic<bool> stop{false};
  std::atomic<uint64_t> seq_counter{1};

  // Writer alternates value sizes across the initial capacity boundary to
  // force repeated node reallocation -- the resize path where the old,
  // now-detached HotNode used to get its seq/seq_version incorrectly bumped
  // after the index already pointed at the new node.
  std::thread writer([&]() {
    const size_t sizes[] = {16, 40, 16, 100, 16, 500};
    size_t idx = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      uint64_t seq = seq_counter.fetch_add(1, std::memory_order_relaxed) + 1;
      size_t len = sizes[idx++ % (sizeof(sizes) / sizeof(sizes[0]))];
      // Every byte of the value encodes the seq's low byte, so a reader can
      // detect whether the returned bytes actually correspond to a value
      // the writer produced for the returned seq.
      std::string val(len, static_cast<char>(seq & 0xFF));
      hot_table.UpdateInPlace(key, val, kTypeValue, seq);
    }
  });

  std::atomic<uint64_t> read_ops{0};
  auto reader_func = [&]() {
    std::string val;
    Status s;
    SequenceNumber seq = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      if (hot_table.Get(key, &val, &s, &seq)) {
        ASSERT_OK(s);
        if (seq == 1) {
          ASSERT_EQ(val, std::string(16, 'a'));
        } else {
          // A torn (seq, value) pair -- new seq paired with stale bytes from
          // before a resize -- would fail this check, since the stale bytes
          // encode a different seq's low byte.
          char expected = static_cast<char>(seq & 0xFF);
          for (char c : val) {
            ASSERT_EQ(c, expected);
          }
        }
        read_ops.fetch_add(1, std::memory_order_relaxed);
      }
    }
  };

  std::vector<std::thread> readers;
  for (int i = 0; i < 4; ++i) {
    readers.emplace_back(reader_func);
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  stop.store(true);

  writer.join();
  for (auto& r : readers) {
    r.join();
  }

  ASSERT_GT(read_ops.load(), 0u);
}

TEST_F(HotMemTableTest, RouterConcurrentRebuildNoUseAfterFree) {
  HotTableRouter router(1024);

  std::atomic<bool> stop{false};
  std::atomic<uint64_t> query_ops{0};

  // Several threads keep calling Add()/MayContain() while the main thread
  // repeatedly Rebuild()s/Disable()s the router out from under them. This
  // does not assert on MayContain()'s result (which is inherently racy here)
  // -- it exists to be run under ASAN/TSAN to catch the use-after-free that
  // used to be reachable when the old bloom's backing Arena was freed before
  // the new one was published.
  auto query_func = [&](int tid) {
    uint64_t i = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      std::string key =
          "key_" + std::to_string(tid) + "_" + std::to_string(i++);
      router.Add(key);
      router.MayContain(key);
      query_ops.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> queriers;
  for (int t = 0; t < 4; ++t) {
    queriers.emplace_back(query_func, t);
  }

  std::thread rebuilder([&]() {
    int i = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      if (i % 2 == 0) {
        router.Rebuild(1024);
      } else {
        router.Disable();
      }
      i++;
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  stop.store(true);

  rebuilder.join();
  for (auto& t : queriers) {
    t.join();
  }

  ASSERT_GT(query_ops.load(), 0u);
}

TEST_F(HotMemTableTest, PrePopulatedNodeAndSingleDelete) {
  InternalKeyComparator cmp(BytewiseComparator());
  HotMemTable hot_table(cmp, 1024 * 1024, 128);

  // Pre-populate key with seq = 0
  ASSERT_TRUE(hot_table.Add("prepop_key", "", kTypeValue, 0));
  ASSERT_EQ(hot_table.KeyCount(), 0);
  ASSERT_TRUE(hot_table.IsEmpty());

  // Get on pre-populated key must return false (fall through to cold memtable)
  std::string val;
  Status s;
  SequenceNumber seq = 0;
  ASSERT_FALSE(hot_table.Get("prepop_key", &val, &s, &seq));

  // Update in place with actual data
  ASSERT_TRUE(
      hot_table.UpdateInPlace("prepop_key", "active_val", kTypeValue, 100));
  ASSERT_EQ(hot_table.KeyCount(), 1);
  ASSERT_FALSE(hot_table.IsEmpty());
  ASSERT_TRUE(hot_table.Get("prepop_key", &val, &s, &seq));
  ASSERT_OK(s);
  ASSERT_EQ(val, "active_val");
  ASSERT_EQ(seq, 100);

  // SingleDelete update
  ASSERT_TRUE(
      hot_table.UpdateInPlace("prepop_key", "", kTypeSingleDeletion, 101));
  ASSERT_TRUE(hot_table.Get("prepop_key", &val, &s, &seq));
  ASSERT_TRUE(s.IsNotFound());
  ASSERT_EQ(seq, 101);
}

TEST_F(HotMemTableTest, HotMemTableIteratorFilteringAndSeeking) {
  InternalKeyComparator cmp(BytewiseComparator());
  HotMemTable hot_table(cmp, 1024 * 1024, 128);

  // Add a pre-populated key (seq = 0)
  hot_table.Add("key0_dummy", "", kTypeValue, 0);

  // Add valid written keys
  hot_table.Add("key1", "val1", kTypeValue, 10);
  hot_table.Add("key2", "val2", kTypeValue, 20);
  hot_table.Add("key3", "val3", kTypeValue, 30);

  size_t count = 0;
  std::unique_ptr<InternalIterator> iter(
      hot_table.NewIterator(nullptr, &count));
  // Only the 3 written keys should be counted and iterated
  ASSERT_EQ(count, 3);

  iter->SeekToFirst();
  ASSERT_TRUE(iter->Valid());
  ParsedInternalKey pik;
  ASSERT_OK(ParseInternalKey(iter->key(), &pik, false));
  ASSERT_EQ(pik.user_key, "key1");
  ASSERT_EQ(iter->value(), "val1");

  iter->Next();
  ASSERT_TRUE(iter->Valid());
  ASSERT_OK(ParseInternalKey(iter->key(), &pik, false));
  ASSERT_EQ(pik.user_key, "key2");

  iter->Next();
  ASSERT_TRUE(iter->Valid());
  ASSERT_OK(ParseInternalKey(iter->key(), &pik, false));
  ASSERT_EQ(pik.user_key, "key3");

  iter->Next();
  ASSERT_FALSE(iter->Valid());
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
