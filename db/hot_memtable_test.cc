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

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
