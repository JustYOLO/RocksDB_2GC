#include "db/spatial_cms.h"

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "test_util/testharness.h"

namespace ROCKSDB_NAMESPACE {

class SpatialCMSTest : public testing::Test {};

TEST_F(SpatialCMSTest, BasicAddAndEstimate) {
  SpatialCountMinSketch cms(8);

  ASSERT_EQ(cms.Estimate("user_key_1"), 0);

  cms.AddPrefix("user_key_1");
  cms.AddPrefix("user_key_1");
  cms.AddPrefix("user_key_1");

  ASSERT_GE(cms.Estimate("user_key_1"), 3);
  ASSERT_EQ(cms.GetTotalEvictions(), 3);

  // Different prefix should have 0 or low count
  ASSERT_EQ(cms.Estimate("other_prefix_xyz"), 0);
}

TEST_F(SpatialCMSTest, PrefixTruncation) {
  SpatialCountMinSketch cms(8);

  // 8-byte prefix of both is 'user_key'
  cms.AddPrefix("user_key_alpha");
  cms.AddPrefix("user_key_beta");

  ASSERT_GE(cms.Estimate("user_key_gamma"), 2);
}

TEST_F(SpatialCMSTest, DecayAndSkewEvaluation) {
  SpatialCountMinSketch cms(8);

  // Add highly skewed counts to one prefix
  for (int i = 0; i < 200; ++i) {
    cms.AddPrefix("hot_prefix_000");
  }
  // Add a few scattered counts to cold prefixes
  for (int i = 0; i < 10; ++i) {
    std::string cold = "cold_" + std::to_string(i);
    cms.AddPrefix(cold);
  }

  // Decay and evaluate with threshold 2.0
  cms.DecayAndEvaluateSkew(2.0);

  ASSERT_TRUE(cms.IsWorkloadSkewed());
  ASSERT_GT(cms.GetLastSkewRatio(), 2.0);

  // Verify counters decayed (halved)
  ASSERT_LE(cms.Estimate("hot_prefix_000"), 105);
}

TEST_F(SpatialCMSTest, ConcurrentAdds) {
  SpatialCountMinSketch cms(8);
  constexpr int kNumThreads = 4;
  constexpr int kOpsPerThread = 1000;

  std::vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&cms, t]() {
      for (int i = 0; i < kOpsPerThread; ++i) {
        std::string key = "thread_" + std::to_string(t) + "_key_" + std::to_string(i);
        cms.AddPrefix(key);
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  ASSERT_EQ(cms.GetTotalEvictions(), kNumThreads * kOpsPerThread);
}

TEST_F(SpatialCMSTest, ResetTest) {
  SpatialCountMinSketch cms(8);
  cms.AddPrefix("prefix_123");
  ASSERT_GE(cms.Estimate("prefix_123"), 1);

  cms.Reset();
  ASSERT_EQ(cms.Estimate("prefix_123"), 0);
  ASSERT_EQ(cms.GetTotalEvictions(), 0);
  ASSERT_FALSE(cms.IsWorkloadSkewed());
}

TEST_F(SpatialCMSTest, HasHotKeyInRangeTest) {
  SpatialCountMinSketch cms(8);
  for (int i = 0; i < 10; ++i) {
    cms.AddPrefix("hot_prefix_001");
  }

  // Hot prefix at smallest key
  ASSERT_TRUE(cms.HasHotKeyInRange("hot_prefix_001", "cold_key_999", 5));
  // Hot prefix at largest key
  ASSERT_TRUE(cms.HasHotKeyInRange("cold_key_000", "hot_prefix_001", 5));
  // Both keys cold
  ASSERT_FALSE(cms.HasHotKeyInRange("cold_key_000", "cold_key_999", 5));
  // Threshold 0 always returns true
  ASSERT_TRUE(cms.HasHotKeyInRange("cold_key_000", "cold_key_999", 0));
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
