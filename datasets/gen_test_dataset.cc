#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  uint64_t num_keys = 100000;
  std::string filename = "data/test_dist_100K_uint64";

  if (argc > 1) {
    num_keys = std::strtoull(argv[1], nullptr, 10);
  }
  if (argc > 2) {
    filename = argv[2];
  }

  FILE* f = fopen(filename.c_str(), "wb");
  if (!f) {
    fprintf(stderr, "Failed to open output file: %s\n", filename.c_str());
    return 1;
  }

  // 1. Write 8-byte count header
  if (fwrite(&num_keys, sizeof(uint64_t), 1, f) != 1) {
    fprintf(stderr, "Failed to write header\n");
    fclose(f);
    return 1;
  }

  // 2. Generate random 64-bit keys (clustered/realistic)
  std::mt19937_64 rng(912912);
  std::uniform_int_distribution<uint64_t> dist(1000000ULL, 100000000000ULL);

  std::vector<uint64_t> buffer(4096);
  uint64_t generated = 0;
  while (generated < num_keys) {
    uint64_t chunk = std::min<uint64_t>(4096, num_keys - generated);
    for (uint64_t i = 0; i < chunk; ++i) {
      buffer[i] = dist(rng);
    }
    if (fwrite(buffer.data(), sizeof(uint64_t), chunk, f) != chunk) {
      fprintf(stderr, "Write error at key %llu\n", (unsigned long long)generated);
      fclose(f);
      return 1;
    }
    generated += chunk;
  }

  fclose(f);
  printf("Generated %llu keys in %s\n", (unsigned long long)num_keys, filename.c_str());
  return 0;
}
