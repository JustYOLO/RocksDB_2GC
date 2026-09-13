#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

void downsample_file(const std::string& prefix) {
  std::string in_path = "data/" + prefix + "_800M_uint64";
  std::string out_600M = "data/" + prefix + "_600M_uint64";
  std::string out_400M = "data/" + prefix + "_400M_uint64";
  std::string out_200M = "data/" + prefix + "_200M_uint64";

  FILE* fin = fopen(in_path.c_str(), "rb");
  if (!fin) {
    fprintf(stderr, "Could not open input file: %s\n", in_path.c_str());
    return;
  }

  uint64_t count = 0;
  if (fread(&count, sizeof(uint64_t), 1, fin) != 1) {
    fprintf(stderr, "Could not read count header from: %s\n", in_path.c_str());
    fclose(fin);
    return;
  }
  printf("Processing %s: declared count = %llu\n", in_path.c_str(), (unsigned long long)count);

  FILE* f600 = fopen(out_600M.c_str(), "wb");
  FILE* f400 = fopen(out_400M.c_str(), "wb");
  FILE* f200 = fopen(out_200M.c_str(), "wb");

  if (!f600 || !f400 || !f200) {
    fprintf(stderr, "Failed to open output files for %s\n", prefix.c_str());
    if (f600) fclose(f600);
    if (f400) fclose(f400);
    if (f200) fclose(f200);
    fclose(fin);
    return;
  }

  // Pre-write dummy headers (8 bytes) to seek back and write true counts later
  uint64_t dummy = 0;
  fwrite(&dummy, sizeof(uint64_t), 1, f600);
  fwrite(&dummy, sizeof(uint64_t), 1, f400);
  fwrite(&dummy, sizeof(uint64_t), 1, f200);

  uint64_t count_600 = 0;
  uint64_t count_400 = 0;
  uint64_t count_200 = 0;

  const size_t BUF_SIZE = 1048576; // 1M elements = 8MB buffer
  std::vector<uint64_t> in_buf(BUF_SIZE);
  std::vector<uint64_t> buf_600;
  std::vector<uint64_t> buf_400;
  std::vector<uint64_t> buf_200;
  buf_600.reserve(BUF_SIZE);
  buf_400.reserve(BUF_SIZE / 2);
  buf_200.reserve(BUF_SIZE / 4);

  uint64_t global_idx = 0;
  size_t n_read = 0;
  while ((n_read = fread(in_buf.data(), sizeof(uint64_t), BUF_SIZE, fin)) > 0) {
    for (size_t i = 0; i < n_read; ++i) {
      uint64_t val = in_buf[i];
      // 400M: every 2nd element (idx % 2 == 0)
      if (global_idx % 2 == 0) {
        buf_400.push_back(val);
        count_400++;
      }
      // 200M: every 4th element (idx % 4 == 0)
      if (global_idx % 4 == 0) {
        buf_200.push_back(val);
        count_200++;
      }
      // 600M: delete every 4th element (idx % 4 != 0)
      if (global_idx % 4 != 0) {
        buf_600.push_back(val);
        count_600++;
      }
      global_idx++;
    }
    if (buf_600.size() >= BUF_SIZE / 2) {
      fwrite(buf_600.data(), sizeof(uint64_t), buf_600.size(), f600);
      buf_600.clear();
    }
    if (buf_400.size() >= BUF_SIZE / 2) {
      fwrite(buf_400.data(), sizeof(uint64_t), buf_400.size(), f400);
      buf_400.clear();
    }
    if (buf_200.size() >= BUF_SIZE / 2) {
      fwrite(buf_200.data(), sizeof(uint64_t), buf_200.size(), f200);
      buf_200.clear();
    }
  }

  if (!buf_600.empty()) {
    fwrite(buf_600.data(), sizeof(uint64_t), buf_600.size(), f600);
    buf_600.clear();
  }
  if (!buf_400.empty()) {
    fwrite(buf_400.data(), sizeof(uint64_t), buf_400.size(), f400);
    buf_400.clear();
  }
  if (!buf_200.empty()) {
    fwrite(buf_200.data(), sizeof(uint64_t), buf_200.size(), f200);
    buf_200.clear();
  }

  // Write headers
  fseek(f600, 0, SEEK_SET);
  fwrite(&count_600, sizeof(uint64_t), 1, f600);
  fclose(f600);

  fseek(f400, 0, SEEK_SET);
  fwrite(&count_400, sizeof(uint64_t), 1, f400);
  fclose(f400);

  fseek(f200, 0, SEEK_SET);
  fwrite(&count_200, sizeof(uint64_t), 1, f200);
  fclose(f200);

  fclose(fin);
  printf("Finished %s: 600M=%llu, 400M=%llu, 200M=%llu\n",
         prefix.c_str(),
         (unsigned long long)count_600,
         (unsigned long long)count_400,
         (unsigned long long)count_200);
}

int main(int argc, char** argv) {
  if (argc > 1) {
    for (int i = 1; i < argc; ++i) {
      downsample_file(argv[i]);
    }
  } else {
    downsample_file("books");
    downsample_file("osm_cellids");
  }
  return 0;
}
