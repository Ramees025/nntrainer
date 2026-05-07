// SPDX-License-Identifier: Apache-2.0
/**
 * @file   test_tokenizer_bin.cpp
 * @brief  Smoke test for the binary tokenizer cache path.
 *
 * Usage:
 *   test_tokenizer_bin <tokenizer.json> [text...]
 *
 * The first invocation loads from JSON and writes <tokenizer.json>.bin as a
 * side-effect; the second invocation hits the binary cache path. The test
 * prints encode/decode results and the wallclock spent on tokenizer loading
 * for both paths so the speedup can be inspected manually.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tokenizers_cpp.h>

namespace {
double LoadAndEncode(const std::string &path, const std::string &text,
                     std::vector<int32_t> &out_ids, std::string &out_decoded) {
  auto t0 = std::chrono::steady_clock::now();
  auto tok = tokenizers::Tokenizer::FromFile(path, /*auto_cache=*/true);
  auto t1 = std::chrono::steady_clock::now();
  if (tok == nullptr) {
    throw std::runtime_error("FromFile returned nullptr");
  }
  out_ids = tok->Encode(text, /*add_special_tokens=*/false);
  out_decoded = tok->Decode(out_ids, /*skip_special_tokens=*/false);
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "Usage: %s <tokenizer.json> [text...]\n", argv[0]);
    return 2;
  }
  const std::string json_path = argv[1];
  std::string text = "Hello, world! This is a tokenizer round-trip smoke test.";
  if (argc >= 3) {
    text = argv[2];
    for (int i = 3; i < argc; ++i) {
      text += " ";
      text += argv[i];
    }
  }

  // Wipe any pre-existing .bin so we always exercise the JSON path first.
  const std::string bin_path = json_path + ".bin";
  std::error_code ec;
  std::filesystem::remove(bin_path, ec);

  std::vector<int32_t> ids_json;
  std::string dec_json;
  const double ms_json = LoadAndEncode(json_path, text, ids_json, dec_json);

  std::vector<int32_t> ids_bin;
  std::string dec_bin;
  const double ms_bin = LoadAndEncode(json_path, text, ids_bin, dec_bin);

  std::cout << "JSON load + first parse:  " << ms_json << " ms\n";
  std::cout << "Binary cache reload:      " << ms_bin << " ms ("
            << (ms_json > 0 ? (ms_json / std::max(ms_bin, 1e-9)) : 0.0)
            << "x speedup)\n";
  std::cout << "Token count: " << ids_json.size() << "\n";

  if (ids_json != ids_bin) {
    std::cerr << "FAIL: token id sequences differ between JSON and binary "
                 "loads.\n";
    return 1;
  }
  if (dec_json != dec_bin) {
    std::cerr << "FAIL: decoded strings differ between JSON and binary "
                 "loads.\n";
    return 1;
  }

  std::cout << "Decoded text: " << dec_bin << "\n";
  std::cout << "PASS\n";
  return 0;
}
