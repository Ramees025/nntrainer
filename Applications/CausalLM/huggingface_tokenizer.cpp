
/*!
 *  Copyright (c) 2023 by Contributors
 * \file huggingface_tokenizer.cc
 * \brief Huggingface tokenizer
 */
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace tokenizers {
/*!
 * \brief A simple c++ header of tokenizer via C API.
 */
class HFTokenizer : public Tokenizer {
public:
  explicit HFTokenizer(TokenizerHandle handle) : handle_(handle) {
#ifdef COMPILE_WASM_RUNTIME
    setenv("TOKENIZERS_PARALLELISM", "false", true);
#endif
  }

  HFTokenizer(const HFTokenizer &) = delete;
  HFTokenizer(HFTokenizer &&other) { std::swap(other.handle_, handle_); }

  ~HFTokenizer() {
    if (handle_ != nullptr) {
      tokenizers_free(handle_);
    }
  }

  // use i32 to be consistent with sentencepiece
  std::vector<int32_t> Encode(const std::string &text,
                              bool add_special_tokens) {
    TokenizerEncodeResult result;
    tokenizers_encode(handle_, text.data(), text.length(),
                      static_cast<int>(add_special_tokens), &result);
    std::vector<int32_t> ret(result.token_ids, result.token_ids + result.len);
    tokenizers_free_encode_results(&result, 1);
    return ret;
  }

  // use i32 to be consistent with sentencepiece
  std::vector<int32_t> Encode(const std::string &text) final {
    return Encode(text, false);
  }

  std::vector<std::vector<int32_t>>
  EncodeBatch(const std::vector<std::string> &texts, bool add_special_tokens) {
    std::vector<const char *> texts_raw;
    std::vector<size_t> seq_lens;
    size_t num_seqs = texts.size();
    texts_raw.reserve(num_seqs);
    seq_lens.reserve(num_seqs);
    for (const auto &text : texts) {
      texts_raw.push_back(text.data());
      seq_lens.push_back(text.length());
    }
    std::vector<TokenizerEncodeResult> results(num_seqs);
    tokenizers_encode_batch(handle_, texts_raw.data(), seq_lens.data(),
                            texts.size(), static_cast<int>(add_special_tokens),
                            results.data());
    std::vector<std::vector<int32_t>> ret;
    ret.reserve(texts.size());
    for (size_t i = 0; i < texts.size(); ++i) {
      ret.push_back(std::vector<int32_t>(
        results[i].token_ids, results[i].token_ids + results[i].len));
    }
    tokenizers_free_encode_results(results.data(), texts.size());
    return ret;
  }

  std::vector<std::vector<int32_t>>
  EncodeBatch(const std::vector<std::string> &texts) final {
    return EncodeBatch(texts, false);
  }

  // use i32 to be consistent with sentencepiece
  std::string Decode(const std::vector<int32_t> &ids,
                     bool skip_special_tokens) {
    tokenizers_decode(handle_, reinterpret_cast<const uint32_t *>(ids.data()),
                      ids.size(), static_cast<int>(skip_special_tokens));
    const char *data;
    size_t len;
    tokenizers_get_decode_str(handle_, &data, &len);
    return std::string(data, len);
  }

  std::string Decode(const std::vector<int32_t> &ids) final {
    return Decode(ids, false);
  }

  size_t GetVocabSize() final {
    size_t size;
    tokenizers_get_vocab_size(handle_, &size);
    assert(size > 0);
    return size;
  }

  std::string IdToToken(int32_t id) final {
    const char *data;
    size_t len;
    tokenizers_id_to_token(handle_, static_cast<uint32_t>(id), &data, &len);
    return std::string(data, len);
  }

  int32_t TokenToId(const std::string &token) final {
    int32_t id;
    tokenizers_token_to_id(handle_, token.data(), token.length(), &id);
    return id;
  }

  bool SaveBinary(const std::string &path) final {
    if (handle_ == nullptr) {
      return false;
    }
    int rc = tokenizers_save_to_bin(handle_, path.data(), path.length());
    return rc == 0;
  }

private:
  // internal handle
  TokenizerHandle handle_{nullptr};
};

std::unique_ptr<Tokenizer> Tokenizer::FromBlobJSON(const std::string &json) {
  return std::make_unique<HFTokenizer>(
    tokenizers_new_from_str(json.data(), json.length()));
}

std::unique_ptr<Tokenizer>
Tokenizer::FromBlobByteLevelBPE(const std::string &vocab,
                                const std::string &merges,
                                const std::string &added_tokens) {
  return std::make_unique<HFTokenizer>(byte_level_bpe_tokenizers_new_from_str(
    vocab.data(), vocab.length(), merges.data(), merges.length(),
    added_tokens.data(), added_tokens.length()));
}

namespace {

// 9-byte magic + 4-byte LE u32 version. Must match patches/bin_io_append.rs.
constexpr const char kBinMagic[] = "NTRTKBIN";
constexpr size_t kBinMagicLen = 9;     // includes the trailing NUL
constexpr size_t kBinHeaderLen = kBinMagicLen + 4;

bool HasBinaryHeader(const std::string &blob) {
  if (blob.size() < kBinHeaderLen) {
    return false;
  }
  return std::memcmp(blob.data(), kBinMagic, kBinMagicLen) == 0;
}

bool LoadFileToString(const std::string &path, std::string &out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    return false;
  }
  std::streamsize size = f.tellg();
  if (size < 0) {
    return false;
  }
  f.seekg(0, std::ios::beg);
  out.assign(static_cast<size_t>(size), '\0');
  if (size > 0 && !f.read(&out[0], size)) {
    return false;
  }
  return true;
}

// Emit `msg` to stderr at most once for the given `flag`. Each distinct
// warning site should pass its own static once_flag so unrelated warnings do
// not suppress one another.
void WarnOnce(std::once_flag &flag, const std::string &msg) {
  std::call_once(flag, [&] {
    std::cerr << "[tokenizer] " << msg << std::endl;
  });
}

} // namespace

std::unique_ptr<Tokenizer>
Tokenizer::FromBlobBinary(const std::string &bin_blob) {
  if (!HasBinaryHeader(bin_blob)) {
    return nullptr;
  }
  TokenizerHandle handle =
    tokenizers_new_from_bin(bin_blob.data(), bin_blob.length());
  if (handle == nullptr) {
    return nullptr;
  }
  return std::make_unique<HFTokenizer>(handle);
}

std::unique_ptr<Tokenizer> Tokenizer::FromFile(const std::string &path,
                                               bool auto_cache) {
  // Detect whether the caller explicitly asked for the binary format.
  const bool explicit_bin =
    path.size() >= 4 &&
    path.compare(path.size() - 4, 4, ".bin") == 0;

  const std::string bin_path = explicit_bin ? path : (path + ".bin");

  // Step 1: try the binary cache.
  {
    std::string blob;
    if (LoadFileToString(bin_path, blob)) {
      auto tok = FromBlobBinary(blob);
      if (tok != nullptr) {
        return tok;
      }
      static std::once_flag stale_cache_flag;
      WarnOnce(stale_cache_flag,
               "binary tokenizer cache " + bin_path +
                 " is incompatible (magic/version mismatch); falling back to "
                 "JSON.");
    }
  }

  // Step 2: caller asked for the binary path explicitly but it could not be
  // loaded; nothing else to try.
  if (explicit_bin) {
    throw std::runtime_error("Failed to load binary tokenizer: " + path);
  }

  // Step 3: load JSON.
  std::string json_blob;
  if (!LoadFileToString(path, json_blob)) {
    throw std::runtime_error("Failed to open tokenizer file: " + path);
  }
  auto tok = FromBlobJSON(json_blob);

  // Step 4: opportunistically dump a binary cache for next time.
  if (auto_cache && tok != nullptr) {
    if (!tok->SaveBinary(bin_path)) {
      static std::once_flag write_fail_flag;
      WarnOnce(write_fail_flag,
               "failed to write binary tokenizer cache to " + bin_path +
                 "; subsequent loads will continue to use JSON.");
    }
  }
  return tok;
}

} // namespace tokenizers
