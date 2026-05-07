/**
 *  Copyright (c) 2023 by Contributors
 * @file tokenizers_cpp.h
 * @brief A C++ binding to common set of tokenizers
 * @author Contributors
 * @bug No known bugs
 */
#ifndef TOKENIZERS_CPP_H_
#define TOKENIZERS_CPP_H_

#include <memory>
#include <string>
#include <vector>

namespace tokenizers {

/**
 * @brief a universal tokenizer that loads
 *  either HF's tokenizer or sentence piece,
 *  depending on the constructor
 */
class Tokenizer {
public:
  /** @brief virtual destructor */
  virtual ~Tokenizer() {}

  /**
   * @brief Encode text into ids.
   * @param text The input text.
   * @return The encoded token ids.
   */
  virtual std::vector<int32_t> Encode(const std::string &text) = 0;

  /**
   * @brief Encode text into ids with special tokens option.
   * @param text The input text.
   * @param add_special_tokens Whether to add special tokens.
   * @return The encoded token ids.
   */
  virtual std::vector<int32_t> Encode(const std::string &text,
                                      bool add_special_tokens) = 0;

  /**
   * @brief Encode a batch of texts into ids.
   * @param texts The input texts.
   * @return The encoded token ids.
   */
  virtual std::vector<std::vector<int32_t>>
  EncodeBatch(const std::vector<std::string> &texts) {
    // Fall back when the derived class does not implement this function.
    std::vector<std::vector<int32_t>> ret;
    ret.reserve(texts.size());
    for (const auto &text : texts) {
      ret.push_back(Encode(text));
    }
    return ret;
  }

  /**
   * @brief Decode token ids into text.
   * @param text The token ids.
   * @return The decoded text.
   */
  virtual std::string Decode(const std::vector<int32_t> &ids) = 0;

  /**
   * @brief Returns the vocabulary size. Special tokens are considered.
   */
  virtual size_t GetVocabSize() = 0;

  /**
   * @brief Convert the given id to its corresponding token if it exists. If
   * not, return an empty string.
   */
  virtual std::string IdToToken(int32_t token_id) = 0;

  /**
   * @brief Convert the given token to its corresponding id if it exists. If
   * not, return -1.
   */
  virtual int32_t TokenToId(const std::string &token) = 0;

  //---------------------------------------------------
  // Factory functions from byte-blobs
  // These factory function takes in in-memory blobs
  // so the library can be independent from filesystem
  //---------------------------------------------------
  /**
   * @brief Create HF tokenizer from a single in-memory json blob.
   *
   * @param json_blob The json blob.
   * @return The created tokenzier.
   */
  static std::unique_ptr<Tokenizer> FromBlobJSON(const std::string &json_blob);

  /**
   * @brief Create HF tokenizer from a binary blob written by SaveBinary().
   *
   * The blob carries a magic header / version so callers can recover from a
   * mismatched format by falling back to FromBlobJSON. Returns nullptr when
   * the blob does not match the expected format.
   *
   * @param bin_blob The binary blob (full file contents including header).
   * @return The created tokenizer, or nullptr if the blob is invalid.
   */
  static std::unique_ptr<Tokenizer>
  FromBlobBinary(const std::string &bin_blob);

  /**
   * @brief Persist this tokenizer to disk in fast-loading binary format.
   *
   * Returns true on success. Default implementation returns false; concrete
   * tokenizer types that support binary serialization override this.
   *
   * @param path Destination file path.
   */
  virtual bool SaveBinary(const std::string &path) {
    (void)path;
    return false;
  }

  /**
   * @brief Unified file-based loader that prefers binary format.
   *
   * Resolution order:
   *   1. If `path` ends with ".bin", attempt FromBlobBinary on `path` only;
   *      throws on any failure (no JSON fallback when binary was explicitly
   *      requested).
   *   2. Otherwise, attempt to load `<path>.bin` first. If it exists and
   *      parses cleanly, use it.
   *   3. Otherwise, load `path` as JSON via FromBlobJSON. When `auto_cache`
   *      is true and the JSON load succeeded, the loaded tokenizer is dumped
   *      to `<path>.bin` so subsequent runs hit step 2.
   *
   * Magic-header / version mismatches in step 2 trigger a one-time warning
   * and silently fall through to step 3.
   *
   * @param path Path to the tokenizer.json (or tokenizer.bin) file.
   * @param auto_cache When true (default), produce a `.bin` cache next to the
   *                   JSON on first load.
   */
  static std::unique_ptr<Tokenizer> FromFile(const std::string &path,
                                             bool auto_cache = true);
  /**
   * @brief Create BPE tokenizer
   *
   * @param vocab_blob The blob that contains vocabs.
   * @param merges_blob The blob that contains the merges.
   * @param added_tokens The added tokens.
   * @return The created tokenizer.
   */
  static std::unique_ptr<Tokenizer>
  FromBlobByteLevelBPE(const std::string &vocab_blob,
                       const std::string &merges_blob,
                       const std::string &added_tokens = "");
  /**
   * @brief Create SentencePiece.
   *
   * @param model_blob The blob that contains vocabs.
   * @return The created tokenizer.
   */
  static std::unique_ptr<Tokenizer>
  FromBlobSentencePiece(const std::string &model_blob);
  /**
   * @brief Create RWKVWorldTokenizer.
   *
   * @param model_blob The blob that contains vocabs.
   * @return The created tokenizer.
   */
  static std::unique_ptr<Tokenizer>
  FromBlobRWKVWorld(const std::string &model_blob);
};

} // namespace tokenizers
#endif // TOKENIZERS_CPP_H_
