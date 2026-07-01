// SPDX-License-Identifier: Apache-2.0
/**
 * HexKL SDKL backend for nntrainer.
 *
 * Offloads prefill-phase FP32 SGEMM (M > 1) to the Snapdragon HMX unit via
 * the HexKL CPU Macro API (libsdkl.so / sdkl.h).
 *
 * Design:
 *   g_wh_cache  — per-weight WH-layout FP16 buffer (regular malloc, keyed on
 *                 (B_ptr, N, K, TransB)).  Built on first call then reused.
 *                 Avoids exhausting sdkl_npu_alloc pool (no per-weight alloc).
 *   g_W_buf     — single sdkl_npu_alloc buffer, sized for the largest weight.
 *                 Per dispatch: WH-layout is memcpy'd here from g_wh_cache
 *                 (~0.24ms) instead of full FP32→FP16+WH conversion (~3.5ms).
 *   g_X_stage   — malloc, largest padded activation [M_pad × K].
 *   g_C_stage   — malloc, largest padded output    [M_pad × N].
 *
 * HMX constraint: n_row (M) must be a multiple of 32; rows are zero-padded.
 *
 * HMX lock: acquired only around the DSP dispatch (not held idle) so CDSP
 * power-state does not throttle CPU during decode between prefill steps.
 */

#ifdef USE_HMX

// remote.h constants needed by sdkl.h enum body — define them inline to
// avoid depending on the full Hexagon SDK headers at build time.
#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif
#ifndef CDSP1_DOMAIN_ID
#define CDSP1_DOMAIN_ID 4
#endif

#include "hexkl_backend.h"

// sdkl.h uses C99 'restrict' which is not a C++ keyword. Map it.
#ifndef restrict
#define restrict __restrict__
#endif
#include <sdkl.h>
#undef restrict

#include <android/log.h>
#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <time.h>
#include <unordered_map>

#define HEXKL_LOG_TAG "nntr_hexkl"
#define HEXKL_LOGI(...) \
  __android_log_print(ANDROID_LOG_INFO,  HEXKL_LOG_TAG, __VA_ARGS__)
#define HEXKL_LOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, HEXKL_LOG_TAG, __VA_ARGS__)

namespace nntrainer {
namespace hexkl {

// ---------------------------------------------------------------------------
// Per-weight WH-layout cache (regular malloc — avoids sdkl_npu_alloc pool)
// ---------------------------------------------------------------------------

struct WeightKey {
  uintptr_t    ptr;
  unsigned int n_col;   // N
  unsigned int n_inner; // K
  bool         transB;
  bool operator==(const WeightKey &o) const {
    return ptr == o.ptr && n_col == o.n_col &&
           n_inner == o.n_inner && transB == o.transB;
  }
};

struct WeightKeyHash {
  size_t operator()(const WeightKey &k) const noexcept {
    size_t h = std::hash<uintptr_t>{}(k.ptr);
    h ^= std::hash<unsigned>{}(k.n_col)   + 0x9e3779b9u + (h << 6) + (h >> 2);
    h ^= std::hash<unsigned>{}(k.n_inner) + 0x9e3779b9u + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(k.transB)      + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h;
  }
};

// Values are regular malloc'd [N×K] FP16 in WH layout.
static std::unordered_map<WeightKey, _Float16 *, WeightKeyHash> g_wh_cache;
static std::mutex g_cache_mutex;

// ---------------------------------------------------------------------------
// Persistent shared buffers — grown as needed, freed at finalize()
// ---------------------------------------------------------------------------

static std::mutex g_hmx_mutex;   // serializes HMX dispatch + g_W_buf/stage access

// W: single sdkl_npu_alloc buffer (DSP-accessible). WH-layout is memcpy'd
// from g_wh_cache each dispatch — much cheaper than full FP32→FP16+WH.
static _Float16 *g_W_buf        = nullptr;
static size_t    g_W_buf_bytes  = 0;

// X / C: plain malloc (per SDKL example — only W requires sdkl_npu_alloc)
static float    *g_X_stage      = nullptr;
static size_t    g_X_stage_bytes = 0;
static float    *g_C_stage      = nullptr;
static size_t    g_C_stage_bytes = 0;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

static bool           g_initialized = false;
static int            g_domain      = CDSP_DOMAIN_ID;
static std::once_flag g_init_flag;

static inline int64_t now_us() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000LL;
}

// ---------------------------------------------------------------------------
// NEON-vectorized FP32→FP16 conversion with optional transpose.
//
// For TransB=true  (B is [N×K] row-major, stride ldb):
//   dst[n*K + k] = (f16) B[n*ldb + k]   — sequential reads, no reorder.
//
// For TransB=false (B is [K×N] row-major, stride ldb):
//   dst[n*K + k] = (f16) B[k*ldb + n]   — transpose required.
//   Uses blocked traversal (BK×BN tiles) so B reads stay sequential
//   within each tile; NEON converts 4 FP32 → 4 FP16 per cycle.
// ---------------------------------------------------------------------------
static void f32_to_f16_convert(
    _Float16 *__restrict__ dst,
    const float *__restrict__ B,
    unsigned N, unsigned K, unsigned ldb, bool TransB)
{
  if (TransB) {
    // B is [N×K] — straight copy row by row, NEON 8-wide
    for (unsigned n = 0; n < N; ++n) {
      const float *brow = B + (size_t)n * ldb;
      _Float16    *drow = dst + (size_t)n * K;
      unsigned k = 0;
      for (; k + 8 <= K; k += 8) {
        float32x4_t v0 = vld1q_f32(brow + k);
        float32x4_t v1 = vld1q_f32(brow + k + 4);
        vst1q_f16((__fp16 *)(drow + k),
                  vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
      }
      for (; k < K; ++k) drow[k] = (_Float16)brow[k];
    }
  } else {
    // B is [K×N] — blocked transpose + NEON 4-wide conversion.
    // Outer tile over k keeps B reads sequential; inner tile over n
    // keeps the dst scatter writes L1-local.
    const unsigned BK = 4, BN = 32;
    for (unsigned k0 = 0; k0 < K; k0 += BK) {
      unsigned k_end = k0 + BK < K ? k0 + BK : K;
      for (unsigned n0 = 0; n0 < N; n0 += BN) {
        unsigned n_end = n0 + BN < N ? n0 + BN : N;
        for (unsigned k = k0; k < k_end; ++k) {
          const float *brow = B + (size_t)k * ldb + n0;
          unsigned n = n0;
          for (; n + 4 <= n_end; n += 4) {
            float32x4_t v = vld1q_f32(brow + (n - n0));
            __fp16 tmp[4]; vst1_f16(tmp, vcvt_f16_f32(v));
            dst[(size_t)n     * K + k] = (_Float16)tmp[0];
            dst[(size_t)(n+1) * K + k] = (_Float16)tmp[1];
            dst[(size_t)(n+2) * K + k] = (_Float16)tmp[2];
            dst[(size_t)(n+3) * K + k] = (_Float16)tmp[3];
          }
          for (; n < n_end; ++n)
            dst[(size_t)n * K + k] = (_Float16)brow[n - n0];
        }
      }
    }
  }
}

// Accumulated timing (reset each forward pass by tracking call count).
static int64_t g_t_cache_us = 0, g_t_memcpy_us = 0,
               g_t_lock_us = 0, g_t_dsp_us = 0, g_t_unlock_us = 0;
static int     g_call_count = 0;

void initialize() {
  std::call_once(g_init_flag, []() {
    int ret = sdkl_npu_initialize(g_domain, nullptr, nullptr);
    if (ret != 0) {
      HEXKL_LOGE("sdkl_npu_initialize failed: %d", ret);
      return;
    }
    // sdkl_npu_initialize locks HMX by default.  Release the lock
    // immediately; sgemm_hmx will acquire it per-dispatch so that
    // the CDSP does not stay at high-power between prefill matmuls
    // (which would throttle the CPU during decode).
    sdkl_npu_unlock_hmx(g_domain);
    char ver[SDKL_VERSION_STR_LEN] = {};
    sdkl_npu_get_version(g_domain, ver);
    HEXKL_LOGI("HexKL ready: %s", ver);
    g_initialized = true;
  });
}

void finalize() {
  if (!g_initialized) return;
  g_initialized = false;

  {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    for (auto &kv : g_wh_cache) free(kv.second);
    g_wh_cache.clear();
  }

  std::lock_guard<std::mutex> lock(g_hmx_mutex);
  if (g_W_buf)   { sdkl_npu_free(g_W_buf);   g_W_buf   = nullptr; g_W_buf_bytes   = 0; }
  if (g_X_stage) { free(g_X_stage);           g_X_stage = nullptr; g_X_stage_bytes = 0; }
  if (g_C_stage) { free(g_C_stage);           g_C_stage = nullptr; g_C_stage_bytes = 0; }

  sdkl_npu_finalize(g_domain);
  HEXKL_LOGI("HexKL finalized — total sgemm_hmx calls: %d", g_call_count);
}

// ---------------------------------------------------------------------------
// preload_weight_f32 / is_weight_cached — warm the WH cache without dispatch.
// Call this for each FC weight during model loading or decode steps so that
// the first real prefill (M > 1) hits the cache and skips the build cost.
// ---------------------------------------------------------------------------

bool is_weight_cached(bool TransB, unsigned N, unsigned K, const float *B) {
  if (!g_initialized) return false;
  WeightKey key{reinterpret_cast<uintptr_t>(B), N, K, TransB};
  std::lock_guard<std::mutex> cl(g_cache_mutex);
  return g_wh_cache.count(key) != 0;
}

void preload_weight_f32(bool TransB, unsigned N, unsigned K,
                        const float *B, unsigned ldb) {
  if (!g_initialized) return;
  WeightKey key{reinterpret_cast<uintptr_t>(B), N, K, TransB};
  {
    std::lock_guard<std::mutex> cl(g_cache_mutex);
    if (g_wh_cache.count(key)) return;  // already cached
  }
  const size_t w_bytes = (size_t)N * K * sizeof(_Float16);
  _Float16 *wh_buf = static_cast<_Float16 *>(malloc(w_bytes));
  if (!wh_buf) return;
  f32_to_f16_convert(wh_buf, B, N, K, ldb, TransB);
  if (sdkl_cpu_rm_to_wh_f16_inplace(N, K, wh_buf) != 0) { free(wh_buf); return; }
  std::lock_guard<std::mutex> cl(g_cache_mutex);
  auto [it, inserted] = g_wh_cache.emplace(key, wh_buf);
  if (!inserted) free(wh_buf);
}

// ---------------------------------------------------------------------------
// sgemm_hmx
// ---------------------------------------------------------------------------

bool sgemm_hmx(bool TransB,
               unsigned int M, unsigned int N, unsigned int K,
               const float *A, unsigned int lda,
               const float *B, unsigned int ldb,
               float *C, unsigned int ldc) {
  if (!g_initialized) return false;

  // HMX requires n_row to be a multiple of 32.
  const unsigned int M_pad    = ((M + 31u) / 32u) * 32u;
  const size_t       w_bytes  = (size_t)N * K * sizeof(_Float16);
  const size_t       x_bytes  = (size_t)M_pad * K * sizeof(float);
  const size_t       c_bytes  = (size_t)M_pad * N * sizeof(float);

  int64_t t0, t1, t2, t3, t4, t5;
  t0 = now_us();

  // Look up (or build) the WH-layout FP16 weight in regular malloc memory.
  WeightKey key{reinterpret_cast<uintptr_t>(B), N, K, TransB};
  _Float16 *wh_buf = nullptr;
  {
    std::lock_guard<std::mutex> cache_lock(g_cache_mutex);
    auto it = g_wh_cache.find(key);
    if (it != g_wh_cache.end()) wh_buf = it->second;
  }
  if (!wh_buf) {
    wh_buf = static_cast<_Float16 *>(malloc(w_bytes));
    if (!wh_buf) { HEXKL_LOGE("malloc wh_buf(%zu) failed", w_bytes); return false; }

    // NEON-vectorized FP32→FP16 (+ transpose for TransB=false).
    f32_to_f16_convert(wh_buf, B, N, K, ldb, TransB);
    int64_t t_cvt = now_us();

    if (sdkl_cpu_rm_to_wh_f16_inplace(N, K, wh_buf) != 0) {
      HEXKL_LOGE("sdkl_cpu_rm_to_wh_f16_inplace failed N=%u K=%u", N, K);
      free(wh_buf);
      return false;
    }
    int64_t t_wh = now_us();
    // Split log on first build: shows convert vs WH-layout cost separately.
    HEXKL_LOGI("WH build N=%u K=%u: cvt=%lldus wh=%lldus",
               N, K, (long long)(t_cvt - t0), (long long)(t_wh - t_cvt));

    {
      std::lock_guard<std::mutex> cache_lock(g_cache_mutex);
      auto [it, inserted] = g_wh_cache.emplace(key, wh_buf);
      if (!inserted) { free(wh_buf); wh_buf = it->second; }
    }
  }

  t1 = now_us();

  std::lock_guard<std::mutex> lock(g_hmx_mutex);

  // Grow W buffer (sdkl_npu_alloc — single buffer for DSP access) if needed.
  if (w_bytes > g_W_buf_bytes) {
    if (g_W_buf) sdkl_npu_free(g_W_buf);
    g_W_buf = nullptr; g_W_buf_bytes = 0;
    if (sdkl_npu_alloc(w_bytes, reinterpret_cast<void **>(&g_W_buf)) != 0 || !g_W_buf) {
      HEXKL_LOGE("sdkl_npu_alloc W(%zu) failed N=%u K=%u", w_bytes, N, K);
      return false;
    }
    g_W_buf_bytes = w_bytes;
    HEXKL_LOGI("W_buf grown to %zu bytes", w_bytes);
  }
  // Copy WH-layout from cached malloc to the DSP-accessible sdkl_npu_alloc buffer.
  memcpy(g_W_buf, wh_buf, w_bytes);

  // Grow X staging buffer (malloc) if needed.
  if (x_bytes > g_X_stage_bytes) {
    free(g_X_stage);
    g_X_stage = static_cast<float *>(malloc(x_bytes));
    if (!g_X_stage) { g_X_stage_bytes = 0; HEXKL_LOGE("malloc X_stage(%zu) failed", x_bytes); return false; }
    g_X_stage_bytes = x_bytes;
  }

  // Grow C staging buffer (malloc) if needed.
  if (c_bytes > g_C_stage_bytes) {
    free(g_C_stage);
    g_C_stage = static_cast<float *>(malloc(c_bytes));
    if (!g_C_stage) { g_C_stage_bytes = 0; HEXKL_LOGE("malloc C_stage(%zu) failed", c_bytes); return false; }
    g_C_stage_bytes = c_bytes;
  }

  // Copy A into g_X_stage; zero-pad extra rows for M alignment.
  memcpy(g_X_stage, A, (size_t)M * K * sizeof(float));
  if (M_pad > M)
    memset(g_X_stage + (size_t)M * K, 0, (size_t)(M_pad - M) * K * sizeof(float));

  t2 = now_us();

  // Lock HMX only for the duration of the DSP dispatch.
  if (sdkl_npu_lock_hmx(g_domain) != 0) {
    HEXKL_LOGE("sdkl_npu_lock_hmx failed");
    return false;
  }
  t3 = now_us();

  int ret = sdkl_npu_mm_f32f16_f32(g_domain, (int)M_pad, (int)N, (int)K,
                                    g_C_stage, g_X_stage, g_W_buf);
  t4 = now_us();
  sdkl_npu_unlock_hmx(g_domain);
  t5 = now_us();

  g_t_cache_us  += t1 - t0;
  g_t_memcpy_us += t2 - t1;
  g_t_lock_us   += t3 - t2;
  g_t_dsp_us    += t4 - t3;
  g_t_unlock_us += t5 - t4;
  ++g_call_count;

  // Log first call and then every 196 calls.
  if (g_call_count == 1 || g_call_count % 196 == 0) {
    HEXKL_LOGI("TIMING call=%d M=%u N=%u K=%u: cache=%lldus memcpy=%lldus lock=%lldus dsp=%lldus unlock=%lldus",
               g_call_count, M, N, K,
               (long long)(g_t_cache_us), (long long)(g_t_memcpy_us),
               (long long)(g_t_lock_us),  (long long)(g_t_dsp_us),
               (long long)(g_t_unlock_us));
  }
  if (g_call_count % 196 == 0) {
    HEXKL_LOGI("TIMING PASS call=%d total=%lldms",
               g_call_count,
               (long long)((g_t_cache_us+g_t_memcpy_us+g_t_lock_us+g_t_dsp_us+g_t_unlock_us)/1000));
    g_t_cache_us = g_t_memcpy_us = g_t_lock_us = g_t_dsp_us = g_t_unlock_us = 0;
  }

  if (ret == 0) {
    // Copy real M rows back; ldc==N is guaranteed by the dispatch guard.
    memcpy(C, g_C_stage, (size_t)M * N * sizeof(float));
  } else {
    HEXKL_LOGE("sdkl_npu_mm_f32f16_f32 failed: %d (M=%u M_pad=%u N=%u K=%u)",
               ret, M, M_pad, N, K);
  }

  return ret == 0;
}

// ---------------------------------------------------------------------------
// shgemm_hmx: FP16 weights already available — just WH-layout + dispatch
// ---------------------------------------------------------------------------

bool shgemm_hmx(bool TransB,
                unsigned int M, unsigned int N, unsigned int K,
                const float *A, unsigned int lda,
                const __fp16 *B_f16, unsigned int ldb,
                float *C, unsigned int ldc) {
  if (!g_initialized) return false;

  const unsigned int M_pad   = ((M + 31u) / 32u) * 32u;
  const size_t       w_bytes = (size_t)N * K * sizeof(_Float16);
  const size_t       x_bytes = (size_t)M_pad * K * sizeof(float);
  const size_t       c_bytes = (size_t)M_pad * N * sizeof(float);

  int64_t t0 = now_us();

  // Look up (or build) WH-layout cache — B is already FP16, so just transpose
  // (if needed) + apply WH layout.  No FP32→FP16 conversion.
  WeightKey key{reinterpret_cast<uintptr_t>(B_f16), N, K, TransB};
  _Float16 *wh_buf = nullptr;
  {
    std::lock_guard<std::mutex> cl(g_cache_mutex);
    auto it = g_wh_cache.find(key);
    if (it != g_wh_cache.end()) wh_buf = it->second;
  }
  if (!wh_buf) {
    wh_buf = static_cast<_Float16 *>(malloc(w_bytes));
    if (!wh_buf) { HEXKL_LOGE("malloc wh_buf(%zu) failed", w_bytes); return false; }

    // B is already FP16 — only transpose if needed (no FP32→FP16 conversion).
    if (TransB) {
      for (unsigned n = 0; n < N; ++n) {
        const __fp16 *brow = B_f16 + (size_t)n * ldb;
        _Float16     *drow = wh_buf + (size_t)n * K;
        unsigned k = 0;
        for (; k + 8 <= K; k += 8)
          vst1q_f16((__fp16 *)(drow + k), vld1q_f16(brow + k));
        for (; k < K; ++k) drow[k] = (_Float16)brow[k];
      }
    } else {
      const unsigned BK = 4, BN = 32;
      for (unsigned k0 = 0; k0 < K; k0 += BK) {
        unsigned k_end = k0 + BK < K ? k0 + BK : K;
        for (unsigned n0 = 0; n0 < N; n0 += BN) {
          unsigned n_end = n0 + BN < N ? n0 + BN : N;
          for (unsigned k = k0; k < k_end; ++k) {
            const __fp16 *brow = B_f16 + (size_t)k * ldb + n0;
            unsigned n = n0;
            for (; n + 8 <= n_end; n += 8) {
              float16x8_t v = vld1q_f16(brow + (n - n0));
              __fp16 tmp[8]; vst1q_f16(tmp, v);
              for (int i = 0; i < 8; ++i)
                wh_buf[(size_t)(n+i) * K + k] = (_Float16)tmp[i];
            }
            for (; n < n_end; ++n)
              wh_buf[(size_t)n * K + k] = (_Float16)brow[n - n0];
          }
        }
      }
    }

    if (sdkl_cpu_rm_to_wh_f16_inplace(N, K, wh_buf) != 0) {
      HEXKL_LOGE("sdkl_cpu_rm_to_wh_f16_inplace failed N=%u K=%u", N, K);
      free(wh_buf); return false;
    }
    {
      std::lock_guard<std::mutex> cl(g_cache_mutex);
      auto [it, inserted] = g_wh_cache.emplace(key, wh_buf);
      if (!inserted) { free(wh_buf); wh_buf = it->second; }
    }
  }

  int64_t t1 = now_us();

  std::lock_guard<std::mutex> lock(g_hmx_mutex);

  if (w_bytes > g_W_buf_bytes) {
    if (g_W_buf) sdkl_npu_free(g_W_buf);
    g_W_buf = nullptr; g_W_buf_bytes = 0;
    if (sdkl_npu_alloc(w_bytes, reinterpret_cast<void **>(&g_W_buf)) != 0 || !g_W_buf) {
      HEXKL_LOGE("sdkl_npu_alloc W(%zu) failed", w_bytes); return false;
    }
    g_W_buf_bytes = w_bytes;
    HEXKL_LOGI("W_buf grown to %zu bytes (shgemm path)", w_bytes);
  }
  memcpy(g_W_buf, wh_buf, w_bytes);

  if (x_bytes > g_X_stage_bytes) {
    free(g_X_stage);
    g_X_stage = static_cast<float *>(malloc(x_bytes));
    if (!g_X_stage) { g_X_stage_bytes = 0; return false; }
    g_X_stage_bytes = x_bytes;
  }
  if (c_bytes > g_C_stage_bytes) {
    free(g_C_stage);
    g_C_stage = static_cast<float *>(malloc(c_bytes));
    if (!g_C_stage) { g_C_stage_bytes = 0; return false; }
    g_C_stage_bytes = c_bytes;
  }

  memcpy(g_X_stage, A, (size_t)M * K * sizeof(float));
  if (M_pad > M)
    memset(g_X_stage + (size_t)M * K, 0, (size_t)(M_pad - M) * K * sizeof(float));

  int64_t t2 = now_us();

  if (sdkl_npu_lock_hmx(g_domain) != 0) {
    HEXKL_LOGE("sdkl_npu_lock_hmx failed"); return false;
  }
  int64_t t3 = now_us();

  int ret = sdkl_npu_mm_f32f16_f32(g_domain, (int)M_pad, (int)N, (int)K,
                                    g_C_stage, g_X_stage, g_W_buf);
  int64_t t4 = now_us();
  sdkl_npu_unlock_hmx(g_domain);
  int64_t t5 = now_us();

  g_t_cache_us  += t1 - t0;
  g_t_memcpy_us += t2 - t1;
  g_t_lock_us   += t3 - t2;
  g_t_dsp_us    += t4 - t3;
  g_t_unlock_us += t5 - t4;
  ++g_call_count;

  if (g_call_count == 1 || g_call_count % 196 == 0) {
    HEXKL_LOGI("SH_TIMING call=%d M=%u N=%u K=%u: cache=%lldus memcpy=%lldus lock=%lldus dsp=%lldus unlock=%lldus",
               g_call_count, M, N, K,
               (long long)(g_t_cache_us), (long long)(g_t_memcpy_us),
               (long long)(g_t_lock_us),  (long long)(g_t_dsp_us),
               (long long)(g_t_unlock_us));
  }
  if (g_call_count % 196 == 0) {
    HEXKL_LOGI("SH_PASS call=%d total=%lldms",
               g_call_count,
               (long long)((g_t_cache_us+g_t_memcpy_us+g_t_lock_us+g_t_dsp_us+g_t_unlock_us)/1000));
    g_t_cache_us = g_t_memcpy_us = g_t_lock_us = g_t_dsp_us = g_t_unlock_us = 0;
  }

  if (ret == 0) {
    memcpy(C, g_C_stage, (size_t)M * N * sizeof(float));
  } else {
    HEXKL_LOGE("sdkl_npu_mm_f32f16_f32 failed: %d (M=%u M_pad=%u N=%u K=%u)",
               ret, M, M_pad, N, K);
  }
  return ret == 0;
}

} // namespace hexkl
} // namespace nntrainer

#endif // USE_HMX
