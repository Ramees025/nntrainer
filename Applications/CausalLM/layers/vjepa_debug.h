// SPDX-License-Identifier: Apache-2.0
/**
 * @file   vjepa_debug.h
 * @date   10 June 2026
 * @brief  Shared debug utilities for VJEPA layer activation printing.
 *
 * Controlled by the VJEPA_DEBUG environment variable:
 *   VJEPA_DEBUG=1  → print activation stats for every layer
 *   unset or 0     → no debug output (default)
 */

#ifndef __VJEPA_DEBUG_H__
#define __VJEPA_DEBUG_H__

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace causallm {
namespace debug {

/** @brief Check if VJEPA debug mode is enabled. */
inline bool is_enabled() {
  static int cached = -1;
  if (cached < 0) {
    const char *env = std::getenv("VJEPA_DEBUG");
    cached = (env && std::string(env) == "1") ? 1 : 0;
  }
  return cached == 1;
}

/** @brief Print activation stats (min, max, mean, NaN/Inf count) for FP32. */
inline void print_activation_stats(const std::string &layer_name,
                                   const float *data, size_t count) {
  if (!is_enabled() || count == 0)
    return;
  size_t nan_count = 0, inf_count = 0;
  float vmin = std::numeric_limits<float>::max();
  float vmax = -std::numeric_limits<float>::max();
  double vsum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    float v = data[i];
    if (std::isnan(v)) {
      ++nan_count;
      continue;
    }
    if (std::isinf(v)) {
      ++inf_count;
      continue;
    }
    if (v < vmin)
      vmin = v;
    if (v > vmax)
      vmax = v;
    vsum += v;
  }
  size_t valid = count - nan_count - inf_count;
  double mean = valid > 0 ? vsum / valid : 0.0;
  std::cout << "[VJEPA_DEBUG] " << layer_name << " output (" << count
            << " elems): min=" << std::setprecision(6) << vmin
            << ", max=" << vmax << ", mean=" << mean
            << ", NaN=" << nan_count << ", Inf=" << inf_count << std::endl;
  // Print first 5 values
  std::cout << "[VJEPA_DEBUG] " << layer_name << " first 5: ";
  size_t n_print = count > 5 ? 5 : count;
  for (size_t i = 0; i < n_print; ++i)
    std::cout << data[i] << " ";
  std::cout << std::endl;
}

#ifdef ENABLE_FP16
/** @brief Print activation stats for FP16. */
inline void print_activation_stats(const std::string &layer_name,
                                   const _FP16 *data, size_t count) {
  if (!is_enabled() || count == 0)
    return;
  size_t nan_count = 0, inf_count = 0;
  float vmin = std::numeric_limits<float>::max();
  float vmax = -std::numeric_limits<float>::max();
  double vsum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    float v = static_cast<float>(data[i]);
    if (std::isnan(v)) {
      ++nan_count;
      continue;
    }
    if (std::isinf(v)) {
      ++inf_count;
      continue;
    }
    if (v < vmin)
      vmin = v;
    if (v > vmax)
      vmax = v;
    vsum += v;
  }
  size_t valid = count - nan_count - inf_count;
  double mean = valid > 0 ? vsum / valid : 0.0;
  std::cout << "[VJEPA_DEBUG] " << layer_name << " output (" << count
            << " elems, FP16): min=" << std::setprecision(6) << vmin
            << ", max=" << vmax << ", mean=" << mean
            << ", NaN=" << nan_count << ", Inf=" << inf_count << std::endl;
  size_t n_print = count > 5 ? 5 : count;
  std::cout << "[VJEPA_DEBUG] " << layer_name << " first 5: ";
  for (size_t i = 0; i < n_print; ++i)
    std::cout << static_cast<float>(data[i]) << " ";
  std::cout << std::endl;
}
#endif

} // namespace debug
} // namespace causallm

#endif /* __VJEPA_DEBUG_H__ */
