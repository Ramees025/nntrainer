# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NNTrainer is a C++17 software framework for on-device training and inference of neural networks on resource-constrained embedded devices (Tizen, Android, Ubuntu, Windows; ARM and x86_64). It supports on-device personalization (transfer/few-shot/continuous learning) and efficient LLM inference via memory-optimization features like **FSU (Flash Storage Utilization)**, **MoE cache**, and **proactive weight loading**.

The build system is **Meson + Ninja**, tests use **GTest/GMock**, and the project is ABI-sensitive with **strict warnings (`werror=true`)**.

## Build & Test Commands

### Standard Linux build
```bash
git submodule sync && git submodule update --init --depth 1
meson setup build
ninja -C build
ninja -C build install          # installs libs + headers under the configured prefix
ninja test -C build             # runs all enabled unit tests
```

### Running a single unit test
Each unit-test target under `test/unittest/` is a standalone GTest binary. After building:
```bash
./build/test/unittest/unittest_nntrainer_tensor                       # run all tests in one binary
./build/test/unittest/unittest_nntrainer_tensor --gtest_filter='TensorTest.*'
meson test -C build unittest_layers -v                                # run by Meson test name
```

### Common meson options (see `meson_options.txt` for the full list)
Pass with `-D<option>=<value>` to `meson setup` or `meson configure build`:
- `platform` — `none|tizen|yocto|android|windows`
- `enable-fp16`, `enable-blas`, `enable-openmp`, `enable-opencl`, `enable-cublas`
- `enable-fsu` (+ `fsu-path`) — flash-storage weight offloading for LLM inference
- `enable-biqgemm` (+ `biqgemm-path`), `enable-ruy`, `hgemm-experimental-kernel`
- `enable-tflite-interpreter`, `enable-onnx-interpreter`, `enable-nnstreamer-backbone`
- `arm-arch` (`none|armv7l|armv8.2-a|armv9.2-a`), `arm-march` (explicit `-march=…` override)
- `enable-test`, `enable-long-test`, `test-timeout`, `reduce-tolerance`
- `enable-app`, `install-app`, `enable-benchmarks`, `enable-profile`, `enable-trace`, `enable-debug`

### Android / Tizen / Windows
- Android: `ndk-build` via `jni/Android.mk.in` and the `prepare_*.sh` scripts in `jni/`; Android unit tests run with `tools/android_test.sh` then `adb shell` into `/data/local/tmp/nntr_android_test`.
- Tizen: `gbs build` (also executes unit tests). Spec is in `packaging/nntrainer.spec`.
- Windows: see `docs/getting-started-windows.md`; uses the `libiomp_win/` bundled OpenMP and MSVC runtime.
- Debian/Ubuntu packaging: `debuild -us -uc` driven by files in `debian/`.

### Formatting
C/C++ source is formatted with `clang-format` using the repo's `.clang-format`. Headers may deviate slightly from clang-format output and may break the 80-col rule; source files must be clang-format-clean. Do not bikeshed formatting — keep diffs minimal.

## Repository Layout

Top-level directories (each has a `meson.build`):
- `nntrainer/` — **core library**; all training/inference code lives here.
- `api/` — public APIs. `api/capi/` is the C API (the official Tizen surface) and `api/ccapi/` is the C++ API used on other platforms.
- `Applications/` — runnable examples and end-to-end models (CausalLM/LLM inference, ResNet, VGG, YOLOv2/v3, LLaMA, PicoGPT, MNIST, Tizen_native, Android, ONNX, TFlite_export, etc.). Most have their own `jni/` for Android and are also registered as meson subprojects.
- `test/` — GTest-based unit tests (`test/unittest/…`), C-API tests (`test/tizen_capi/`), C++-API tests (`test/ccapi/`), model-level tests (`test/unittest/models/`), and `test/input_gen/` Python scripts that generate golden tensors used as test fixtures.
- `nnstreamer/` — NNStreamer tensor-filter and tensor-trainer sub-plugins that expose NNTrainer to the NNStreamer pipeline.
- `benchmarks/` — microbenchmarks (Google Benchmark); enabled with `-Denable-benchmarks=true`.
- `jni/` — Android NDK glue (`Android.mk.in`, `Application.mk`, dependency-prep shell scripts).
- `debian/`, `packaging/` — Debian and Tizen RPM packaging.
- `tools/` — helper scripts: `android_test.sh`, cross-build helpers, Python utilities.
- `subprojects/` — Meson subprojects / wraps for third-party deps (GoogleTest, iniparser, CLBlast, ruy, benchmark, etc.).
- `configurations/` — sample `.ini` network configs used by examples and tests.
- `docs/` — developer docs (getting-started, coding-convention, components, memory-management, how-to-*).

## Core Library Architecture (`nntrainer/`)

Understanding the interaction of the layers below is usually required to change anything non-trivial. The typical flow for a training/inference step is:

**`NeuralNetwork` (models/) → `NetworkGraph` (graph/) → `LayerNode`s wrapping `Layer`s (layers/) → `Tensor`s managed by `Manager` and a `MemoryPool` (tensor/) → `Optimizer` (optimizers/)**.

### `nntrainer/models/` — top-level training/inference orchestration
- `neuralnet.h/.cpp` defines `NeuralNetwork`, the user-facing model class that owns the graph, optimizer, data buffers, and training loop.
- `model_loader.cpp` parses `.ini` configs (via `compiler/ini_interpreter`) into a `NeuralNetwork`.
- `dynamic_training_optimization.*` implements runtime-adaptive training choices (e.g., DTO skipping updates).
- `execution_mode.h` enumerates TRAIN / INFERENCE / VALIDATION modes.

### `nntrainer/graph/` — computation graph
- `network_graph.*` holds topologically-sorted `LayerNode`s, allocates tensors against the shared `Manager`, and drives forward / backward / apply-gradient phases.
- `graph_core.*` / `graph_node.h` are the generic DAG primitives; `connection.*` models an input/output edge with a tensor index.

### `nntrainer/layers/` — layer implementations
- `layer_devel.h` is the internal polymorphic base (`Layer`) that every built-in layer inherits, while `layer_impl.*` provides a convenience base with common property handling.
- `layer_node.*` is the graph-side wrapper: it owns the `Layer`, its `RunLayerContext` / `InitLayerContext`, property parsing (`common_properties.*`), and execution ordering.
- `layer_context.*` defines `InitLayerContext` (used during graph construction to request weights/tensors) and `RunLayerContext` (used at forward/backward time to access them).
- `loss/` contains loss layers; `cl_layers/` holds OpenCL-accelerated variants; `*cell*.cpp` pair with their recurrent wrappers (e.g., `lstmcell.cpp` + `lstm.cpp`).
- New layers register themselves via `app_context.cpp` (CPU path) or `cl_context.cpp` (OpenCL path); external plugins arrive through `plugged_layer.h`.

### `nntrainer/tensor/` — tensors, memory, and backends
- `Tensor` is a polymorphic front-end over typed storage backends: `float_tensor.*`, `half_tensor.*` (fp16), `char_tensor.*` (int8/uint8), `int4_tensor.*`, `q4_0_tensor.*`, `bcq_tensor.*`. Each backend implements the same set of elementary ops.
- `manager.*` is the tensor allocator-of-record: it tracks every weight, gradient, and activation tensor's lifetime and delegates physical allocation to a `MemoryPool` planned by one of the `*_planner.*` classes (`basic_planner`, `optimized_v1/v2/v3_planner`).
- `cache_pool.*`, `cache_elem.*`, `cache_loader.*` implement the **FSU** path: weights live on disk and are paged in/out of a pool on demand; `enable-fsu=true` activates this.
- `lazy_tensor.*` defers op evaluation so multiple tensor ops can be fused/scheduled by the graph.
- `cl_operations/` holds OpenCL tensor ops; `cpu_backend/` holds BLAS/NEON/AVX/hgemm/biqgemm kernels selected at build time.
- Kernel selection (BLAS vs OpenBLAS vs Ruy vs CUBLAS vs BiQGEMM vs custom hgemm) is driven by `meson_options.txt` flags and dispatched at compile time; do not hard-code a backend in new code.

### `nntrainer/compiler/` — model interchange / graph rewrites
- `ini_interpreter.*` reads the project's native `.ini` network configs (see `configurations/` and `docs/configuration-ini.md`).
- `tflite_interpreter.*` + `tflite_opnode.*` import/export TFLite; `tflite_export_realizer.*` is the export-side graph rewrite. Enabled by `enable-tflite-interpreter`.
- `onnx_interpreter.*` does the same for ONNX (`enable-onnx-interpreter`).
- `*_realizer.*` classes implement graph transforms (flatten inlining, activation fusion, batch-norm folding, recurrent unrolling, slicing, remap). The `Realizer` interface in `realizer.h` chains them.

### `nntrainer/optimizers/`
- `optimizer_devel.*` is the `Optimizer` base; built-ins are `sgd`, `adam`, `adamw`, `lion`. `optimizer_wrapped.*` + `plugged_optimizer.h` support external optimizers.
- Learning-rate schedulers (`lr_scheduler_*`) live next to the optimizers and are attached via `optimizer_context.*`.

### `nntrainer/opencl/` and `nntrainer/cl_context.*`
- Thin OpenCL wrapper (`opencl_buffer`, `opencl_kernel`, `opencl_program`, `opencl_command_queue_manager`, `opencl_context_manager`).
- Kernels live as `.cl` files under `nntrainer/opencl/CL/` (path configurable via the `opencl-kernel-path` meson option; `enable-kernel-caching` enables build-time kernel caching).
- `cl_context.*` mirrors `app_context.*` but for GPU layers and ops.

### `nntrainer/engine.*`, `nntrainer/context.h`, `nntrainer/app_context.*`
- `Engine` is the process-wide runtime singleton (thread pool via `nntr-num-threads`, OpenMP via `omp-num-threads`, backend selection).
- `AppContext` is the registry where layers, optimizers, LR schedulers, and data buffers register themselves by string keyword so the `.ini` / CCAPI / CAPI layers can instantiate them by name. Extending the framework usually means registering a factory here.

## Public API layers

- `api/ccapi/include/` — C++ API used by applications and tests; pure interface headers, implementations in `api/ccapi/src/`.
- `api/capi/include/nntrainer.h` — official C API (Tizen). `api/capi/src/` wraps the CCAPI. The CAPI is ABI-stable; breaking it requires explicit intent and versioning.
- `api/nntrainer-api-common.h` — shared enum/error definitions used by both APIs.
- Both are built into standalone shared libs (`capi-ml-training`, `ccapi-ml-training`) with pkg-config files under `packaging/`.

## Testing Conventions

- All tests use **GTest + GMock** via `nntrainer_test_util` (see `test/meson.build`).
- Tests are registered as meson `test()` entries and run with `ninja test -C build`. Each `unittest_*` is a separate binary so you can run it directly or via `meson test -C build <name>`.
- `test/input_gen/` contains Python generators (PyTorch/Keras-based) that produce reference tensors consumed by the C++ tests in `test/unittest/models/` and `test/unittest/layers/`. Regenerating fixtures is the prescribed way to update golden data — do not hand-edit binary test files.
- Long-running tests are gated behind `-Denable-long-test=true`; tolerances are tightened with `-Dreduce-tolerance=false`.
- Android test runs go through `tools/android_test.sh` which packages the built artifacts and runs them on a connected device.
- Coverage is produced by `test/unittestcoverage.py` over a build configured with `-fprofile-arcs -ftest-coverage`.

## Review / Editing Conventions (from `.cursorrules`)

Priority order when reviewing or making changes:
1. **Correctness & API/ABI safety** — preserve behavior unless the change is explicitly intended; public headers, exported symbols, v-tables, and struct layouts are ABI surface. Watch for UB, lifetime issues, alignment, strict-aliasing, signed overflow, and concurrency hazards.
2. **CI/portability** — the code must build across Tizen/Ubuntu/Android/Windows and ARM/x86_64. Don't introduce compiler- or libc-specific behavior without guards. Keep dependencies consistent with the existing build.
3. **Memory/ownership** — prefer RAII and clear ownership; no new leaks or double-frees; error paths must release resources.
4. **Performance** — in tensor ops and training loops, avoid unneeded allocations/copies, prefer const-correctness and move semantics, and justify/benchmark hot-path changes.
5. **Tests** — add narrow GTest cases for new behavior and for regressions you fix.

Do **not**: broaden refactors beyond the task, weaken warnings, invent new conventions, or reformat unrelated code. Warnings are errors here (`werror=true`).

Commits should summarize the change (bug + fix, or the feature), and include a Signed-off-by sign-off.
