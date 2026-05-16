# Vendor Backend Dispatch Architecture

This document describes how nntrainer dispatches tensor operations to
the right vendor backend (CPU NEON / x86 AVX / OpenCL / QNN-NPU)
without `#ifdef` at the call site, and the design choices that
shaped the API.

It is the contract every new backend must follow, and the rationale
for the parts of the design that are non-obvious.

> **Companion docs:** [`PLUGGABLE_EN.md`](./PLUGGABLE_EN.md) (English) and
> [`PLUGGABLE_KO.md`](./PLUGGABLE_KO.md) (Korean) are step-by-step
> walkthroughs of the same machinery with diagrams, aimed at onboarding.
> This file is the *contract and rationale*; those are the *tutorial*.

---

## 1. The dispatch chain

```
Engine                            (process-wide singleton)
  └─ Context                      (one per vendor: cpu / gpu / qnn)
       └─ ContextData             (per-vendor metadata blob)
            └─ ComputeOps*        (virtual interface, vendor subclass)
                 └─ kernels       (sgemm, ele_mul, gemm_q4_0, ...)
```

Reading top-to-bottom:

* **Engine** holds a registry of `Context` objects, keyed by name
  (`"cpu"`, `"gpu"`, `"qnn"`). At process startup it calls
  `ensureComputeOps()` once to bind the global CPU `g_compute_ops`,
  then registers each available `Context`. Registration has two
  entry points, both named `Engine::registerContext`:
  - **built-in:** `registerContext(name, ctx*)` — used for CPU
    (`AppContext`) and, when `enable-opencl`, GPU (`ClContext`).
  - **plugin `.so`:** `registerContext(library_path, base_path)`
    — `dlopen`s the shared object, resolves the
    `extern "C" ml_train_context_pluggable` symbol
    (a `ContextPluggable{createfunc, destroyfunc}` defined in
    `context.h`), calls `createfunc()`, and registers the returned
    Context under its `getName()`. QNN ships this way:
    `engine.cpp` does `registerContext("libqnn_context.so", "")`
    under `enable-npu`, decoupling the QNN SDK from the core build.

  Which Contexts exist at all is therefore a **build-time** decision
  (meson options: `enable-opencl`, `enable-npu`, and the per-arch
  CPU split below — and the same gate is how future S.LSI / MediaTek
  / other vendor backends will be added). The call site never
  branches on vendor; only the registered set differs per build.

* **Context** is the user-facing entry point for a vendor backend.
  Layers, optimizers, and tensor allocations all flow through a
  Context. Each Context owns a `std::shared_ptr<ContextData>`.

* **ContextData** is a thin polymorphic container for per-vendor
  state — `compute_ops` pointer, mem allocator, and (when
  subclassed) anything else the backend needs (e.g. an OpenCL
  command queue, a QNN backend handle). Tensors carry a
  `shared_ptr<ContextData>` so they remember which backend they
  belong to.

* **ComputeOps** is the dispatch interface — an abstract C++ class
  with one virtual method per kernel (`sgemm_fp32`, `ele_mul_fp32`,
  `gemm_q4_0_batch_fp32`, ...). Concrete subclasses (`CpuComputeOps`,
  `ClComputeOps`) override what they implement. Default method
  bodies throw `std::runtime_error("not implemented")` so a missing
  override surfaces immediately rather than silently producing
  garbage.

* **Kernels** are the actual code that runs (NEON intrinsics,
  OpenCL kernel launches, QNN graph submissions, etc.). They live
  inside the ComputeOps subclass methods.

  **Per-arch CPU split (build-time).** The CPU backend is *not* one
  body of code. `nntrainer/tensor/cpu_backend/` ships three sibling
  implementations — `arm/` (NEON), `x86/` (AVX2 + ggml/BLAS),
  `fallback/` (scalar) — and `cpu_backend/meson.build` picks exactly
  one via `host_machine.cpu_family()`: `arm`/`aarch64`/android →
  `arm/`, `x86_64`/`x86` → `x86/`, else `fallback/`. Each provides
  its own `init_backend()` that ends with
  `g_compute_ops = get_cpu_ops()` (the single `CpuComputeOps` in
  `cpu_ops_table.cpp`). So at the *call site* there is still one
  `g_compute_ops`; what changed is **which arch's kernels were
  compiled into it**. This is the same build-time-selection
  principle as vendor Context gating above — CPU arch, GPU, NPU and
  future vendors are all chosen when the binary is built/loaded, not
  branched at the call site.

---

## 2. Where ContextData lives on a Tensor

`std::shared_ptr<ContextData> ct_data_` is a member of `TensorBase`
(not the public `Tensor` Pimpl wrapper). This places it alongside
`dim`, `strides`, and `data` — the rest of the storage metadata.

`Tensor` is a thin handle that forwards `setContextData` /
`getContextData` to the underlying `TensorBase` instance. The user
sees a flat `tensor.setContextData(ct)` API; the impl lives where
the kernel call actually happens.

**Why TensorBase, not Tensor:**
1. State belongs in the Pimpl impl, not the wrapper. Putting it on
   `Tensor` violates the pattern and forced custom copy/assign
   logic, an `inheritContextDataTo` helper, and a special
   `CREATE_IF_EMPTY_DIMS` macro to keep ct_data alive across
   in-place reallocation. All of that disappears when ct_data
   lives where storage lives.
2. `FloatTensor::dot()` and friends can resolve their dispatch ops
   directly via `getOps()` (a `TensorBase` helper) — no `ops`
   parameter threaded through every virtual signature, no extra
   indirection.
3. PyTorch's `TensorImpl::device_`, TF's `Tensor::device()`, JAX's
   buffers all carry the device on the impl, not on a separate
   handle. We follow the same convention.

**When ct_data gets attached:** at layer compile time. In
`network_graph.cpp::finalizeContext()`, the engine fetches the
matching context (`getRegisteredContext(layer_compute_engine)`),
pulls its `ContextData`, and hands it to
`LayerNode::configureRunContext`, which stamps every weight,
input, output, and persistent tensor of that layer with it. By
the time `forwarding()` runs, every tensor knows its backend.

**Inheritance contract:** binary/unary ops propagate the receiver's
ct_data to the result tensor via `inheritContextTo`. So
`auto t = a.dot(b); t.add(c);` keeps the chain on the same
backend automatically. The propagation happens AFTER the kernel
call because `CREATE_IF_EMPTY_DIMS` may reallocate the output
tensor mid-kernel and discard whatever was set before.

---

## 3. Cross-vendor mismatch policy

When `a.dot(b)` is called and both operands have a non-null
ContextData that points to different instances (e.g. one CPU-
resident tensor, one OpenCL-resident tensor), the op throws
`std::invalid_argument` with a clear message pointing the caller
at `Tensor::to()` for migration.

If either side is unattached (default state for tensors created
via the bare `Tensor(dim)` ctor), the check is permissive — fall
through to global `g_compute_ops`. This preserves backward
compatibility for every test and call site that never touches
ContextData.

`Tensor::to(target_ct)` is the migration entry point. In the
current host-shared-memory regime, it deep-copies and re-tags. When
true device-only memory backends land (CUDA, NPU with DMA-only
buffers), this is the one place that grows host↔device or
device↔device transfer logic.

---

## 4. ComputeOps interface — virtual, not function-pointer table

Earlier iterations used a `struct ComputeOps` filled with function
pointers, populated per-arch by an `*_ops_table.cpp` file. That
worked for stateless CPU kernels but broke for backends that need
per-instance state (an OpenCL `cl_command_queue`, a QNN graph
handle). Function-pointer signatures cannot smuggle a `this`
pointer without leaking into thread-local globals — which defeats
the whole "Context owns its own ops" design.

The current design makes `ComputeOps` an abstract C++ class with
~80 virtual methods. Concrete subclasses can hold backend state as
plain member variables and reach it from inside the override
without any indirection. The virtual call overhead (~1–3 ns) is
dwarfed by kernel cost (sgemm ≥ 100 µs); no measurable impact on
any layer-level test.

### "Pure virtual or default-throw?"

Default-throw. The 80 ops in `ComputeOps` have a long tail (~60 of
them are quantized variants, FP16 paths, accelerator-only batch
GEMMs). Forcing every backend to implement all of them with
`= 0` would mean OpenCL/QNN backends ship a wall of NYI stubs
that adds no value. Default-throw lets each backend override only
what it has, with the throw bubbling up immediately and tagged
with the op name (`ComputeOps::sgemm_fp16 not implemented by this
backend`) so misconfiguration is loud.

### Accelerator-only ops with `supports_*()` predicates

A handful of ops exist ONLY on accelerator backends — Q4_0 batch
GEMM, INT4 batch GEMV/GEMM, etc. CPU has no equivalent because
the speedup comes from issuing many parallel kernel launches.

For these, ComputeOps pairs each op with a `supports_*()`
predicate that defaults to `false`. Call sites in
`float_tensor.cpp` use the pattern:

```cpp
if (o->supports_gemm_q4_0_batch_fp32() && M > 1) {
  o->gemm_q4_0_batch_fp32(...);
} else {
  // fall back to a loop of single sgemm calls
  for (i ...) o->gemm_q4_0_fp32(...);
}
```

This keeps the dispatch correct on every backend without
preprocessor branches.

---

## 5. Backend integration granularity

Different backends naturally integrate at different granularities.
This is not a flaw — it reflects the nature of each accelerator.

### Op-level integration (CPU, OpenCL)

Each kernel maps to one ComputeOps method override. The Tensor
class's binary ops (`dot`, `multiply`, `add`, ...) call directly
into `getOps()->X(...)` and the kernel runs synchronously (or
async with a flush).

This works when the backend exposes a flat list of pre-compiled
kernels and the orchestration is "submit, wait, return".

* `CpuComputeOps` (NEON/AVX2 kernels)
* `ClComputeOps` (OpenCL kernels, gemm_q4_0_async_cl etc.)

### Subgraph-compile integration (QNN)

QNN-style backends (Qualcomm AI Engine Direct, similar story for
Apple Core ML or Vulkan compute graphs) work by capturing a
subgraph, compiling it, and submitting the whole compiled blob to
the NPU. Per-op dispatch through ComputeOps would defeat the
graph-fusion optimisations the backend exists to provide.

For these, the integration sits at the LAYER level rather than
the ComputeOps level:
* `QNNContext` registers QNN-specific layer factories
  (`QNNGraph`, `QNNLinear`).
* Those layers' `forwarding()` methods are stubs by design — the
  real execution happens once `QNNGraph::compile()` flushes the
  captured subgraph.
* QNNContext's `ContextData` still binds the CPU fallback
  ComputeOps so any tensor operation that escapes the QNN graph
  (e.g. user code holding a tensor across `forwarding()` calls)
  runs on CPU instead of throwing.

The single `ComputeOps` interface is therefore not a one-size-
fits-all replacement for all vendor integrations — it is the
common dispatch layer for op-level backends, and for graph-compile
backends, it is the CPU fallback while the layer-level integration
does the real work.

---

## 6. Thread safety of init

`ensureComputeOps()` is the canonical entry point for `g_compute_ops`
initialization. It is `std::call_once`-guarded:

```cpp
void ensureComputeOps() {
  std::call_once(g_compute_ops_init_flag, []() { init_backend(); });
}
```

`call_once` provides both mutual exclusion (no two threads can
race `init_backend()`'s non-idempotent `__ggml_init` and
`__openblas_set_num_threads` calls) and acquire/release
synchronisation (any thread returning from `call_once` observes
the `g_compute_ops` write made by the initialiser).

The inline fast-path `getComputeOps()` in `compute_ops.h` keeps
the racy `if (g_compute_ops == nullptr)` check for hot-path
performance. Even if a thread reads a stale `nullptr`, it falls
into `ensureComputeOps()` → `call_once` → the read after returning
is synchronised through the once_flag, so the final return picks
up the latest value.

`AppContext::initialize`, `Engine::add_default_object`,
`ClContext::initialize`, and `QNNContext::initialize` all route
through `ensureComputeOps()` rather than calling `init_backend()`
directly, so the call_once funnel cannot be bypassed.

---

## 7. Adding a new vendor backend (checklist)

Suppose you are adding `XyzBackend` (some new accelerator).

1. **Headers / kernels.** Place the vendor SDK and your kernel
   wrappers under `nntrainer/tensor/xyz_operations/` (or
   `nntrainer/xyz/`) and gate the meson `subdir()` behind an
   `enable-xyz` option so default builds are unaffected.

2. **ContextData subclass** (only if the backend needs per-context
   state). Note `getType()` returns `const char *`, not
   `std::string` (see `context_data.h`):
   ```cpp
   class XyzBackendVar : public ContextData {
   public:
     const char *getType() const override { return "xyz"; }
     XyzSession *session = nullptr;
     XyzKernelCache cache;
   };
   ```

3. **MemAllocator subclass** (only if the backend needs
   device-visible memory — skip for plain host memory). Mirror
   `ClSVMAllocator` (`cl_svm_allocator.h`, OpenCL SVM) or
   `QNNRpcManager` (`qnn/jni/qnn_rpc_manager.h`, DSP RPC):
   ```cpp
   class XyzAllocator : public MemAllocator {
     void alloc(void **p, size_t n, size_t a) override; // device alloc
     void free(void *p) override;
     std::string getName() override { return "xyz"; }
   };
   ```
   Attach it via `getContextData()->setMemAllocator(...)` in the
   Context's `initialize()`. `Engine` copies it into its
   `allocator` map at registration so the memory pool uses the
   right allocator per vendor.

4. **ComputeOps subclass.** Override the ops you support; leave
   the rest to default-throw. For accelerator-only batched ops,
   override the `supports_*()` predicate to return `true`.
   ```cpp
   class XyzComputeOps : public ComputeOps {
   public:
     XyzComputeOps(XyzSession *s) : session_(s) {}
     void sgemm_fp32(...) override { /* uses session_ */ }
     bool supports_gemm_q4_0_batch_fp32() const override {
       return true;
     }
     void gemm_q4_0_batch_fp32(...) override { /* ... */ }
   private:
     XyzSession *session_;
   };
   ```

5. **Context subclass.** Mirror `ClContext` / `QNNContext`. Pass
   the `XyzBackendVar` to the `Context` base ctor. Its
   `initialize()` should:
   * call `ensureComputeOps()` (CPU fallback for unsupported ops)
   * construct an `XyzComputeOps` and `setComputeOps(...)` it on
     this context's `ContextData`
   * `setMemAllocator(...)` the `XyzAllocator` (step 3)
   * register any vendor-specific Layer factories
   * `getName()` must return a non-empty `"xyz"` — the empty
     string is rejected at registration.

6. **Engine wiring.** Two options:
   * **built-in:** under `#if ENABLE_XYZ == 1`, add
     `registerContext("xyz", &XyzContext::Global())` to
     `Engine::add_default_object` (this is how CPU/GPU are wired).
   * **plugin `.so`** (decouples a proprietary SDK, like QNN):
     export the C entry point, build `libxyz_context.so`, and let
     `Engine::add_default_object` load it by path:
     ```cpp
     // xyz_context.cpp, compiled with -DPLUGGABLE
     nntrainer::Context *create_xyz_context() {
       auto *c = new nntrainer::XyzContext(); c->Global(); return c;
     }
     void destory_xyz_context(nntrainer::Context *c) { delete c; }
     extern "C" {
     nntrainer::ContextPluggable ml_train_context_pluggable{
       create_xyz_context, destory_xyz_context};
     }
     ```
     ```cpp
     // engine.cpp, under #if ENABLE_XYZ == 1
     registerContext("libxyz_context.so", "");
     ```

7. **Tests.** Add a unit test that:
   * constructs a `MockXyzComputeOps` (or an
     instrumented real one) attached via ContextData
   * exercises the dispatch path end-to-end
   * verifies the op was actually called

For an op-level backend, that's it. For a graph-compile backend
(QNN-style), most of the work is in custom Layer classes; the
ComputeOps subclass can be just the CPU fallback.
