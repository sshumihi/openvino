# XPU Shared‑Buffer Architecture: One‑Copy Native‑INT4 Weights + KV Cache for GPU‑Prefill + NPU‑Decode

Scope: the `intel_xpu` meta‑plugin that fuses the Intel **GPU** and **NPU** into a single `compile_model`,
running an LLM as **GPU prefill (seq_len > 1) → NPU decode (seq_len == 1)** while both engines read **one
physical copy** of the INT4 weights and share the KV cache. Target platform: LNL / PTL (Core Ultra,
"AI Boost" NPU + iGPU, unified memory), **GPU built on the Level‑Zero backend** (`GPU_RT_TYPE=L0`).

**Default = native INT4.** The weights stay 4‑bit end‑to‑end: GPU prefill dequantizes them in‑kernel and NPU
decode runs them as on‑device INT4 MACs — no host unpack, no f16 expansion, one shared buffer. This is the
target path and what the rest of this document describes. A legacy **DCOFF host‑unpack** path (weights stay u4
but the NPU executes f16 on the host) exists only as a working fallback / KV‑cache concept proof and is
fully isolated in **Appendix A**; it does not meet the perf/memory target.

Validation (Qwen3‑4B, i4‑sym per‑channel, prompt "The capital of France is", **no env flags**):

| | Result |
|---|---|
| Output | CPU‑exact — " Paris. The capital of the United Kingdom is London." |
| Weight copies in memory | **1** (~2.1 GB i4, shared GPU+NPU) |
| Decode | **~36 ms/tok** (native, steady @1K; == stock standalone NPUW 38.8 ms/tok → shared buffer has no decode penalty; vs 168 with DCOFF). §7.1 |
| Prefill @1K (first token) | **~230 ms** end‑to‑end (≈ GPU compute; ~1.4× stock GPU PA 163 ms) after 3 opts — last‑token logits slice, NPU‑request reuse, KV bind‑cache (§7.1). vs ~700 ms before, ~1333 ms CPU bridge |
| Process committed | ~6.1 GB |
| GPU `usm_device` | ~170 MB — the KV‑write graph's GPU buffers (§3.3); **the shared KV buffer itself is HOST‑imported, not `usm_device`** (verified, see §7). No weight/KV *copy*. (≈ 1.5 MB with the legacy CPU bridge.) |

**Build:** `GPU_RT_TYPE=L0`. **Model export:** `optimum-cli export openvino --weight-format int4 --sym
--group-size -1 --ratio 1.0`. **No env flags needed** — the plugin defaults to native + malloc + skip‑prefill
and sets `OV_XPU_REF_COMPRESSED_FC` itself. (Opt‑outs: `XPU_DCOFF=1`, `XPU_USM_WEIGHTS=1`,
`XPU_KEEP_NPU_PREFILL=1`; tracing `XPU_MEM_DEBUG=1`.)

---

## 1. Low‑level buffer‑sharing mechanism

The hard problem: the **GPU plugin runs on Level‑Zero** (`GPU_RT_TYPE=L0`) and the **NPU plugin runs on
Level‑Zero** — but as **separate L0 contexts/drivers**. A buffer one context *allocates* is not *importable*
by the other. Empirically (POC `C:\yqiu\poc-npu-gpu-buffer-sharing`, reproduced on this box):

| Allocation | Cross‑context importable? |
|---|---|
| context‑owned `zeMemAllocHost` (GPU ctx) | ❌ "part of an existing allocation" |
| OCL `clHostMemAllocINTEL` (OCL GPU build) | ❌ (NPU‑L0 cannot import it) |
| **page‑aligned `_aligned_malloc`, imported into each L0 ctx** | ✅ **same pointer** — the shared‑weight design |

### 1.1 The shared buffer: malloc + dual L0 import (`shared_weight_buffer.hpp`)
`SharedWeightBuffer` (RAII) owns one host buffer per weight and hands out `host_ptr`. Default allocation is a
neutral `_aligned_malloc(size, 4096)`, **imported independently into each engine's L0 context** — the import
returns the *same* pointer, so both engines address the identical physical pages:
```cpp
// SharedWeightBuffer::import_into_gpu_l0(gpu_ctx)   — registers the malloc in the GPU's L0 context
ze_ctx = (ze_context_handle_t) gpu_ctx->get_property()["OCL_CONTEXT"];   // == ze ctx on the L0 GPU backend
ze_external_memmap_sysmem_ext_desc_t sys{ ...SYSMEM..., .pSystemMemory = host_ptr, .size = alloc_size };
ze_host_mem_alloc_desc_t  h{ ...HOST_MEM..., .pNext = &sys };
zeMemAllocHost(ze_ctx, &h, alloc_size, 4096, &imported);   // imported == host_ptr
```
- **GPU‑L0 import** (above, at compile) makes the malloc a registered L0 allocation so `share_usm` /
  `zeMemGetAddressRange` works on it (§6.2).
- **NPU‑L0 import** happens when the NPU binds the weight closure as an L0 input at inference (§5.3) — the same
  `zeMemAllocHost` system‑memory import, succeeding because it is a neutral malloc (not a context‑owned alloc).

The XPU plugin links `ze_loader`; `ze_loader.dll` ships in the Release dir. Cleanup `zeMemFree`s the import
handle before `_aligned_free`. (System‑memory import requires LNL/PTL drivers; MTL's older NPU driver rejects
it.)

### 1.2 Per‑weight segmentation
The weight store is **not** one buffer — each model `Constant` ≥ 4096 B gets its own `SharedWeightBuffer`. This
lets the consolidation point each closure at exactly one segment, and keeps every allocation < 2 GB (a 2³¹
boundary in the GPU `share_usm` path makes a single > 2 GB host allocation unaddressable; per‑weight buffers
are all individually addressable). The compiled model owns `std::vector<SharedWeightBuffer> m_weight_segments`;
the address ranges are published to both engines as a serialized `"ptr:size;ptr:size;…"` list.

### 1.3 Why "one physical copy" holds
The same `host_ptr` is: written once (memcpy from the mmap'd `.bin`), imported into the GPU‑L0 context (GPU
reads it via `share_usm`), and imported into the NPU‑L0 context (NPU reads it natively). No engine keeps a
private device copy of the weights — **proved three independent ways in §10** (memory accounting, a host‑poke
mutation that changes both engines' output, and a GPU‑L0 audit showing every weight maps `type=HOST`,
`base==host_ptr`, zero `DEVICE` allocations). Note the imported malloc is *not* counted as GPU‑allocated
`usm_host` (it's external), so the GPU stats show only its own scratch — see §10.1.

---

## 2. Weights: native INT4, i4‑symmetric per‑channel

NPUW turns a submodel's weight `Constant`s into **closures** — graph `Parameter`s fed at runtime from a weights
bank (§5.1). A closure carries a `LazyTensor` whose `const_source()` returns `{ptr, bytes}` for a plain
un‑transformed Const — that pointer is the per‑weight buffer, so the closure is **byte‑identical** to the
shared buffer and zero‑copy‑shareable.

The export must be **i4‑symmetric, group‑size −1 (per‑channel)**:
- **Symmetric** so there are no zero‑points — the compiler‑dynamic‑quant INT4 matmul does not accept asym zp.
- **Per‑channel (−1)** so the partition uses the **`DQMatMulCWi`** pattern (`partitioning/patterns/opt.cpp`),
  which **does not permute** the weight (it only flips the matmul's `transpose_b`). The weight stays in its
  `.bin` row‑major layout = byte‑identical to the shared buffer. (Group‑size 128 would use `DQMatMulGQi`,
  which `permute({0,2,1})`s the weight into an NPU‑specific layout → not shareable.)

NPUW runs its baseline **compiler‑dynamic‑quant** path: `NPUW_DQ=YES` + `NPU_COMPILER_DYNAMIC_QUANTIZATION=YES`,
no DCOFF. The VCL compiler emits an on‑device INT4 matmul (weights stay i4, activations dynamically quantized
to int8). The XPU plugin leaves DCOFF off by default; `XPU_DCOFF=1` opts into the legacy path (Appendix A).

### 2.1 Exact layout in the shared buffer
Each segment holds one `Constant`'s bytes **verbatim from the `.bin`** (no repack). For Qwen3‑4B (hidden 2560,
36 layers, 32 Q / 8 KV heads, head_dim 128, intermediate 9728, vocab 151936) — confirmed from the live
consolidation / `XPU-ZC` dumps:

| Tensor (per layer unless noted) | OV type / shape | bytes | quant |
|---|---|---|---|
| `q_proj.weight` | **`i4 [4096, 2560]`** | 5.2 MB | sym i4, packed 2/byte, row‑major `[out, in]` |
| `k/v_proj.weight` | `i4 [1024, 2560]` | 1.3 MB ea | sym i4 |
| `o_proj.weight` | `i4 [2560, 4096]` | 5.2 MB | sym i4 |
| `gate/up_proj.weight` | `i4 [9728, 2560]` | 12.5 MB ea | sym i4 |
| `down_proj.weight` | `i4 [2560, 9728]` | 12.5 MB | sym i4 |
| every `*_proj.weight/scale` | **`f16 [out, 1]`** | out·2 B | **per‑output‑channel** scale, **no zero‑point** |
| `input/post_attn_layernorm` | `f32 [1, 1, 2560]` | 10 KB | RMSNorm weights |
| `q/k_norm`, rotary cos/sin | `f16 [128]` / `f16 [1, 1152, 128]` | small | — |
| `embed_tokens.weight` (1×, tied lm_head) | **`u8 [151936, 2560]`** | 389 MB | int8‑asym: + `u8 [151936,1]` zp + `f16 [151936,1]` scale |

So the bulk transformer weights are **plain row‑major `i4 [out, in]`** with one f16 scale per output channel and
**no zero‑point**.

### 2.2 How each engine consumes those bytes at inference
Both engines read the **same `i4 [out, in]` + `f16 [out,1]` scale** segments; they differ only in *where* the
dequant/matmul happens:

- **GPU prefill (oneDNN INT8×INT4, §6.1)** reads the i4 weight as row‑major `oiyx` (o=out, i=in — no reorder),
  **dynamically quantizes the f16 activation to int8**, runs the int8·int4 matmul on the XMX/DPAS systolic
  array, and applies the per‑output‑channel `f16 scale` — symmetric, **no `−zp` term**. The embedding is
  consumed by a `Gather` over the `u8` rows with `(u8 − zp)·scale`. All weight reads are zero‑copy from the
  shared buffer (L0 `share_usm` on the imported malloc, §6.2). Output is f16/f32 accumulation → logits + KV.
- **NPU decode (native INT4)** keeps the weight **4‑bit end‑to‑end**: the i4 closure is a view of the shared
  malloc (§5.3), bound as an L0 input; the NPU L0 backend imports that malloc and the VCL‑emitted kernel
  streams the 4‑bit weights into SRAM, **dynamically quantizes the activation to int8**, runs an int8·int4 MAC,
  and applies the per‑channel `f16 scale` at the output. No host unpack, no f16 weight expansion.

One layout, one copy, two native consumers.

---

## 3. KV cache sharing (GPU prefill ↔ NPU decode)

The KV cache is the original cross‑engine zero‑copy surface (the concept the DCOFF path first proved).

### 3.1 Layout
One contiguous `kvcache` buffer of per‑layer K/V slots ordered by **alphabetical port name**, f16:
```
past_key_values.0.key    [1, H_kv, max_kv, head_dim]
past_key_values.0.value  [1, H_kv, head_dim, max_kv]   ← TRANSPOSED if v_tensors_transposed_gen
…  prefix‑sum offsets, emitted by NPUW_KVCACHE_LAYOUT and by create_generate_request_variants (must agree)
```

### 3.2 NPU side
`create_generate_request_variants` (`llm_infer_request.cpp`) binds each generate‑model `past_key_values.*`
input to `kvbuf + offset` via `make_tensor` + `set_tensor`, `memset`s the region, and aliases smaller KV‑size
variants onto the largest. So NPU decode reads/writes its KV **in place** in the shared buffer.

### 3.3 GPU → NPU handoff: GPU KV‑write (default)
The GPU runs a **stateless** clone (`ov::pass::StatefulToStateless`) exposing `present.N.{key,value}`. **The GPU
itself produces and writes the KV cache in the NPU's exact format, zero‑copy** — no host round‑trip:

- **Graph rewrite** (`xpu_rewrite_present_to_static_slots`, `plugin.cpp` step 6f): after StatefulToStateless,
  each `present.N.{key,value}` is rewired to `Convert(f16)` → *(value)* `Transpose([0,1,3,2])` →
  `ScatterUpdate(Broadcast(0, slot_shape), Range(0, seq), upd, axis)`, where `slot_shape` / `axis` / the
  transpose come from `NPUW_KVCACHE_LAYOUT` + `NPUW_KVCACHE_V_TRANSPOSED_GEN`. ScatterUpdate places the
  dynamic‑seq K/V into rows `[0, prompt_len)` of a **static** `[1, H_kv, max_kv, head_dim]` slot — the *static*
  output shape (= the Broadcast/data shape) is what makes it zero‑copy‑bindable.
- **Zero‑copy bind**: each `present.*` output is bound to
  `gpu_ctx->create_tensor(f16, slot, {SHARED_MEM_TYPE=USM_USER_BUFFER, MEM_HANDLE=kvbuf+offset})` and
  `set_tensor`. `intel_gpu`'s `prepare_output` binds a static, no‑convert `RemoteTensorImpl` as the network's
  output memory, so the GPU writes the slot **in place** in the shared buffer (no copy). These remote tensors are
  created **once at compile** (`build_kv_output_bindings`) and `set_tensor`'d once at request init — not rebuilt
  per prefill (§7.1.1; the outputs are static so the binding is invariant).
- **Ordering / NPU adoption**: the NPU request is created **once** (at `XpuSyncInferRequest` construction) and
  reused. Each prefill just *signals* the prompt length via the `XPU_EXTERNAL_PREFILL_LEN` compiled‑model
  property; the NPUW request consumes it at the top of its **next** `infer()` (the first decode) —
  `init_from_external_prefill` re‑selects the generate variant + sets `num_stored_tokens=prompt_len`, adopting
  the GPU's KV without re‑running prefill. No pre‑infer `memset` is needed: the GPU's `ScatterUpdate(Broadcast(0,
  …))` rewrites the full `max_kv` slot during the (blocking) prefill `infer()`, so unused rows are zeroed and any
  stale KV from a prior conversation is overwritten before NPU decode reads it (§7.1).

This moves the per‑element f32→f16 + transpose + scatter off the CPU onto GPU stock ops. Combined with the other
prefill optimizations (§7.1), first‑token went **1333 ms (CPU bridge) → ~700 (KV‑write) → ~420 (NPU‑request
reuse) → ~230 ms (logits slice)** at 1K — now ≈ GPU compute. **Decode unchanged**, **token‑exact vs CPU**.
*Why a separate pass, not fused into SDPA:*
the fast **micro** SDPA kernel hides K/V inside an opaque compiler‑generated `ugemm` micro‑GEMM (no editable
K/V loads); fusing would force the slower `sdpa_opt` kernel for a ~1 ms saving. The stock‑ops pass keeps micro
and still does the GPU cast+layout+direct write. (Cost: ScatterUpdate writes the full `max_kv` slot each
prefill → a ~170 MB *transient* GPU intermediate, freed after prefill; KV buffer stays host‑USM, `usm_device`≈0.)

---

## 4. XPU plugin design + GPU/NPU plugin interaction

### 4.1 Compile pipeline (`Plugin::compile_model(filesystem::path)`)
The string‑path entry is the hybrid pipeline (Python `core.compile_model("...xml","XPU")` reaches it via
`to_fs_path → CoreImpl::compile_model(path)`):
```
 1  set OV_XPU_REF_COMPRESSED_FC=1 if unset          // GPU reads i4 with no reorder (§6.1)
 2  model = read_model(xml, bin, {})                  // .bin MMAP'd (lazy pages)
 3  for each Constant ≥4096B:                          // PER-WEIGHT relocation
        buf = create_empty(gpu_ctx, bytes, zero=false) //   malloc + import_into_gpu_l0 (default)
        memcpy(buf, c.data); replace_node(c, Constant(type,shape,buf->host_ptr, so=buf))
        ranges += "{ptr}:{size};"
 5  if (all segments GPU-L0-imported) model.rt_info[xpu_shared_weight_ranges] = ranges  // gpu_shares
 6  npu = compile("NPU", { NPU_USE_NPUW, NPUW_LLM,
                           (default: native — NO DCOFF; XPU_DCOFF? {NPUW_DCOFF_TYPE=f16,…}),
                           XPU_SKIP_NPU_PREFILL (default in hybrid),  XPU_RAW_WEIGHT_RANGES=ranges,
                           NPUW_DEVICES=NPU, NPU_COMPILER_TYPE=PLUGIN })          // VCL 8.1.0
 6b persistent = USM_host( NPUW_WEIGHTS_TOTAL_BYTES(non-raw) + NPUW_HOST_CLOSURE_TOTAL_BYTES )
 6d npu.set_property(XPU_SHARED_WEIGHT_PTR = persistent)  // consolidate: raw-share + copy residue + refresh
 6e kvcache = create_empty(NPUW_KVCACHE_TOTAL_BYTES);  npu.set_property(XPU_SHARED_KVCACHE_PTR = kvcache)
 6f m = StatefulToStateless(model.clone())            // present.*/past_key_values.* ports
        rewrite_present_to_static_slots(m, NPUW_KVCACHE_LAYOUT)  // present.* → static NPUW KV slots (§3.3, default)
        gpu_prefill = compile("GPU", m)
 7  gpu_full = SKIP (hybrid)                            // §4.3 — full stateful GPU model not built
 8  XpuCompiledModel{ npu, gpu_prefill, weight_segments[], persistent, kvcache }
```
NPU is compiled first (stricter consumer; its bank evaluation fixes the exact persistent size).

### 4.2 Property plumbing (how the three plugins agree on the buffers)
The XPU plugin is the orchestrator; the GPU and NPU plugins never talk to each other:
- **GPU** learns the ranges via the model's **rt_info** `xpu_shared_weight_ranges`; `ProgramBuilder` parses it
  into `m_shared_weight_ranges`; `is_shared_weight_ptr` does a containment test and routes to `share_usm`.
  Set **only when `gpu_shares`** is true (every segment was L0‑imported); otherwise the GPU copies.
- **NPU** learns the ranges via the **property** `XPU_RAW_WEIGHT_RANGES` (string) → Bank `m_xpu_raw_ranges`
  (used for zero‑copy views at *compile*, §5.2); persistent via `XPU_SHARED_WEIGHT_PTR`; KV via
  `XPU_SHARED_KVCACHE_PTR`; `XPU_SKIP_NPU_PREFILL` to alias the prefill model (§4.3).
- `XpuCompiledModel` forwards any `NPUW*` property to the NPU model and exposes `XPU_WEIGHT_SEGMENT_COUNT/BYTES`,
  `XPU_SHARED_WEIGHT_SIZE`, `XPU_SHARED_KVCACHE_SIZE`. It holds every buffer (RAII) for the model lifetime.

### 4.3 Skipping unused models (default in hybrid)
The GPU only runs the **stateless prefill** model, so: (a) the full stateful GPU model (step 7) is **not built**
(`gpu_compiled` null; `meta_compiled()` falls back to gpu_prefill→npu for export/runtime‑model/property;
`XPU_COMPILE_GPU_FULL=1` re‑enables it); (b) the **NPU prefill model** is the GPU's job, so NPUW aliases
`m_prefill_compiled = m_kvcache_compiled` instead of compiling a second model (saves ~2.2 GB, §7). Both are on
by default; the pure‑NPUW path (`XPU_ACTIVE_DEVICE=NPU` at compile, or `XPU_KEEP_NPU_PREFILL=1`) keeps the NPU
prefill model since it then runs prefill on the NPU.

### 4.4 Inference orchestration (`XpuSyncInferRequest::infer`)
```
if active_device == "NPU"                     → infer_npu_decode()   // pure NPUW (prefill+decode on NPU)
else if gpu_prefill && (seq_len>1 || !done)   → infer_gpu_prefill()  // (re)prefill: new conversation / chat turn / first infer
else if prefill_done                          → infer_npu_decode()   // NPU decode per token (seq_len==1, after a prefill)
else                                          → infer_passthrough()  // GPU-only (asserts if gpu_full skipped)
```
A `seq_len>1` infer is **always** a (re)prefill — a new conversation or chat turn — so it routes to GPU prefill
even after a prior prefill on the same request. There is **no `seq_len>1 && !prefill_done` latch** on the prefill
branch (that earlier latch sent a 2nd conversation's prefill into NPU decode and crashed, §7.1); `prefill_done`
is used only on the *decode* branch to route post-prefill single-token infers. The `|| !done` also routes a
degenerate 1-token first prompt through prefill instead of the GPU-only passthrough.

---

## 5. NPUW weightless blob + memory flow (compile & inference)

### 5.1 The blob is weightless — weights are graph inputs, not baked in
NPUW partitions the model and, for repeated transformer blocks, compiles the block **once** (funcall/REP
reuse) and runs it 36× with different weights. In `partitioning.cpp:1787‑1802` each bulk weight `Constant` is
replaced by a new `Parameter` and the original captured as a `LazyTensor` closure; only tiny scalar consts stay
inline. Serialization to the VCL compiler uses **`NO_WEIGHTS_COPY`** (`model_serializer.cpp`) and the
weightless `compileWsOneShot` flow (`compiler_impl.cpp`), so the **main L0 blob contains no weights** — at
inference they are read from the shared buffer. Verified: the VCL compile step adds only **+13 MB** committed.

### 5.2 Compile‑time weight flow (zero‑copy bank)
The weights bank used to copy each weight into an NPU device tensor (`evaluate_and_allocate_on_device`:
`create_host_tensor + copy_to`). Now: when a closure's `const_source()` lies in `m_xpu_raw_ranges` (passed at
*compile* time via `XPU_RAW_WEIGHT_RANGES`, registered on the bank **before** eval at `llm_compiled_model.cpp`),
the bank stores a zero‑copy `ov::Tensor(meta, ptr)` view of the shared buffer and skips the alloc+copy (508
views / 146 small materialized). This removes the NPU device weight copy and — because the NPU now reads
weights from the shared buffer rather than a per‑token device copy — **sped hybrid decode 150 → ~36 ms/tok**
(steady @1K; matches standalone native‑INT4 NPUW, so the shared buffer adds no decode cost).

### 5.3 Inference‑time weight flow
The i4 closure (a view of the shared malloc) is set as the NPU L0 input; the NPU L0 backend **imports** the
malloc (`zeMemAllocHost` system‑memory import, §1.1) and the on‑device INT4 kernel reads 4‑bit weights into
SRAM during compute — no host unpack, no f16 expansion. (Block‑reuse keeps one block's weights resident at a
time; the embedding/lm_head submodel reads the u8 embedding from the shared buffer.)

---

## 6. GPU compute‑graph adaptation to NPU‑managed formats + GPU memory flow

The GPU must consume the **same physical i4 bytes** NPUW keeps and produce KV in the NPU's expected layout.

### 6.1 Weight format — no reorder, read in place (FCs run on **oneDNN**)
The constraint is that the GPU must read the **same row‑major `i4 [out, in]`** NPUW keeps, with no reorder (a
reorder would copy the weight out of the shared buffer). `OV_XPU_REF_COMPRESSED_FC=1` (the plugin sets this,
§4.1) declines compressed‑INT4 in the OCL `bf_tiled`/`bf_tiled_dyn_b`/`gemv` kernels (which request a *blocked*
`os_is_yx_osv32_isv2` layout), and `constant.cpp` keeps the weight in `oiyx` + `share_usm` (§6.2) → the weight
pointer stays in the shared buffer, zero‑copy.

**What actually executes the FCs is oneDNN, not the OCL kernel selector** (verified from the GPU runtime graph:
181 `FullyConnected` nodes, `primitiveType=undef`, `runtimePrecision=i8`, paired with `DynamicQuantize`). On
this XMX/DPAS iGPU the GPU dynamic‑quantizes the f16 activation → int8 and runs an **INT8×INT4 GEMM on the
systolic array**, reading the shared `oiyx` i4 weights directly (no device copy; `usm_device`≈0). oneDNN owns
the FC `impl_type` *before* the OCL kernel selector is consulted — so the OCL `bfyx_ref` fallback never runs,
and a custom OCL tiled‑`oiyx` kernel (prototyped, then removed) is never selected and cannot beat the XMX path.
The "slow prefill" people notice is the **one‑time first‑inference kernel/oneDNN JIT** (~600 ms, amortized at
the compiled‑model level), not the per‑token FC compute (~1.5 ms/tok steady).

### 6.2 GPU zero‑copy bind on the L0 backend
`constant.cpp` `is_shared_weight_ptr(data) → engine::share_usm(layout, ptr)` for weights in the shared ranges.
On the **L0 GPU backend** (`ze_engine.cpp:126` `reinterpret_handle`), this wraps `ptr` via
`zeMemGetAddressRange` → `ze::gpu_usm`. It works on the **imported malloc** (§1.1) — the malloc the NPU also
imports is GPU‑zero‑copy too. This is the configuration where **both** engines read one buffer with no copy.
(On an OCL GPU build, `share_usm` requires a real USM allocation and would reject a malloc → the GPU copies;
that is why the L0 backend is the build requirement for the one‑copy property.)

### 6.3 GPU memory flow (compile & inference)
- **Compile**: the prefill model's FC weights are bound zero‑copy. The 2.1 GB weights are the **imported
  malloc** — *external*, so they do **not** count as GPU‑allocated `usm_host`; the GPU's own stats show only
  its scratch (~0.3–0.6 GB of activations/folded consts, plus the KV‑write graph buffers when on, §7). No
  weight copy (a copy would add +2.1 GB of GPU `usm_device` — measured ~0 for weights; see §10.1).
- **Inference**: prefill reads weights in place and writes KV into the shared `kvcache` (via the bound
  `present.*` outputs, §6.4). The **shared KV buffer is a HOST‑imported malloc** (`type=HOST`, `base==ptr` in
  the GPU L0 ctx — same dual‑import path as the weights, so the NPU imports it too), **not** `usm_device`. The
  ~170 MB `usm_device` seen with KV‑write is the GPU graph's own buffers for the static `present.*` outputs +
  `ScatterUpdate` zeros (only present when KV‑write is on; ≈ 1.5 MB with the legacy bridge) — it is GPU‑side
  compute memory, not a copy of the KV cache or weights.

### 6.4 KV output format — GPU produces the NPU's static slot
The other NPU‑managed format the GPU must match is the **KV cache** (static `[1, H_kv, max_kv, head_dim]` f16,
value‑transposed). Rather than reformat on the host, the prefill model is rewritten (§3.3) so the GPU emits the
KV already in that layout (Convert + Transpose + ScatterUpdate into a static slot) and the slot is bound
zero‑copy to the shared buffer. So the GPU adapts to **both** NPU‑managed formats — i4 weights (read in place,
§6.1) and the f16 KV slot (written in place) — with no host transcoding on either side.

### 6.5 What "the L0 backend" means — same OpenCL‑C kernels, Level‑Zero runtime
`GPU_RT_TYPE=L0` selects the GPU plugin's **runtime** (memory, queue, kernel compile + submit) — **not** the
kernel language. The compute kernels are the *same* OpenCL‑C kernels in `cl_kernels/` (plus the micro‑GEMM) on
both backends; only *how* they are built and launched changes. So a VTune trace under L0 showing OCL‑named
kernels (`sdpa_micro_prefill`, `dynamic_quantize_gpu_opt`, …) is expected — not a contradiction.

- **Build** (`runtime/ze/ze_kernel_builder.cpp`): an OpenCL‑C kernel becomes an L0 module via `zeModuleCreate`
  — either compiling the OCL‑C source directly (`ze_module_format_oclc`, when the driver supports it, probed by
  `check_l0_build_support`) or, as a fallback, compiling to a native binary through the OCL builder and loading
  it (`ZE_MODULE_FORMAT_NATIVE`). The `sdpa_micro` `ugemm` micro‑GEMM (`micro::Package`) is a native vISA
  microkernel linked into that module.
- **Launch** (`runtime/ze/ze_stream.cpp`): `zeCommandListAppendLaunchKernel` (Level‑Zero command lists), not
  `clEnqueueNDRangeKernel`.
- **Memory** (`runtime/ze/ze_memory.cpp`): USM via `zeMemAlloc*`. Crucially, the L0 `ze_context` (exposed as the
  GPU `RemoteContext`'s `OCL_CONTEXT` property) is what the XPU plugin imports the shared malloc into (§1.1) and
  what `share_usm` wraps (§6.2) — the OCL backend can't import a neutral malloc, so **L0 is the requirement for
  the one‑copy property**.

Those two kernels are the GPU **prefill** hot ops and corroborate the design: `sdpa_micro_prefill` = the micro
(XMX/DPAS) attention kernel (the reason KV‑write is a separate pass, not fused into SDPA, §3.3);
`dynamic_quantize_gpu_opt` = the F16→INT8 activation quantization feeding the INT8×INT4 oneDNN FC (§6.1). NPU
decode runs on the VCL blob and does not appear in the GPU/L0 trace.

---

## 7. Performance & memory

### 7.1 Head‑to‑head vs stock OpenVINO GenAI (pure GPU / pure NPU)

Same model (`Qwen3‑4B‑int4‑sym‑cw`), same 1024‑token prompt, 32 output tokens, OV 2026.3 nightly GenAI
(`llm_bench`, avg of 3, warm‑up excluded). The hybrid row is `bench_npuw_vs_xpu.py xpu 1024 32` on this build:

| Config | Time‑to‑first‑token (prefill) | Decode (2nd‑token) | 2nd‑token throughput |
|---|---|---|---|
| Stock GenAI **pure GPU** (PagedAttention) | **163 ms** | **23.9 ms/tok** | 41.8 tok/s |
| Stock GenAI **pure NPU** (NPUW) | **710 ms** | **38.8 ms/tok** | 25.8 tok/s |
| **XPU hybrid** (GPU prefill → NPU decode) | **~230 ms** | **~36 ms/tok** | ~28 tok/s |

**Decode — NPU parity, no shared‑buffer penalty.** Hybrid decode (~36 ms/tok) matches stock standalone NPUW
(38.8) to within noise: routing decode through the NPU while it reads the *shared* INT4 buffer costs nothing
vs a private‑weight NPUW model. (Stock pure‑GPU decode is faster at 23.9 ms/tok, but occupies the GPU; the
hybrid decodes on the NPU by design — to free the GPU and keep one weight copy.)

**Prefill — three optimizations took first‑token ~700 → ~230 ms** (now ~1.4× the heavily‑tuned stock GPU PA
path at 163 ms, vs ~4.3× before). Instrumented breakdown (`OV_XPU_DEBUG`, `infer_gpu_prefill`, @1K):

| Component (per conversation) | Recreate, full‑seq logits (old) | **Full default stack** |
|---|---|---|
| **GPU prefill compute** (stateless clone + KV‑write graph + LM head) | ~245 ms | **~230 ms** — LM head now runs on **1** position, not `seq` (§7.1.2) |
| NPU generate‑request create + memset (`npu_init`) | ~220 ms | **~0 ms** — request **reused**; a property *signal* replaces it (§7.1, consumed lazily at the next decode infer) |
| KV‑output bind (72× `create_tensor`+`set_tensor`) + input set‑up | ~10 ms | **~0.4 ms** — bindings pre‑created at **compile** (§7.1.1) |
| logits materialization in the async wrapper | ~175 ms (full‑seq `[1,seq,151936]` f32 ≈ **622 MB**) | **~0 ms** — logits sliced to `[1,1,151936]` ≈ **600 KB** (§7.1.2) |
| **End‑to‑end first‑token** | **~700 ms** | **~230 ms** |

The three levers, in order of impact: **(1) logits slice** (§7.1.2, ~200 ms — the big one), **(2) NPU‑request
reuse** (§7.1 below, ~220 ms but it overlapped the others), **(3) KV‑output bind‑cache** (§7.1.1, ~10 ms). The
**GPU prefill compute (~230 ms) is ~3× faster than NPU prefill (710 ms)** — the GPU‑prefill premise holds, and
decode is unchanged throughout.

**(2) NPU‑request reuse.** The NPU generate request was recreated every prefill (~220 ms `create_infer_request`).
Now it's built once (`XpuSyncInferRequest` ctor) and reused; each prefill only **signals** the prompt length via
the `XPU_EXTERNAL_PREFILL_LEN` compiled‑model property, which the NPUW request consumes at the top of its next
`infer()` — re‑selecting the generate variant and resetting `num_stored_tokens`. Stale KV from a prior
conversation is safely overwritten because the KV‑write graph's `ScatterUpdate(Broadcast(0,…))` rewrites the full
`max_kv` slot every prefill (§3.3). This also fixed a latent bug: a 2nd conversation's prefill on a reused
request was misrouted to NPU decode (a `!m_prefill_done` dispatch latch) and crashed — `seq_len>1` now always
(re)prefills, so **multi‑turn / multi‑prompt reuse works** (validated token‑exact: a 2nd prompt on a reused
request == a fresh request; `test_npu_reuse.py`). (Numbers ±10% run‑to‑run.)

#### 7.1.1 KV‑output binding moved to compile (small win, but correct)
Binding the 72 `present.*` outputs to their shared‑buffer slots was redone every prefill (rebuild 72 remote
tensors + `set_tensor`). Because the outputs are **static** (§3.3) the binding is conversation‑invariant, so the
remote tensors are now created **once at compile** (`XpuCompiledModel::build_kv_output_bindings`, the costly
`create_tensor`) and `set_tensor`'d once at request construction; `infer_gpu_prefill` skips the loop
(`m_kv_outputs_bound`). Measured per‑prefill bind+input setup dropped from ~10 ms to **~0.4 ms** — so this was a
*cleanup*, not a big first‑token win: the rebind was never the bottleneck (the buffer is already L0‑imported, so
`create_tensor` is cheap).

#### 7.1.2 Prefill logits sliced to the last token (the big lever, ~200 ms)
The prefill only needs `logits[-1]` — the first generated token is `argmax(logits[-1])`; the other rows are
never read. But the GPU prefill model emitted **full‑sequence** logits `[1, seq, 151936]` f32 = **622 MB at 1K**,
so the OV async wrapper materialized all of it (~175 ms) *and* the LM‑head MatMul ran on all `seq` positions
(part of the GPU compute). **Fix (port of OpenVINO GenAI's `apply_slice_before_matmul`, `utils.cpp`):** insert a
`Slice(start=-1, stop=-2, step=-1, axis=seq)` on the **activation input** of the LM‑head MatMul in the prefill
model, so it runs on exactly the last position — logits become `[1, 1, 151936]` ≈ **600 KB**. Implemented in
`plugin.cpp` (`xpu_slice_prefill_logits_to_last`, after `StatefulToStateless` + the KV‑write rewrite, before the
GPU compile). `find`‑pattern handles `MatMul→Result`,
`MatMul→Add→Result`, `MatMul→Transpose→Result`; guarded to a rank‑3 activation. **Measured: first‑token
~430 → ~230 ms (−46%)** at 1K (the ~175 ms logits copy vanishes; the GPU LM‑head shrinks ~`seq`×), **decode
unchanged, token‑exact** (first token = `argmax(logits[-1])` is identical by construction; CPU‑exact 16/16,
multi‑conversation pass). Runtime logits tensor verified `[1, 1, 151936]` for a 1024‑token prompt. This is the
same trick that keeps stock GPU GenAI prefill at 163 ms (`pipeline_stateful.cpp` applies it for all non‑NPU
devices; NPUW does the equivalent via `cut_lm_head` + `SliceOutEmbeds`).

### 7.2 Memory overhead beyond the shared weight buffers

The shared weights are **one** ~2.1 GB copy. Everything else, measured (committed / private bytes,
`XPU_MEM_DEBUG`):

| Component | Committed | Status |
|---|---|---|
| Python + OV runtime baseline | ~0.7 GB | fixed |
| **Shared weight buffer** (1 copy, GPU+NPU) | 2.1 GB | the model |
| GPU prefill compile (activations + folded consts) | ~0.7 GB | needed (one‑shot prefill) |
| GPU KV‑write graph buffers (static `present.*` outputs + `ScatterUpdate`/`Broadcast` zeros) | ~170 MB `usm_device` + ~340 MB `usm_host` | overhead of the KV‑write rewrite (vs 0 / 287 MB with the CPU bridge); the **shared KV buffer is a separate HOST‑imported malloc** — verified `type=HOST`, `base==ptr` in the GPU L0 ctx |
| NPU persistent buffer (host closures not raw‑shared) | **6 MB** | minimized (was 1792 MB before per‑channel + raw‑share) |
| NPU **prefill** model compile | **0** | **eliminated** (alias to generate) — was +2.2 GB |
| NPU bank device weight copy | **0** | **eliminated** (zero‑copy views) — was on L0 device |
| VCL blob (weightless) + L0 activation scratch | small | weightless |
| **NPUW partition transient** (`matchRepeatedSubgraphs`/`createFunction`) | **~2.1 GB** | **OPEN** — see below |

Total ≈ 6.1 GB committed; decode ~36 ms/tok; first‑token ~230 ms @1K (≈ GPU compute, after logits slice + NPU‑request reuse + bind‑cache, §7.1); output CPU‑exact.

**The remaining ~2.1 GB overhead** is *not* the blob (weightless), *not* the VCL compiler (+13 MB), *not* the
bank (zero‑copy). It is materialized inside `Partitioner::matchRepeatedSubgraphs` / `createFunction`
(`partitioning.cpp:1919/1751`) while building the repeated‑block function body ("CWAI" form) — pinpointed by
per‑pass committed checkpoints (group 1 +379 MB, group 3 +1690 MB). It persists (held by the partition output).
Making `createFunction` reference the shared weights lazily would reclaim it (→ ~4 GB total) but is deep
partitioner surgery — the next memory item. The NPU L0 also holds the weightless blobs + per‑block activation
scratch (device memory, not in committed; bounded by block‑reuse).

---

## 8. Configuration

| Setting | Default | Override |
|---|---|---|
| GPU backend | `GPU_RT_TYPE=L0` (build) | OCL build → GPU copies (no one‑copy) |
| Weight execution | **native INT4** | `XPU_DCOFF=1` → legacy host‑unpack (Appendix A) |
| Weight buffer | **malloc + dual L0 import** | `XPU_USM_WEIGHTS=1` → OCL USM‑host (Appendix A) |
| NPU prefill model | **skipped** (GPU prefills) | `XPU_ACTIVE_DEVICE=NPU` / `XPU_KEEP_NPU_PREFILL=1` |
| GPU FC impl | **oneDNN** INT8×INT4 (reads shared i4 in place; plugin sets `OV_XPU_REF_COMPRESSED_FC` to decline the OCL blocked kernels) | — |
| Debug prints | off | `OV_XPU_DEBUG=1` ([XPU]/[XPU‑ZC]/[NPUW]/[DCOFF]/[DIAG]); `XPU_MEM_DEBUG=1` ([…][MEM] breakdowns) |

The GPU KV‑write (§3.3), NPU‑request reuse, compile‑stage KV‑output binding, and last‑token logits slice (§7.1)
are now **unconditional** (no toggles) — the A/B opt‑out env flags were removed once the paths were validated.

Model export: **i4‑sym, group‑size −1**. Python bindings built into `ov-xpu-env`; `run_4b_hybrid.py` drives
it (default model = sym‑cw; no env flags needed).

---

## 9. Source map

| Concern | File |
|--------|------|
| Buffer alloc / RAII / **L0 import** | `intel_xpu/src/shared_weight_buffer.hpp` (`create_empty`, `import_into_gpu_l0`, `gpu_l0_imported`) |
| Compile pipeline / relocation / defaults (native, malloc, skip‑prefill, REF_COMPRESSED_FC) + **KV‑write graph rewrite** (`xpu_rewrite_present_to_static_slots`) + **prefill logits last‑token slice** (`xpu_slice_prefill_logits_to_last`, §7.1.2) | `intel_xpu/src/plugin.cpp` |
| Buffer ownership / properties / `meta_compiled()` / **compile‑stage KV‑output binding cache** (`build_kv_output_bindings`, §7.1.1) | `intel_xpu/src/compiled_model.{hpp,cpp}` |
| Orchestration: KV‑output bind applied once at request init + **NPU‑request reuse** (signal `XPU_EXTERNAL_PREFILL_LEN`) + **seq_len>1 always‑reprefill dispatch** | `intel_xpu/src/sync_infer_request.cpp` |
| **NPU‑request reuse consume** (`init_from_external_prefill` lazily at `infer()` top + clear) | `npuw/llm_infer_request.cpp` (`m_external_prefill_len`, `LLMInferRequest::infer`) |
| Debug‑print gate (`OV_XPU_DEBUG`) | `intel_xpu/src/xpu_debug.hpp`, `npuw/xpu_debug.hpp` |
| GPU **L0** backend: share+import / kernel build / launch / memory (§6.5) | `intel_gpu/src/runtime/ze/ze_engine.cpp` (`reinterpret_handle`/`share_usm`), `ze_kernel_builder.cpp` (`zeModuleCreate` OCL‑C/native), `ze_stream.cpp` (`zeCommandListAppendLaunchKernel`), `ze_memory.cpp` (`zeMemAlloc*`); `GPU_RT_TYPE` in `cmake/features.cmake` |
| GPU zero‑copy bind (weights + KV) / `OV_XPU_REF_COMPRESSED_FC` decline / output binding | `intel_gpu/src/plugin/ops/constant.cpp`, `program_builder.cpp`, `.../fully_connected_kernel_bf_tiled.cpp` (decline), `intel_gpu/src/plugin/sync_infer_request.cpp` (`prepare_output` static‑output bind) |
| GPU FC execution (INT8×INT4 on XMX) | `intel_gpu/src/graph/impls/onednn/fully_connected_onednn.cpp` |
| NPUW closure / weightless / DQ pattern (CWi) | `npuw/partitioning/partitioning.cpp` (1787‑1802, `matchRepeatedSubgraphs`/`createFunction`), `npuw/partitioning/patterns/opt.cpp` (`DQMatMulCWi`) |
| NPUW bank zero‑copy views + raw‑share | `npuw/weights_bank.cpp` (`evaluate_and_allocate_on_device`, `raw_resident_ptr`, `consolidate_to_xpu_buffer`, `set_xpu_raw_ranges`) |
| NPUW LLM compile / prefill‑skip / compile‑time ranges / KV layout | `npuw/llm_compiled_model.cpp` (`m_prefill_compiled` alias, `XPU_RAW_WEIGHT_RANGES`, `NPUW_KVCACHE_LAYOUT`) |
| NPUW weightless serialization / blob | `intel_npu/src/compiler_adapter/src/{model_serializer.cpp,compiler_impl.cpp,weightless_graph.cpp}` |
| Const source ptr / KV binding | `npuw/lazy_tensor.{hpp,cpp}`, `npuw/llm_infer_request.cpp` |
| POC (malloc dual‑import validation) | `C:\yqiu\poc-npu-gpu-buffer-sharing` (Approach 3 / Mode 11, `result.md`) |
| One‑copy proof (§10) | `prove_shared_buffer.py`; properties `XPU_WEIGHT_RANGES`, `XPU_VERIFY_GPU_L0` (`intel_xpu/src/compiled_model.cpp`) |
| NPU‑request reuse / multi‑conversation token‑exactness (§7.1) | `test_npu_reuse.py` (reused == fresh, variant switch, both modes) |

---

## 10. Verifying the one‑physical‑copy property (`prove_shared_buffer.py`)

The central claim — GPU and NPU read **one** shared buffer, neither holds a private weight copy — is
proved three independent ways by `prove_shared_buffer.py`, using two read‑only helper properties on the
XPU compiled model: **`XPU_WEIGHT_RANGES`** (per‑segment `ptr:size`, so a host test can poke a real weight)
and **`XPU_VERIFY_GPU_L0`** (an L0 audit of every weight segment + the KV buffer in the GPU's L0 context).

### 10.1 Memory accounting — only one copy exists
Measured (Qwen3‑4B, native default):

| | Value | Meaning |
|---|---|---|
| shared weight segments | 2209 MB | the **one** copy GPU+NPU read |
| NPU persistent buffer | 0.8 MB | not a 2nd weight copy |
| process committed | ~6.9 GB | ≈ baseline + **one** 2.2 GB copy (two copies would be ~9 GB) |
| GPU `usm_device` | 170 MB | KV‑write graph buffers — **not** the weights, **not** the KV buffer (that's HOST, §10.3) |

### 10.2 Isolated host‑poke (definitive, behavioral)
Corrupt **one** 12.5 MB transformer FC weight in the shared malloc from the host (`ctypes.memset`), timed
to isolate each engine (a private device copy would be immune to a host poke of the shared buffer):
- **poke BEFORE prefill** → the GPU‑prefilled first token changes (`12095 → 264`) ⇒ **GPU reads the shared buffer**.
- **poke AFTER prefill** → first token stays `12095` and the KV handoff is identical (prefill ran on the
  original weights), yet the **NPU decode changes** (`…3639,15072… → …17968,603…`) ⇒ **the NPU read the
  poked weight directly** — not because its prefill/KV *input* differed (this timing rules out the obvious
  confound). Restoring the bytes returns the output to the exact baseline.

### 10.3 GPU Level‑Zero identity
For all **508/508** weight segments **and** the KV buffer, the GPU's L0 context reports `base == host_ptr`
and `type = HOST(_IMPORTED)`, with **`0` of type `DEVICE`** — the GPU addresses the same imported host
mallocs (the shared buffers), with no device‑resident weight or KV copy. (The NPU side is covered
behaviorally by §10.2; its internal L0 context is not exposed for a direct query.)

Together: §10.1 shows no second copy exists, §10.2 shows both engines read the live shared bytes, and §10.3
shows the GPU's mapping *is* the shared host malloc.

---
---

# Appendix A — DCOFF host‑unpack path (legacy / KV‑cache concept proof)

> **Not the target.** This path keeps weights u4 in the shared buffer but the **NPU executes f16** by
> unpacking on the host CPU each inference — slower (~168 ms/tok) and with a 4× device f16 expansion. It is
> kept only because it was the first end‑to‑end working pipeline (it proved the KV‑cache sharing and the
> weight‑sharing mechanics) and as a fallback on builds/models where native INT4 cannot compile. Enable with
> `XPU_DCOFF=1` (and typically `XPU_USM_WEIGHTS=1` on an OCL GPU build).

### A.1 Quantization — opposite of native
DCOFF requires a **u4‑asymmetric, group‑128** export (`optimum-cli … --weight-format int4 --group-size 128`,
**no `--sym`**). NPUW's DCOFF leaves each FC weight a plain u4 `Const` (byte‑identical, shareable), whereas
i4‑symmetric would trigger the `DQMatMulGQi` permute *and* a VCL lm_head crash. So the two paths need opposite
exports: native wants **i4‑sym g‑1**, DCOFF wants **u4‑asym g128**.

### A.2 Layout (group‑128)
The bulk weights are `u4 [out, groups, 128]` (3‑D, grouped) + `f16 [out, groups]` scale + `u4` zero‑point —
versus native's 2‑D `i4 [out, in]` + `f16 [out, 1]` (no zp). The KV cache and embedding layouts are identical
to the native path (§2.1, §3).

### A.3 Execution / memory flow
`NPUW_DCOFF_TYPE=f16` + `NPUW_DCOFF_SCALE=YES` relabel the weight params to f16. The i4→f16 dequant runs **on
the host** (AVX2 `unpack_closure`, `base_sync_infer_request.cpp`) each inference, reading `(u4 − zp)·scale`
from the shared buffer and writing f16 into the NPU's L0 input tensors. The weight bytes stay u4 in the shared
buffer (zero‑copy), but the **device executes f16** — a 4× expansion (transient per block) and a per‑token host
unpack, which is why decode is ~168 ms/tok vs native's 50.

### A.4 Buffer mode
On an OCL GPU build (`GPU_RT_TYPE=OCL`), the weight buffers are **OCL USM‑host** (`clHostMemAllocINTEL`,
`XPU_USM_WEIGHTS=1`). The GPU shares them via OCL `share_usm` (`ocl_engine.cpp reinterpret_handle`, with a
`get_usm_allocation_size` check that requires a real USM allocation — a malloc returns 0). The NPU never
imports them; it reads them **on the CPU** during host‑unpack. This is why DCOFF works on the OCL build where
native (which needs the NPU to import a neutral malloc, §1) cannot share with the GPU.

### A.5 Why it was kept
DCOFF was the path that first delivered: GPU‑prefill + NPU‑decode end‑to‑end, the shared **KV cache**, the
per‑weight buffer scheme, and the zero‑copy weight raw‑sharing on the NPU host side. Native INT4 reuses all of
that machinery (the KV bridge §3, the per‑weight buffers §1.2, the bank raw‑share §5.2) and replaces only the
NPU execution (host‑unpack‑f16 → on‑device i4) and the buffer mode (OCL‑USM → malloc + dual L0 import). The
historical development of these mechanisms is in `ARCHITECTURE_gpu_prefill_npu_decode.md` and
`DESIGN_gpu_weight_sharing.md`.
