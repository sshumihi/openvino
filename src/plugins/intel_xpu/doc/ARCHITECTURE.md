# XPU Shared-Buffer Architecture: One-Copy Native-INT4 Weights + KV Cache for GPU-Prefill + NPU-Decode

The `intel_xpu` meta-plugin fuses the Intel **GPU** and **NPU** into a single `compile_model`, running an LLM as
**GPU prefill (seq_len > 1) -> NPU decode (seq_len == 1)** while both engines read **one physical copy** of the
INT4 weights and share the KV cache. Target platform: LNL / PTL (Core Ultra, "AI Boost" NPU + iGPU, unified
memory), **GPU built on the Level-Zero backend** (`GPU_RT_TYPE=L0`).

**Default = native INT4.** The weights stay 4-bit end-to-end: GPU prefill dequantizes them in-kernel and NPU
decode runs them as on-device INT4 MACs - no host unpack, no f16 expansion, one shared buffer. This is the
target path the document describes. A legacy **DCOFF host-unpack** path (Appendix D) exists only as a fallback /
KV-cache concept proof and does not meet the perf/memory target.

The document is in three parts: **Part I - Method** (the sharing mechanism and plugin design),
**Part II - Experiment** (the Phi-4-14B measurements), and **Part III - Appendix** (source map, optimization
history, the one-copy proof, and the legacy DCOFF path).

**Headline (microsoft/phi-4, Phi3 14B, i4-sym per-channel, 1024-token prompt, `GPU_RT_TYPE=L0`, no env flags):**

| | Pure GPU | Pure NPU | **XPU hybrid** |
|---|---|---|---|
| Prefill (first token) | 619 ms | 1982 ms | **660 ms** (= GPU compute) |
| Decode (steady, ms/tok) | 69.3 | 92.4 | **90.4** (= NPU parity) |
| Weight copies in memory | 1 (GPU-resident) | 1 (NPU device bank) | **1** (7.31 GB, **shared** GPU+NPU) |
| Process committed | n/a | n/a | **~9.8 GB** |

The hybrid prefills at GPU speed (3x faster than the NPU) and decodes at NPU speed, over a **single** weight copy
both engines address - the GPU never holds a private weight copy, and the NPU adds no second copy of the
transformer weights.

---
---

# PART I - METHOD

## 1. Low-level buffer-sharing mechanism

The hard problem: the **GPU plugin runs on Level-Zero** (`GPU_RT_TYPE=L0`) and the **NPU plugin runs on
Level-Zero** - but as **separate L0 contexts/drivers**. A buffer one context *allocates* is not *importable* by
the other. Empirically (POC `C:\yqiu\poc-npu-gpu-buffer-sharing`, reproduced on this box):

| Allocation | Cross-context importable? |
|---|---|
| context-owned `zeMemAllocHost` (GPU ctx) | no - "part of an existing allocation" |
| OCL `clHostMemAllocINTEL` (OCL GPU build) | no (NPU-L0 cannot import it) |
| **page-aligned `_aligned_malloc`, imported into each L0 ctx** | **yes - same pointer** - the shared-weight design |

### 1.1 The shared buffer: malloc + dual L0 import (`shared_weight_buffer.hpp`)
`SharedWeightBuffer` (RAII) owns one host buffer per weight and hands out `host_ptr`. Default allocation is a
neutral `_aligned_malloc(size, 4096)`, **imported independently into each engine's L0 context** - the import
returns the *same* pointer, so both engines address the identical physical pages:
```cpp
// SharedWeightBuffer::import_into_gpu_l0(gpu_ctx)   - registers the malloc in the GPU's L0 context
ze_ctx = (ze_context_handle_t) gpu_ctx->get_property()["OCL_CONTEXT"];   // == ze ctx on the L0 GPU backend
ze_external_memmap_sysmem_ext_desc_t sys{ ...SYSMEM..., .pSystemMemory = host_ptr, .size = alloc_size };
ze_host_mem_alloc_desc_t  h{ ...HOST_MEM..., .pNext = &sys };
zeMemAllocHost(ze_ctx, &h, alloc_size, 4096, &imported);   // imported == host_ptr
```
- **GPU-L0 import** (above, at compile) makes the malloc a registered L0 allocation so `share_usm` /
  `zeMemGetAddressRange` works on it (3.4.2).
- **NPU-L0 import** happens when the NPU binds the weight closure as an L0 input at inference (3.1.3) - the same
  `zeMemAllocHost` system-memory import, succeeding because it is a neutral malloc (not a context-owned alloc).

The XPU plugin links `ze_loader`; `ze_loader.dll` ships in the Release dir. Cleanup `zeMemFree`s the import
handle before `_aligned_free`. (System-memory import requires LNL/PTL drivers; MTL's older NPU driver rejects it.)

### 1.2 Why "one physical copy" holds
The same `host_ptr` is: written once (memcpy from the mmap'd `.bin`), imported into the GPU-L0 context (GPU reads
it via `share_usm`), and imported into the NPU-L0 context (NPU reads it natively). No engine keeps a private
device copy of the weights - **proved three independent ways in Appendix C** (memory accounting, a host-poke
mutation that changes both engines' output, and a GPU-L0 audit showing every weight maps `type=HOST`,
`base==host_ptr`, zero `DEVICE` allocations). The imported malloc is *not* counted as GPU-allocated `usm_host`
(it is external), so the GPU stats show only its own scratch.

---

## 2. Weight format: native INT4, i4-symmetric per-channel

NPUW turns a submodel's weight `Constant`s into **closures** - graph `Parameter`s fed at runtime from a weights
bank (3.1). A closure carries a `LazyTensor` whose `const_source()` returns `{ptr, bytes}` for a plain
un-transformed Const - that pointer is the per-weight buffer, so the closure is **byte-identical** to the shared
buffer and zero-copy-shareable.

The export must be **i4-symmetric, group-size -1 (per-channel)**:
- **Symmetric** so there are no zero-points - the compiler-dynamic-quant INT4 matmul does not accept asym zp.
- **Per-channel (-1)** so the partition uses the **`DQMatMulCWi`** pattern (`partitioning/patterns/opt.cpp`),
  which **does not permute** the weight (it only flips the matmul's `transpose_b`). The weight stays in its `.bin`
  row-major layout = byte-identical to the shared buffer. (Group-size 128 would use `DQMatMulGQi`, which
  `permute({0,2,1})`s the weight into an NPU-specific layout -> not shareable.)

NPUW runs its baseline **compiler-dynamic-quant** path: `NPUW_DQ=YES` + `NPU_COMPILER_DYNAMIC_QUANTIZATION=YES`,
no DCOFF. The VCL compiler emits an on-device INT4 matmul (weights stay i4, activations dynamically quantized to
int8).

### 2.1 Exact layout in the shared buffer (Phi-4-14B)
Each segment holds one `Constant`'s bytes **verbatim from the `.bin`** (no repack). Phi-4 (Phi3, 14B): hidden
5120, 40 layers, 40 Q / 10 KV heads, head_dim 128, intermediate 17920, vocab 100352, **fused** qkv and gate_up
projections, **untied** embed/lm_head. Confirmed from the IR:

| Tensor (per layer unless noted) | OV type / shape | bytes | note |
|---|---|---|---|
| `qkv_proj.weight` (fused Q+K+V) | **`i4 [7680, 5120]`** | 18.8 MB | Q 5120 + K 1280 + V 1280; sym i4, packed 2/byte, row-major `[out, in]` |
| `o_proj.weight` | `i4 [5120, 5120]` | 12.5 MB | sym i4 |
| `gate_up_proj.weight` (fused) | `i4 [35840, 5120]` | 91.8 MB | 2 x intermediate 17920; sym i4 |
| `down_proj.weight` | `i4 [5120, 17920]` | 45.9 MB | sym i4 |
| every `*_proj.weight/scale` | **`f16 [out, 1]`** | out x 2 B | **per-output-channel** scale, **no zero-point** |
| `input/post_attn_layernorm` | `f32 [1, 1, 5120]` | 20 KB | RMSNorm weights |
| `embed_tokens.weight` (1x) | **`u8 [100352, 5120]`** | 490 MB | int8: + `u8` zp + `f16` scale |
| `lm_head.weight` (1x, **untied**) | **`u8 [100352, 5120]`** | 490 MB | int8 (output proj kept higher-precision) |

Per-layer i4 weight = 18.8 + 12.5 + 91.8 + 45.9 = **168.9 MB** x 40 layers = **6.6 GB**, plus the two int8
embed/lm_head (490 MB each). **Total relocated: 407 segments = 7.31 GB** (measured `XPU_WEIGHT_SEGMENT_BYTES`).
So the bulk transformer weights are **plain row-major `i4 [out, in]`** with one f16 scale per output channel and
**no zero-point**; only embed and lm_head are int8.

### 2.2 How each engine consumes those bytes
Both engines read the **same `i4 [out, in]` + `f16 [out,1]` scale** segments; they differ only in *where* the
dequant/matmul happens:
- **GPU prefill (oneDNN INT8xINT4, 3.4.1)** reads the i4 weight as row-major `oiyx` (no reorder), dynamically
  quantizes the f16 activation to int8, runs the int8.int4 matmul on the XMX/DPAS systolic array, and applies the
  per-output-channel `f16 scale` - symmetric, no `-zp` term. The embedding is a `Gather` over the `u8` rows. All
  weight reads are zero-copy from the shared buffer (L0 `share_usm` on the imported malloc, 3.4.2).
- **NPU decode (native INT4)** keeps the weight **4-bit end-to-end**: the i4 closure is a view of the shared
  malloc (3.1.3), bound as an L0 input; the NPU L0 backend imports that malloc and the VCL-emitted kernel streams
  the 4-bit weights into SRAM, dynamically quantizes the activation to int8, runs an int8.int4 MAC, and applies
  the per-channel `f16 scale`. No host unpack, no f16 weight expansion.

One layout, one copy, two native consumers.

---

## 3. XPU plugin design

The XPU plugin is the orchestrator; the GPU and NPU plugins never talk to each other. It manages four things:
the **shared weights** (3.1), the **shared KV cache** (3.2), **model compilation** (3.3), and the
**NPU/GPU inference pipeline** (3.4).

### 3.1 Shared weight management

#### 3.1.1 Per-weight relocation + segmentation
At compile (`plugin.cpp` step 3) each model `Constant` >= 4096 B is copied once into its own `SharedWeightBuffer`
(malloc + GPU-L0 import) and the original `Constant` is replaced by one viewing that buffer:
```cpp
buf = SharedWeightBuffer::create_empty(gpu_ctx, bytes, /*zero=*/false);  // malloc + import_into_gpu_l0
std::memcpy(buf->host_ptr, c->get_data_ptr(), bytes);
new_c = std::make_shared<Constant>(c->get_element_type(), c->get_shape(), buf->host_ptr, /*so=*/buf);
ov::copy_runtime_info(c, new_c);
ov::copy_weightless_cache_attr(c, new_c);   // <- preserve WeightlessCacheAttribute (see 6.3); without it NPUW
ov::replace_node(c, new_c);                  //    eagerly host-copies every weight (~7 GB) at partition time
```
The store is **not** one buffer - per-weight segmentation lets the consolidation point each closure at exactly
one segment, and keeps every allocation < 2 GB (a 2^31 boundary in the GPU `share_usm` path makes a single > 2 GB
host allocation unaddressable). The compiled model owns `std::vector<SharedWeightBuffer> m_weight_segments`; the
ranges are published as a serialized `"ptr:size;..."` list (Phi-4: **407 segments, 7.31 GB**).

#### 3.1.2 Weightless blob - weights are graph inputs, not baked in
NPUW partitions the model and, for repeated transformer blocks, compiles the block **once** (funcall/REP reuse)
and runs it 40x with different weights. In `partitioning.cpp` each bulk weight `Constant` is replaced by a new
`Parameter` and the original captured as a `LazyTensor` closure; only tiny scalar consts stay inline.
Serialization to the VCL compiler uses **`NO_WEIGHTS_COPY`** + the weightless `compileWsOneShot` flow, so the
**main L0 blob contains no weights** - at inference they are read from the shared buffer. Phi-4: the decoder VCL
compile adds only **+29 MB** committed.

#### 3.1.3 Compile-time bank (zero-copy raw-share) and inference flow
The weights bank used to copy each weight into an NPU device tensor (`create_host_tensor + copy_to`). Now: when a
closure's `const_source()` lies in `m_xpu_raw_ranges` (passed at *compile* via `XPU_RAW_WEIGHT_RANGES`, registered
on the bank **before** eval), the bank stores a zero-copy `ov::Tensor(meta, ptr)` view of the shared buffer and
skips the alloc+copy. Phi-4: of **406** closures, **404 raw-shared** (zero-copy), **2 materialized** (small)
(`[DIAG] raw_views=404`). This removes the NPU device weight copy; because the NPU now reads weights from the
shared buffer rather than a per-token device copy, hybrid decode matches standalone NPUW (no shared-buffer
penalty - 5.1). At inference the i4 closure (a view of the shared malloc) is set as the NPU L0 input; the NPU L0
backend **imports** the malloc (1.1) and the on-device INT4 kernel reads 4-bit weights into SRAM. The
non-raw-shared residue is a tiny **0.6 MB** persistent buffer (Phi-4 `XPU_SHARED_WEIGHT_SIZE`).

### 3.2 KV cache management

One contiguous `kvcache` buffer of per-layer K/V slots ordered by **alphabetical port name**, f16. Phi-4 (10 KV
heads, head_dim 128, 40 layers, `total_size` = align(max_prompt,64)+align(min_response,64) = 1151 at 1K):
```
past_key_values.0.key    [1, 10, 1151, 128]            f16   (2.81 MB/slot)
past_key_values.0.value  [1, 10, 128, 1151]   <-TRANSPOSED   (v_tensors_transposed_gen)
...  prefix-sum offsets emitted by NPUW_KVCACHE_LAYOUT
```
**80 slots** (40 layers x K+V), total **224.8 MB** (measured `XPU_SHARED_KVCACHE_SIZE = 235,724,800 B`),
host-imported into both L0 contexts (same dual-import as the weights).

**NPU side**: `create_generate_request_variants` binds each generate-model `past_key_values.*` input to
`kvbuf + offset` via `make_tensor` + `set_tensor`, and aliases smaller KV-size variants onto the largest. NPU
decode reads/writes its KV **in place** in the shared buffer.

**GPU -> NPU handoff (GPU KV-write, default)**: the GPU runs a **stateless** clone
(`ov::pass::StatefulToStateless`) exposing `present.N.{key,value}` and **writes the KV in the NPU's exact format,
zero-copy**:
- **Graph rewrite** (`xpu_rewrite_present_to_static_slots`, `plugin.cpp` step 6f): each `present.N.{key,value}` is
  rewired to `Convert(f16)` -> *(value)* `Transpose([0,1,3,2])` -> `ScatterUpdate(Broadcast(0, slot_shape),
  Range(0, seq), upd, axis)`, where `slot_shape` / `axis` / the transpose come from `NPUW_KVCACHE_LAYOUT` +
  `NPUW_KVCACHE_V_TRANSPOSED_GEN`. ScatterUpdate places the dynamic-seq K/V into rows `[0, prompt_len)` of a
  **static** slot - the *static* output shape is what makes it zero-copy-bindable.
- **Zero-copy bind**: each `present.*` output is bound to `gpu_ctx->create_tensor(f16, slot,
  {SHARED_MEM_TYPE=USM_USER_BUFFER, MEM_HANDLE=kvbuf+offset})`; `intel_gpu`'s `prepare_output` binds it as a
  static no-convert `RemoteTensorImpl` -> the GPU writes the slot **in place**. The 80 remote tensors are created
  **once at compile** (`build_kv_output_bindings`) and `set_tensor`'d once at request init (Appendix B.1).
- **Ordering / adoption**: the NPU request is created **once** and reused; each prefill *signals* the prompt
  length via `XPU_EXTERNAL_PREFILL_LEN`, consumed at the top of the NPU's next `infer()` (the first decode), which
  re-selects the generate variant and sets `num_stored_tokens=prompt_len` - adopting the GPU's KV without
  re-running prefill. No pre-infer memset: the `ScatterUpdate(Broadcast(0, ...))` rewrites the full slot during
  the (blocking) prefill, zeroing unused rows and overwriting any stale KV.

*Why a separate pass, not fused into SDPA:* the fast **micro** SDPA kernel hides K/V inside an opaque
compiler-generated `ugemm` micro-GEMM (no editable K/V loads); fusing would force the slower `sdpa_opt` kernel for
a ~1 ms saving. The stock-ops pass keeps micro and still does the GPU cast+layout+direct write.

### 3.3 Model compilation pipeline (`Plugin::compile_model(filesystem::path)`)
The string-path entry is the hybrid pipeline (`core.compile_model("...xml","XPU")` reaches it via
`to_fs_path -> CoreImpl::compile_model(path)`):
```
 1  set OV_XPU_REF_COMPRESSED_FC=1 if unset          // GPU reads i4 with no reorder (3.4.1)
 2  model = read_model(xml, bin, {})                  // .bin MMAP'd (lazy pages)
 3  for each Constant >=4096B:                         // PER-WEIGHT relocation (3.1.1)
        buf = create_empty(gpu_ctx, bytes, zero=false) //   malloc + import_into_gpu_l0
        memcpy; replace_node(c, Constant(...,buf->host_ptr)); copy_weightless_cache_attr; ranges += "ptr:size;"
 5  if (all segments GPU-L0-imported) model.rt_info[xpu_shared_weight_ranges] = ranges  // GPU shares
 6  npu = compile("NPU", { NPU_USE_NPUW, NPUW_LLM, native (NO DCOFF), XPU_SKIP_NPU_PREFILL (default),
                           XPU_RAW_WEIGHT_RANGES=ranges, NPUW_DEVICES=NPU, NPU_COMPILER_TYPE=PLUGIN })
 6d npu.set_property(XPU_SHARED_WEIGHT_PTR = persistent)  // consolidate: raw-share + copy residue
 6e kvcache = create_empty(NPUW_KVCACHE_TOTAL_BYTES);  npu.set_property(XPU_SHARED_KVCACHE_PTR = kvcache)
 6f m = StatefulToStateless(model.clone())            // present.*/past_key_values.* ports
        rewrite_present_to_static_slots(m, NPUW_KVCACHE_LAYOUT)   // 3.2
        slice_prefill_logits_to_last(m)                          // Appendix B.2
        gpu_prefill = compile("GPU", m)
 7  gpu_full = SKIP (hybrid)                            // 3.3.1 - full stateful GPU model not built
 8  XpuCompiledModel{ npu, gpu_prefill, weight_segments[], persistent, kvcache }
```
NPU is compiled first (stricter consumer; its bank evaluation fixes the exact persistent size). **Property
plumbing**: the GPU learns the ranges via the model's **rt_info** `xpu_shared_weight_ranges` (parsed by
`ProgramBuilder`, set **only when every segment was L0-imported**); the NPU learns them via the **property**
`XPU_RAW_WEIGHT_RANGES` (compile-time zero-copy views), persistent via `XPU_SHARED_WEIGHT_PTR`, KV via
`XPU_SHARED_KVCACHE_PTR`.

#### 3.3.1 Skipping unused models (default in hybrid)
The GPU only runs the **stateless prefill** model, so: (a) the full stateful GPU model (step 7) is **not built**
(`XPU_COMPILE_GPU_FULL=1` re-enables it); (b) the **NPU prefill model** is the GPU's job, so NPUW aliases
`m_prefill_compiled = m_kvcache_compiled` instead of compiling a second model. This is the dominant compile-time
and memory saving - Phi-4 hybrid compiles in **15 s** vs the pure-NPU standalone's **101 s** (which compiles its
own prefill model), and the alias saves a second multi-GB compile.

### 3.4 NPU/GPU inference pipeline

#### 3.4.1 Dispatch (`XpuSyncInferRequest::infer`)
```
if active_device == "NPU"                     -> infer_npu_decode()   // pure NPUW (prefill+decode on NPU)
else if gpu_prefill && (seq_len>1 || !done)   -> infer_gpu_prefill()  // (re)prefill: new conversation / chat turn
else if prefill_done                          -> infer_npu_decode()   // NPU decode per token (seq_len==1)
else                                          -> infer_passthrough()  // GPU-only (asserts if gpu_full skipped)
```
A `seq_len>1` infer is **always** a (re)prefill - a new conversation or chat turn - so it routes to GPU prefill
even after a prior prefill on the same request (there is no `!prefill_done` latch on the prefill branch; an
earlier such latch sent a 2nd conversation's prefill into NPU decode and crashed - Appendix B.3). `prefill_done`
routes only post-prefill single-token infers.

#### 3.4.2 GPU compute-graph adaptation to NPU-managed formats
The GPU must consume the **same physical i4 bytes** NPUW keeps and produce KV in the NPU's layout - with **no host
transcoding on either side**.
- **Weights - no reorder (FCs run on oneDNN)**: `OV_XPU_REF_COMPRESSED_FC=1` (the plugin sets it) declines
  compressed-INT4 in the OCL `bf_tiled`/`gemv` kernels (which request a *blocked* `os_is_yx_osv32_isv2` layout), so
  `constant.cpp` keeps the weight in `oiyx` + `share_usm` -> the weight pointer stays in the shared buffer,
  zero-copy. What actually executes the FCs is **oneDNN** (verified: `FullyConnected` nodes, `runtimePrecision=i8`,
  paired with `DynamicQuantize`): the GPU dynamic-quantizes the f16 activation -> int8 and runs an **INT8xINT4 GEMM
  on the XMX/DPAS systolic array**, reading the shared `oiyx` i4 weights directly (`usm_device` ~ 0 for weights).
- **GPU zero-copy bind on L0**: `constant.cpp` `is_shared_weight_ptr(data) -> engine::share_usm(layout, ptr)`; on
  the **L0 GPU backend** (`ze_engine.cpp` `reinterpret_handle`) this wraps `ptr` via `zeMemGetAddressRange` ->
  `ze::gpu_usm`. It works on the **imported malloc** (1.1) - the malloc the NPU also imports is GPU-zero-copy too.
  (An OCL GPU build's `share_usm` requires a real USM allocation and would reject a malloc -> the GPU copies; that
  is why **L0 is the build requirement** for the one-copy property.)
- **KV output**: the GPU emits the KV already in the NPU's static `[1, H_kv, total_size, head_dim]` f16
  value-transposed slot (3.2) and the slot is bound zero-copy to the shared buffer.

#### 3.4.3 What "the L0 backend" means
`GPU_RT_TYPE=L0` selects the GPU plugin's **runtime** (memory, queue, kernel compile + submit) - **not** the
kernel language. The compute kernels are the *same* OpenCL-C kernels in `cl_kernels/` (plus the micro-GEMM); only
*how* they are built (`zeModuleCreate`, OCL-C or native vISA) and launched (`zeCommandListAppendLaunchKernel`)
changes. So a trace under L0 showing OCL-named kernels (`sdpa_micro_prefill`, `dynamic_quantize_gpu_opt`) is
expected. Crucially, the L0 `ze_context` (the GPU `RemoteContext`'s `OCL_CONTEXT`) is what the XPU plugin imports
the shared malloc into and what `share_usm` wraps - the OCL backend cannot import a neutral malloc.

---
---

# PART II - EXPERIMENT

## 4. The Phi-4-14B experiment

**Model**: `microsoft/phi-4` (`Phi3ForCausalLM`, **14B**) - GQA 40 Q / 10 KV heads, head_dim 128, hidden 5120,
40 layers, intermediate 17920, vocab 100352, 16k ctx, fused `qkv_proj` + `gate_up_proj`, **untied** embed/lm_head.
Enabled on the XPU hybrid with **zero code changes** - the pipeline is **architecture-agnostic** (NPUW has no
model-type whitelist; the XPU rewrites are driven by `NPUW_KVCACHE_LAYOUT` + the LM-head MatMul pattern, with no
hardcoded heads/dims).

**Export**: `optimum-cli export openvino --model microsoft/phi-4 --weight-format int4 --sym --group-size -1
--ratio 1.0` -> 7.85 GB `.bin` (93% i4-sym per-channel; embed + lm_head int8). Portable to the custom XPU build
(both OV 2026.x). (HF download needs `HTTPS_PROXY=http://proxy-dmz.intel.com:912`; the box's non-standard
`DEVTOOL_SYSTEM_HTTPS_PROXY` is ignored by huggingface_hub.)

**Measurement harness**: `bench_npuw_vs_xpu.py {gpu|npuw|xpu} 1024 32 1 -m phi-4...` on this build - one mode per
process, warm-up excluded, steady decode = mean of tokens after the 6th. All three configs use the **same build,
same harness** for an apples-to-apples comparison: `gpu` = stock `compile_model(xml,"GPU")` (optimized blocked
INT4 kernel, no shared buffer); `npuw` = standalone native-INT4 NPUW on NPU; `xpu` = the hybrid. Output is
**token-exact vs CPU** (16/16; the partition-copy fix of 6.3 is numerically identical - re-verified token-exact on
Qwen3-4B). The measurements are not inflated by any prefix/prompt cache - prefill scales O(L^2) and is unchanged
under unique random prompts (validity checks in 5.1).

## 5. Performance comparison

Phi-4-14B, 1024-token prompt, 32 generated tokens:

| Config | Time-to-first-token (prefill) | Decode (steady ms/tok) | Decode throughput | Compile |
|---|---|---|---|---|
| **Pure GPU** (stock, GPU-managed KV) | **619 ms** | **69.3** | 14.4 tok/s | 7.7 s |
| **Pure NPU** (standalone NPUW) | **1982 ms** | **92.4** | 10.8 tok/s | 101 s |
| **XPU hybrid** (GPU prefill -> NPU decode) | **660 ms** | **90.4** | 11.1 tok/s | 15 s |

**Prefill - the hybrid prefills at GPU speed.** Hybrid first-token (660 ms) is within ~7% of pure GPU (619 ms)
and **3x faster than the NPU's own prefill (1982 ms)** - the GPU-prefill premise holds at 14B. The hybrid figure
is end-to-end GPU compute after three optimizations (last-token logits slice, NPU-request reuse, compile-stage KV
bind-cache; Appendix B).

**Decode - NPU parity, no shared-buffer penalty.** Hybrid decode (90.4 ms/tok) matches standalone NPUW
(92.4 ms/tok) to within noise: routing decode through the NPU while it reads the *shared* INT4 buffer costs
nothing vs a private-weight NPUW model. Pure GPU decodes faster (69.3 ms/tok) but **occupies the GPU**; the hybrid
decodes on the NPU by design - to free the GPU and keep one weight copy. (Decode for a 14B model streams ~3.5x the
weight bytes per token of a 4B model, hence ~90 vs ~36 ms/tok on Qwen3-4B.)

**Compile.** The hybrid compiles in **15 s** vs the pure-NPU standalone's **101 s**: the hybrid skips the NPU
prefill-model compile (aliased to generate, 3.3.1) and partitions only the decode path.

### 5.1 Validity: prefill scales with sequence length, and no prefix cache is active
The §5 figures are at **1024 tokens**. Two checks confirm they reflect real compute, not a cache:

**(a) Prefill scales super-linearly with prompt length** (pure GPU, Phi-4, fixed prompt):

| Prompt tokens | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| GPU prefill | 0.61 s | 1.26 s | 3.09 s | **10.1 s** |
| vs 1K | 1.0x | 2.1x | 5.1x | **16.6x** |

8x the tokens -> ~17x the time, because attention is O(L^2) - the exact signature of full attention over all
positions. A length-independent short-circuit or a content cache would be flat/sub-linear. So **~0.6 s is the
1024-token prefill**; an 8192-token prefill is ~10 s (the hybrid is ~9.7 s, same GPU engine). A 14B 8K prefill in
0.6 s would indeed be impossible - measured, it is 10 s.

**(b) No prefix/prompt cache is active.** NPUW ships a prefix-caching feature (`PrefixCacheManager`,
hash-of-prompt-tokens block reuse) but it is **default-off** (`NPUW_LLM_ENABLE_PREFIX_CACHING=false`),
per-`InferRequest`, and the bench never enables it; the GPU/NPU raw `infer()` path has no other cross-request KV
reuse, and each measured iteration uses a fresh `create_infer_request()` (KV reset per request). Verified
empirically with `bench --unique-prompt` (a different random prompt every iteration, defeating any content-keyed
cache): prefill is **unchanged** vs the fixed prompt -

| | GPU @1K | NPU @1K | hybrid @1K | GPU @8K |
|---|---|---|---|---|
| fixed prompt | 602 ms | 1965 ms | 648 ms | 10.1 s |
| unique prompt | 591 ms | 1955 ms | 642 ms | 11.0 s |

A content-keyed cache would make the *repeated* prompt far faster; it does not. (`first_tok` also varies with the
unique prompt and with length - 68 at 1K, 365 at 2K, 877 at 4K, 13 at 8K - confirming real per-prompt compute.)

### 5.2 Long context (8192-token prompt): prefill advantage grows, a decode tradeoff appears
The §5 table is at 1K, where prefill and decode are both modest. At **8K** the hybrid's split shows its true
shape - the prefill advantage gets much larger, and a decode tradeoff surfaces.

**Prefill @8K (8192-token prompt)** - `bench {gpu|xpu|npuw} 8192 ...`:

| | Pure GPU | XPU hybrid | Pure NPU |
|---|---|---|---|
| prefill @8K | **8.3 s** | **9.0 s** | **25.0 s** |

The hybrid prefills at ~GPU speed (1.1x) and **2.7x faster than the NPU's own prefill**. The absolute saving vs
pure NPU is **~16 s at 8K** (vs ~1.3 s at 1K): the GPU-prefill advantage **scales with context** - this is the
whole point of the hybrid for long prompts. (GPU @8K prefill is ~8-10 s run-to-run, cf. the cold-scaling ~10 s in
5.1; 9.0 s is the warmed hybrid mean.)

**Decode under 8K context (1024 output tokens, context 8192 -> 9216)** - `bench {xpu|npuw} 8192 1024 1`:

| | XPU hybrid | Pure NPU |
|---|---|---|
| decode @8K ctx | **140 ms/tok** | **115 ms/tok** |

Both are nearly flat across 8192->9216 (first64 -> last64 within a few %; 14B decode is weight-bandwidth-bound, so
+1024 tokens of KV barely move it). But unlike the **parity at 1K context** (hybrid 90 vs NPU 92 ms/tok), at 8K
the **hybrid decode is ~22% slower** than standalone NPU. The gap is purely KV-scaling (zero at 1K, growing with
KV size: hybrid +~7 ms/tok per 1K context vs NPU +~3).

**Root cause (identified).** The decode generate model is byte-for-byte identical between the two paths
(confirmed), both KV buffers live in host memory, and neither does a per-token copy - so it is not the model,
not device-vs-host, and not a staging copy. The one remaining difference is the KV **allocation attribute**: the
NPU backend tags its own input tensors (incl. the native KV) with `ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED`
(`is_input=true`), whereas the XPU shared KV is an *externally imported* `_aligned_malloc` (`zeMemAllocHost` +
`ze_external_memmap_sysmem`) that imports with `is_input=false` -> **no write-combined bias**. Non-WC (cached)
host memory makes the NPU DMA snoop CPU caches for coherency; this is amortized for the **sequential** weight
stream (hence 1K-context parity, where decode is weight-bound) but is the plausible cost for the **strided**
attention reads of a large transposed-V KV (hence the 8K penalty).

**Fix attempt 1 - WC-on-import (ineffective).** Threading `is_input=true` through the NPU backend's user-tensor
import so the shared KV is imported with `ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED` left decode **unchanged**
(131 ms/tok, token output intact). Reason: the WC bias on `ze_external_memmap_sysmem` of an existing
`_aligned_malloc` is a no-op - a page's cache type is fixed at allocation, so importing already-cached (WB) pages
with a WC hint does not convert them (and, because it is ignored, it raises no WB/WC aliasing issue). A genuine
test/fix must **allocate the shared KV as WC from the start** (e.g. Windows `VirtualAlloc(PAGE_WRITECOMBINE)`)
then import into both contexts - then both engines map genuinely-WC pages. That is the remaining lever (the KV is
device-written by the GPU and device-read by the NPU, with negligible CPU access, so WC is acceptable for it),
and would also definitively confirm/refute the WC hypothesis - **not yet done**. (A separate
`XPU_ACTIVE_DEVICE=NPU` A/B was inconclusive: that mode keeps the NPU prefill model resident and has its own ~4x
weight-read regression, unrelated to the KV.)

**Net end-to-end (1024-token generation @8K).** Prefill saving (hybrid vs pure NPU) ~16 s; decode penalty
~25 ms/tok. Break-even is **~640 output tokens**: below it the hybrid is faster end-to-end than pure NPU, above it
pure NPU edges ahead. So the hybrid is strongest for **long-prompt, short-to-moderate-output** workloads (RAG,
summarization, extraction, classification) - exactly the long-context regime - while keeping decode off the GPU.
(Pure GPU is fastest end-to-end - ~8.3 s + ~89 ms/tok - but occupies the GPU; freeing it is the hybrid's reason
to exist.)

## 6. NPU / GPU memory breakdown (XPU hybrid)

The target: routing through the NPU + GPU adds **no second copy** of the model on top of the one shared buffer.
Staged committed-memory breakdown of the full hybrid compile (`XPU_MEM_DEBUG=1`, Phi-4, @1K, at the
`[XPU][MEM]` / `[NPUW][MEM]` / `[CM][MEM]` checkpoints):

| Stage | Committed | Delta | What it is |
|---|---|---|---|
| after `read_model` | 636 MB | - | IR graph; `.bin` mmap'd (lazy, not committed) |
| after relocate weights | 8139 MB | **+7504** | **the one shared copy** (407 segs, 7.31 GB i4+int8; GPU+NPU read it) |
| after decoder `getPartitioning` | 8202 MB | **+63** | weightless funcalls + `LazyTensor` closures (was ~+7 GB before the 6.3 fix) |
| after decoder VCL compile | 8231 MB | **+29** | **weightless** main blob + 2 small materialized bank tensors |
| after lm_head VCL compile | 8738 MB | **+502** | **lm_head INT8 weight baked into its device blob** (the one remaining duplicate) |
| after GPU prefill compile | 9994 MB | **+1302** | GPU activations + folded consts + KV-write graph buffers |
| **final** | **9999 MB** | | **~9.8 GB** |

### 6.1 What is shared (one copy, GPU + NPU)
- **Weights: 7.31 GB** (407 segments). One host malloc per weight, imported into both L0 contexts (1.1); GPU reads
  via `share_usm`, NPU reads natively. No private device weight copy on either side (proof: Appendix C).
- **KV cache: 224.8 MB** (80 slots, `[1,10,1151,128]` f16 key + transposed value). A **HOST-imported malloc**
  (`type=HOST`, `base==ptr` in the GPU L0 ctx - verified), **not** `usm_device`; the NPU imports the same buffer.
  GPU writes it in place during prefill, NPU reads/writes it in place during decode.

### 6.2 NPU-specific overhead (~595 MB)
| Cost | Phi-4 | Avoidable? |
|---|---|---|
| **lm_head INT8 device blob** | **~502 MB** | the one remaining weight duplicate - see below |
| decoder weightless VCL blob + 2 materialized bank tensors | ~29 MB | no - the compiled program + non-shareable scalars |
| partition (funcalls + `LazyTensor` closures) | ~63 MB | already minimized (was ~7 GB - 6.3) |
| persistent host-closure buffer (`XPU_SHARED_WEIGHT_SIZE`) | **0.6 MB** | minimized (raw-share covers 404/406 closures) |
| NPU L0 per-block activation scratch | device, block-reuse-bounded | no - compute working set (one block resident) |

What is **optimized away** (each removed a full weight copy): the weightless decoder blob (+29 MB, not ~6.6 GB),
the NPU prefill-model skip (saves a second multi-GB compile, 3.3.1), the bank raw-share (404/406 zero-copy views,
not a ~7 GB device copy), and the partition eager-copy elimination (6.3).

**The lm_head is the only weight the NPU still duplicates.** It is the output projection - `u8 [100352, 5120]` =
**490 MB int8**, kept higher-precision than the INT4 decoder for accuracy. Its submodel is compiled with
`NPUW_ONLINE_PIPELINE=NONE` (`get_default_lm_head_config`), which disables the online partitioner - and with it
the `FUNCALL_FOR_ALL` closure extraction. So the lm_head weight never becomes a `LazyTensor` closure: it stays an
**inline `Constant`**, and the VCL compiler **bakes it into the device blob** (a 490 MB copy on top of the same
bytes in the shared buffer). The contrast is visible in the staged log: the decoder VCL adds +29 MB (weightless),
the lm_head VCL adds +502 MB (weight baked in). Removing it would require routing the lm_head through the
raw-share bank like the decoder funcalls - the next memory item. (The untied **embed** stays shared - it is read
via Gather from the shared buffer by both engines, not baked.)

### 6.3 The partition-copy fix (eliminated ~7 GB)
Before the fix, `getPartitioning` added a **~7 GB second copy** of the decoder weights (it would have been
~+7000 MB here). It was *not* the blob (weightless), *not* the VCL compiler, *not* the bank (zero-copy) - it was
materialized inside `matchRepeatedSubgraphs` / `createFunction`: the XPU relocation (3.1.1) built new
shared-buffer `Constant`s but dropped the `WeightlessCacheAttribute` (its `is_copyable()` returns `false`, so
`ov::copy_runtime_info` silently skips it), and NPUW's `Const` wrapper then treats every relocated weight as "a
new Constant not in the weights file" and **eagerly copies it to host**. Preserving the attribute with
**`ov::copy_weightless_cache_attr`** (`plugin.cpp`, right after relocation) drops `getPartitioning` to **+63 MB**,
with the bank behavior unchanged (`[DIAG] raw_views=404 materialized=2`) and output token-exact. On Qwen3-4B the
same fix took the final footprint from 6.0 -> 3.9 GB (-35%); on Phi-4 it is the difference between ~17 GB and the
measured ~9.8 GB.

### 6.4 GPU-specific overhead (~1.3 GB at compile)
The GPU prefill compile adds **+1302 MB** committed: the prefill activations (1024 tokens x hidden 5120 x layers),
folded constants, and the **KV-write graph buffers** - the static `present.*` outputs (~225 MB, = the KV size) and
the `ScatterUpdate`/`Broadcast(0)` zeros. The 7.31 GB weights are the **imported malloc** (external), so they do
**not** count as GPU-allocated memory; the GPU's own stats show only this scratch. No weight copy (a copy would
add +7.3 GB of GPU `usm_device` - measured ~0 for weights).

## 7. Configuration

| Setting | Default | Override |
|---|---|---|
| GPU backend | `GPU_RT_TYPE=L0` (build) | OCL build -> GPU copies (no one-copy) |
| Weight execution | **native INT4** | `XPU_DCOFF=1` -> legacy host-unpack (Appendix D) |
| Weight buffer | **malloc + dual L0 import** | `XPU_USM_WEIGHTS=1` -> OCL USM-host (Appendix D) |
| NPU prefill model | **skipped** (GPU prefills) | `XPU_ACTIVE_DEVICE=NPU` / `XPU_KEEP_NPU_PREFILL=1` |
| GPU FC impl | **oneDNN** INT8xINT4 (reads shared i4 in place) | - |
| KV-cache sizing | `NPUW_LLM_MAX_PROMPT_LEN` / `..._MIN_RESPONSE_LEN` (must cover prompt+gen) | forwarded to the NPU sub-compile + GPU KV-write |
| Debug prints | off | `OV_XPU_DEBUG=1`; `XPU_MEM_DEBUG=1` ([...][MEM] breakdowns) |

The GPU KV-write (3.2), NPU-request reuse, compile-stage KV-output binding, and last-token logits slice are
**unconditional** - the A/B opt-out flags were removed once validated. Model export: **i4-sym, group-size -1**.
Python bindings are built into `ov-xpu-env`; `bench_npuw_vs_xpu.py` / `xpu_hybrid_llm.py` drive it (no env flags
needed). **KV-cache sizing gotcha**: if `prompt_len > max_prompt_len`, the GPU KV-write `ScatterUpdate` writes
past the static slot -> `zeCommandListAppendLaunchKernel ... code 7000000`; size `NPUW_LLM_MAX_PROMPT_LEN >=
prompt_len`. The bench auto-sizes from its `prompt_len`/`max_new` args.

---
---

# PART III - APPENDIX

## A. Source map

| Concern | File |
|--------|------|
| Buffer alloc / RAII / **L0 import** | `intel_xpu/src/shared_weight_buffer.hpp` (`create_empty`, `import_into_gpu_l0`, `gpu_l0_imported`) |
| Compile pipeline / relocation (+ **`copy_weightless_cache_attr`**, 6.3) / defaults + **KV-write rewrite** (`xpu_rewrite_present_to_static_slots`) + **prefill logits slice** (`xpu_slice_prefill_logits_to_last`, B.2) | `intel_xpu/src/plugin.cpp` |
| Buffer ownership / properties / `meta_compiled()` / **compile-stage KV-output binding cache** (`build_kv_output_bindings`, B.1) | `intel_xpu/src/compiled_model.{hpp,cpp}` |
| Dispatch: KV-output bind at request init + **NPU-request reuse** (signal `XPU_EXTERNAL_PREFILL_LEN`) + **seq_len>1 always-reprefill** | `intel_xpu/src/sync_infer_request.cpp` |
| **NPU-request reuse consume** (`init_from_external_prefill` lazily at `infer()` top) | `npuw/llm_infer_request.cpp` |
| GPU **L0** backend: share+import / kernel build / launch / memory (3.4.3) | `intel_gpu/src/runtime/ze/ze_engine.cpp` (`reinterpret_handle`/`share_usm`), `ze_kernel_builder.cpp`, `ze_stream.cpp`, `ze_memory.cpp`; `GPU_RT_TYPE` in `cmake/features.cmake` |
| GPU zero-copy bind (weights + KV) / `OV_XPU_REF_COMPRESSED_FC` decline / output binding | `intel_gpu/src/plugin/ops/constant.cpp`, `program_builder.cpp`, `.../fully_connected_kernel_bf_tiled.cpp`, `intel_gpu/src/plugin/sync_infer_request.cpp` (`prepare_output`) |
| GPU FC execution (INT8xINT4 on XMX) | `intel_gpu/src/graph/impls/onednn/fully_connected_onednn.cpp` |
| NPUW closure / weightless / DQ pattern (CWi) | `npuw/partitioning/partitioning.cpp` (`matchRepeatedSubgraphs`/`createFunction`), `npuw/partitioning/patterns/opt.cpp` (`DQMatMulCWi`) |
| NPUW bank zero-copy views + raw-share | `npuw/weights_bank.cpp` (`evaluate_and_allocate_on_device`, `set_xpu_raw_ranges`) |
| NPUW LLM compile / prefill-skip / compile-time ranges / KV layout / **lm_head cut** (`get_default_lm_head_config`, 6.2) | `npuw/llm_compiled_model.cpp` |
| NPUW `Const` eager-copy path / `WeightlessCacheAttribute` (6.3) | `npuw/lazy_tensor.cpp` (`Const::Const`), `core/src/op/util/weightless_caching_attributes.cpp` (`copy_weightless_cache_attr`, `is_copyable`) |
| Bench harness (gpu / npuw / xpu modes) | `bench_npuw_vs_xpu.py`; `xpu_hybrid_llm.py` (`--device`, `--verify-cpu`) |
| One-copy proof (Appendix C) | `prove_shared_buffer.py`; properties `XPU_WEIGHT_RANGES`, `XPU_VERIFY_GPU_L0` |
| POC (malloc dual-import validation) | `C:\yqiu\poc-npu-gpu-buffer-sharing` |

## B. Optimizations done

The prefill path went **first-token ~1333 ms (CPU bridge) -> ~700 (KV-write) -> ~420 (NPU-request reuse) ->
~230 ms (logits slice)** on Qwen3-4B (the same levers give Phi-4's 660 ms @14B). Decode went **150 -> ~36 ms/tok**
(Qwen) via the bank raw-share (3.1.3). The three prefill levers, in order of impact:

### B.1 KV-output binding moved to compile (small, but correct)
Binding the 80 `present.*` outputs to their shared-buffer slots was redone every prefill (rebuild remote tensors
+ `set_tensor`). Because the outputs are **static** the binding is conversation-invariant, so the remote tensors
are created **once at compile** (`build_kv_output_bindings`) and `set_tensor`'d once at request construction;
`infer_gpu_prefill` skips the loop. Per-prefill bind+input setup dropped ~10 ms -> ~0.4 ms.

### B.2 Prefill logits sliced to the last token (the big lever, ~200 ms)
The prefill only needs `logits[-1]` (the first generated token is `argmax(logits[-1])`); the other rows are never
read. But the GPU prefill model emitted **full-sequence** logits `[1, seq, vocab]` (~622 MB at 1K on Qwen), which
the OV async wrapper materialized (~175 ms) *and* the LM-head MatMul ran on all `seq` positions. **Fix** (port of
GenAI's `apply_slice_before_matmul`): insert `Slice(start=-1, stop=-2, step=-1, axis=seq)` on the LM-head MatMul's
activation input -> logits become `[1, 1, vocab]`. Implemented in `plugin.cpp`
(`xpu_slice_prefill_logits_to_last`); first-token ~430 -> ~230 ms (Qwen), decode unchanged, token-exact.

### B.3 NPU-request reuse (~220 ms) + the multi-turn dispatch fix
The NPU generate request was recreated every prefill (~220 ms `create_infer_request`). Now it is built once and
reused; each prefill only **signals** the prompt length via `XPU_EXTERNAL_PREFILL_LEN`, consumed at the top of the
NPU's next `infer()`. Stale KV is overwritten by the KV-write `ScatterUpdate(Broadcast(0,...))`. This fixed a
latent bug: a 2nd conversation's prefill on a reused request was misrouted to NPU decode (a `!prefill_done`
latch) and crashed - `seq_len>1` now always (re)prefills, so multi-turn reuse works (validated token-exact:
reused == fresh; `test_npu_reuse.py`).

### B.4 The partition-copy fix
See 6.3 - `copy_weightless_cache_attr` eliminated the ~7 GB (Phi-4) / ~2.1 GB (Qwen) eager host copy at
partition.

## C. Verifying the one-physical-copy property (`prove_shared_buffer.py`)

The central claim - GPU and NPU read **one** shared buffer, neither holds a private weight copy - is proved three
independent ways, using two read-only properties: **`XPU_WEIGHT_RANGES`** (per-segment `ptr:size`, so a host test
can poke a real weight) and **`XPU_VERIFY_GPU_L0`** (an L0 audit of every segment + the KV buffer in the GPU's L0
context).

1. **Memory accounting** - process committed ~= baseline + **one** weight copy (two copies would be ~7 GB more);
   `XPU_SHARED_WEIGHT_SIZE` (persistent residue) is sub-MB, not a 2nd copy.
2. **Isolated host-poke** (behavioral) - corrupt one FC weight in the shared malloc from the host (`ctypes`),
   timed to isolate each engine: poke **before** prefill -> the GPU-prefilled first token changes (GPU reads the
   shared buffer); poke **after** prefill -> first token + KV identical but **NPU decode changes** (the NPU read
   the poked weight directly - this timing rules out the prefill/KV confound). Restoring the bytes returns the
   exact baseline.
3. **GPU L0 identity** - for every weight segment **and** the KV buffer, the GPU's L0 context reports
   `base == host_ptr`, `type = HOST(_IMPORTED)`, **0** of type `DEVICE` - the GPU addresses the same imported host
   mallocs, no device-resident weight or KV copy. (The NPU side is covered behaviorally by (2).)

## D. DCOFF host-unpack path (legacy / KV-cache concept proof)

> **Not the target.** Keeps weights u4 in the shared buffer but the **NPU executes f16** by unpacking on the host
> CPU each inference - slower (~168 ms/tok on Qwen) and with a 4x device f16 expansion. Kept only because it was
> the first end-to-end working pipeline (it proved the KV-cache + weight sharing mechanics) and as a fallback
> where native INT4 cannot compile. Enable with `XPU_DCOFF=1` (+ typically `XPU_USM_WEIGHTS=1` on an OCL build).

- **Quantization (opposite of native)**: DCOFF needs **u4-asym, group-128** (`--weight-format int4 --group-size
  128`, no `--sym`). NPUW's DCOFF leaves each FC weight a plain u4 `Const` (shareable); i4-sym would trigger the
  `DQMatMulGQi` permute + a VCL lm_head crash. Layout: `u4 [out, groups, 128]` + `f16 [out, groups]` scale + u4 zp.
- **Execution**: `NPUW_DCOFF_TYPE=f16` + `NPUW_DCOFF_SCALE=YES` relabel the weight params to f16; the i4->f16
  dequant runs **on the host** (AVX2 `unpack_closure`) each inference, reading `(u4 - zp).scale` from the shared
  buffer into the NPU's L0 f16 inputs. Weight bytes stay u4 (zero-copy), but the **device executes f16** - a 4x
  expansion + per-token host unpack.
- **Buffer mode**: on an OCL GPU build the weight buffers are **OCL USM-host** (`clHostMemAllocINTEL`); the GPU
  shares via OCL `share_usm`, the NPU reads them **on the CPU**. This is why DCOFF works on the OCL build where
  native (which needs the NPU to import a neutral malloc, 1) cannot share with the GPU.

Native INT4 reuses all of this machinery (the KV bridge 3.2, per-weight buffers 3.1.1, bank raw-share 3.1.3) and
replaces only the NPU execution (host-unpack-f16 -> on-device i4) and the buffer mode (OCL-USM -> malloc + dual L0
import). Historical development is in `ARCHITECTURE_gpu_prefill_npu_decode.md` and `DESIGN_gpu_weight_sharing.md`.
