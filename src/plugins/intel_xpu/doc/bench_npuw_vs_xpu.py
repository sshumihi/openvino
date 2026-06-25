r"""Compare decode speed: native-int4 NPUW (standalone) vs XPU hybrid (GPU prefill + NPU decode),
at a 1K context. Run one mode per process to avoid OOM.

Usage:
    bench_npuw_vs_xpu.py npuw [prompt_len] [max_new] [niter] [-m model.xml]
    bench_npuw_vs_xpu.py xpu  [prompt_len] [max_new] [niter] [-m model.xml]
    (-m/--model defaults to the Qwen3-4B-int4-sym-cw export)

npuw  = read_model + compile_model(model,"NPU", <NPUW LLM native-int4 props>) -- pure NPUW, its own
        prefill + device-bank weights + NPU-internal KV. Forces int4-native (NPUW_DQ + dyn-quant).
xpu   = compile_model(xml,"XPU") -- GPU prefill (shared zero-copy weights) + NPU decode (shared KV).
Both: warm up once (first-inference JIT), then measure a fresh request's steady prefill + steady decode.
"""
import argparse, os, sys, time, ctypes, statistics as st
RELEASE = r"C:\yqiu\xpu-buffer-ov\openvino\bin\intel64\Release"
os.environ.setdefault("OPENVINO_LIB_PATHS", RELEASE)
os.add_dll_directory(RELEASE); sys.path.insert(0, os.path.join(RELEASE, "python"))
try: ctypes.CDLL(os.path.join(RELEASE, "OpenCL.dll"))
except OSError: pass
import numpy as np, openvino as ov

ap = argparse.ArgumentParser(description="Compare pure-GPU / pure-NPU / XPU-hybrid prefill/decode speed.")
ap.add_argument("mode", nargs="?", default="xpu",
                help="xpu (GPU prefill + NPU decode) | npuw (pure NPUW) | gpu (pure stock GPU)")
ap.add_argument("prompt_len", nargs="?", type=int, default=1024, help="prompt length in tokens")
ap.add_argument("max_new", nargs="?", type=int, default=64, help="tokens to generate")
ap.add_argument("niter", nargs="?", type=int, default=3, help="measured iterations (after warm-up)")
ap.add_argument("-m", "--model", default=r"E:\models\Qwen3-4B-int4-sym-cw-ov\openvino_model.xml",
                help="model .xml (default: Qwen3-4B-int4-sym-cw)")
ap.add_argument("--unique-prompt", action="store_true",
                help="use a FRESH RANDOM prompt each measured iteration (different token content every time). "
                     "Defeats any content-keyed prefix/prompt/KV cache: if prefill is unchanged vs the default "
                     "fixed prompt, no such cache is inflating the numbers.")
ap.add_argument("--warmup-new", type=int, default=4,
                help="decode tokens during warm-up (primes GPU JIT / NPU-init; default 4). Keep small when "
                     "max_new is large so warm-up does not replay the whole generation.")
args = ap.parse_args()
MODE, PROMPT_LEN, MAX_NEW, NITER, MODEL = args.mode, args.prompt_len, args.max_new, args.niter, args.model

# The XPU hybrid's GPU prefill must read the shared i4 weights with NO reorder (ref-compressed FC).
# The pure-GPU baseline must NOT set it, so the GPU uses its optimized blocked-INT4 kernel (representative
# of stock GPU). npuw is NPU-only, so the flag is irrelevant there.
if MODE == "xpu":
    os.environ.setdefault("OV_XPU_REF_COMPRESSED_FC", "1")

core = ov.Core()

def build():
    # Size the NPU's static KV cache to THIS run: total_size = align(max_prompt) + align(min_response),
    # so it must cover prompt_len + the generated tokens. For the XPU path the plugin forwards these to
    # its NPU sub-compile and the GPU KV-write rewrite picks up the resulting NPUW_KVCACHE_LAYOUT, so the
    # GPU's static present.* slots match. (Default 1024/128 overflows on long prompts -> GPU KV-write
    # ScatterUpdate writes past the slot and zeCommandListAppendLaunchKernel fails.)
    kv_size = {
        "NPUW_LLM_MAX_PROMPT_LEN": PROMPT_LEN,
        "NPUW_LLM_MIN_RESPONSE_LEN": max(MAX_NEW, 128),
    }
    if MODE == "gpu":
        # pure stock GPU: full stateful LLM, GPU-managed internal KV. No NPUW props, no shared buffer.
        return core.compile_model(MODEL, "GPU")
    if MODE == "xpu":
        return core.compile_model(MODEL, "XPU", kv_size)
    # native-int4 NPUW standalone (mirror the XPU plugin's NPU sub-compile, force int4-native)
    model = core.read_model(MODEL)
    props = {
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_DEVICES": "NPU",
        "NPU_COMPILER_TYPE": "PLUGIN",
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 2,
        "NPUW_DQ": "YES",
        "NPU_COMPILER_DYNAMIC_QUANTIZATION": "YES",
        **kv_size,
    }
    return core.compile_model(model, "NPU", props)

t0 = time.perf_counter()
compiled = build()
compile_ms = (time.perf_counter() - t0) * 1e3
has_beam = "beam_idx" in {p.get_any_name() for p in compiled.inputs}
print(f"=== MODE={MODE} prompt_len={PROMPT_LEN} max_new={MAX_NEW} compile={compile_ms:.0f}ms ===", flush=True)

ids = [(i % 50000) + 100 for i in range(PROMPT_LEN)]   # fixed synthetic prompt (default; same every iteration)
_rng = np.random.default_rng(0)                        # for --unique-prompt: a different prompt each iteration

def feed(toks, positions, attn_len):
    d = {"input_ids": np.array([toks], dtype=np.int64),
         "attention_mask": np.ones((1, attn_len), dtype=np.int64),
         "position_ids": np.array([positions], dtype=np.int64)}
    if has_beam: d["beam_idx"] = np.zeros((1,), dtype=np.int32)
    return d

def last(req):
    a = req.get_output_tensor(0).data
    return a[0, -1, :] if a.ndim == 3 else a[0]

def run(req, measure, prompt_ids, n_new):
    L = len(prompt_ids)
    t0 = time.perf_counter()
    req.infer(feed(prompt_ids, list(range(L)), L))
    pf = (time.perf_counter() - t0) * 1e3
    first = int(np.argmax(last(req)))
    per = []
    seq = list(prompt_ids) + [first]
    for _ in range(n_new):
        pos = len(seq) - 1
        t0 = time.perf_counter()
        req.infer(feed([seq[-1]], [pos], pos + 1))
        per.append((time.perf_counter() - t0) * 1e3)
        seq.append(int(np.argmax(last(req))))
    if measure:
        steady = per[6:] if len(per) > 10 else per
        # decode grows as the KV context grows (prompt L -> L+n_new); show the curve via first/last window.
        w = min(64, len(per))
        head, tail = st.mean(per[:w]), st.mean(per[-w:])
        print(f"  [{MODE}] prefill {pf:.0f} ms ({L} tok)   "
              f"decode steady {st.mean(steady):.1f} ms/tok (min {min(per):.1f}, median {st.median(per):.1f}, "
              f"max {max(per):.1f})   ctx-growth: first{w} {head:.1f} -> last{w} {tail:.1f} ms/tok   "
              f"over {len(per)} toks (ctx {L}->{L+len(per)})   first_tok={first}", flush=True)
    return pf, per

# warm up (first-inference JIT for GPU) on one request, then measure fresh requests. Each measured iteration
# uses its own create_infer_request() (fresh KV state). With --unique-prompt the prompt CONTENT also differs
# every iteration, so no content-keyed prefix/KV cache can be hit -> the prefill is a true cold-KV prefill.
print(f"  warming up ({args.warmup_new} decode toks)...", flush=True)
run(compiled.create_infer_request(), measure=False, prompt_ids=ids, n_new=min(args.warmup_new, MAX_NEW))
print(f"  measuring... (unique_prompt={args.unique_prompt})", flush=True)
pfs = []
for i in range(NITER):
    p_ids = _rng.integers(100, 50000, size=PROMPT_LEN).tolist() if args.unique_prompt else ids
    pf, per = run(compiled.create_infer_request(), measure=True, prompt_ids=p_ids, n_new=MAX_NEW)
    pfs.append(pf)
print(f"  [{MODE}] per-iter prefill ms: {[round(x) for x in pfs]}  "
      f"(mean {st.mean(pfs):.0f}, unique_prompt={args.unique_prompt})", flush=True)

del compiled
import gc; del core; gc.collect()
