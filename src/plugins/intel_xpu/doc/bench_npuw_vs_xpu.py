r"""Compare decode speed: native-int4 NPUW (standalone) vs XPU hybrid (GPU prefill + NPU decode),
at a 1K context. Run one mode per process to avoid OOM.

Usage:
    bench_npuw_vs_xpu.py npuw [prompt_len] [max_new]
    bench_npuw_vs_xpu.py xpu  [prompt_len] [max_new]

npuw  = read_model + compile_model(model,"NPU", <NPUW LLM native-int4 props>) -- pure NPUW, its own
        prefill + device-bank weights + NPU-internal KV. Forces int4-native (NPUW_DQ + dyn-quant).
xpu   = compile_model(xml,"XPU") -- GPU prefill (shared zero-copy weights) + NPU decode (shared KV).
Both: warm up once (first-inference JIT), then measure a fresh request's steady prefill + steady decode.
"""
import os, sys, time, ctypes, statistics as st
RELEASE = r"C:\yqiu\xpu-buffer-ov\openvino\bin\intel64\Release"
os.environ.setdefault("OPENVINO_LIB_PATHS", RELEASE)
os.add_dll_directory(RELEASE); sys.path.insert(0, os.path.join(RELEASE, "python"))
try: ctypes.CDLL(os.path.join(RELEASE, "OpenCL.dll"))
except OSError: pass
os.environ.setdefault("OV_XPU_REF_COMPRESSED_FC", "1")
import numpy as np, openvino as ov

MODE = sys.argv[1] if len(sys.argv) > 1 else "xpu"
PROMPT_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
MAX_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 64
MODEL = r"E:\models\Qwen3-4B-int4-sym-cw-ov\openvino_model.xml"
NITER = int(sys.argv[4]) if len(sys.argv) > 4 else 3

core = ov.Core()

def build():
    if MODE == "xpu":
        return core.compile_model(MODEL, "XPU")
    # native-int4 NPUW standalone (mirror the XPU plugin's NPU sub-compile, force int4-native)
    model = core.read_model(MODEL)
    props = {
        "NPU_USE_NPUW": "YES",
        "NPUW_LLM": "YES",
        "NPUW_DEVICES": "NPU",
        "NPU_COMPILER_TYPE": "PLUGIN",
        "NPUW_LLM_BATCH_DIM": 0,
        "NPUW_LLM_SEQ_LEN_DIM": 2,
        "NPUW_LLM_MAX_PROMPT_LEN": 1024,
        "NPUW_LLM_MIN_RESPONSE_LEN": 128,
        "NPUW_DQ": "YES",
        "NPU_COMPILER_DYNAMIC_QUANTIZATION": "YES",
    }
    return core.compile_model(model, "NPU", props)

t0 = time.perf_counter()
compiled = build()
compile_ms = (time.perf_counter() - t0) * 1e3
has_beam = "beam_idx" in {p.get_any_name() for p in compiled.inputs}
print(f"=== MODE={MODE} prompt_len={PROMPT_LEN} max_new={MAX_NEW} compile={compile_ms:.0f}ms ===", flush=True)

ids = [(i % 50000) + 100 for i in range(PROMPT_LEN)]

def feed(toks, positions, attn_len):
    d = {"input_ids": np.array([toks], dtype=np.int64),
         "attention_mask": np.ones((1, attn_len), dtype=np.int64),
         "position_ids": np.array([positions], dtype=np.int64)}
    if has_beam: d["beam_idx"] = np.zeros((1,), dtype=np.int32)
    return d

def last(req):
    a = req.get_output_tensor(0).data
    return a[0, -1, :] if a.ndim == 3 else a[0]

def run(req, measure):
    L = len(ids)
    t0 = time.perf_counter()
    req.infer(feed(ids, list(range(L)), L))
    pf = (time.perf_counter() - t0) * 1e3
    nxt = int(np.argmax(last(req)))
    per = []
    seq = list(ids) + [nxt]
    for _ in range(MAX_NEW):
        pos = len(seq) - 1
        t0 = time.perf_counter()
        req.infer(feed([seq[-1]], [pos], pos + 1))
        per.append((time.perf_counter() - t0) * 1e3)
        seq.append(int(np.argmax(last(req))))
    if measure:
        steady = per[6:] if len(per) > 10 else per
        print(f"  [{MODE}] prefill {pf:.0f} ms ({L} tok)   "
              f"decode steady {st.mean(steady):.1f} ms/tok (min {min(per):.1f}, median {st.median(per):.1f}) "
              f"over {len(per)} toks", flush=True)
    return per

# warm up (first-inference JIT for GPU; primes caches) on one request, then measure a fresh one.
print("  warming up...", flush=True)
run(compiled.create_infer_request(), measure=False)
print("  measuring...", flush=True)
for i in range(NITER):
    run(compiled.create_infer_request(), measure=True)

del compiled
import gc; del core; gc.collect()
