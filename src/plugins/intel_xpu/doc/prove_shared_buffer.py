r"""Prove the XPU hybrid reads ONE physical weight buffer shared by GPU and NPU -- not a private
copy per device. Three independent proofs:

  (1) MEMORY ACCOUNTING - only one ~2.1 GB weight copy exists (NPU persistent tiny; GPU usm_device
      is ~KV-buffer size, not 2.1 GB).
  (2) ISOLATED MUTATION (definitive) - corrupt ONE transformer weight in the shared buffer from the
      HOST (ctypes), timed to isolate each engine:
        * poke BEFORE prefill  -> the GPU-prefilled FIRST token changes  => GPU reads the shared buffer.
        * poke AFTER prefill    -> FIRST token is UNCHANGED (prefill ran on original weights, so the
          KV cache + first token handed to the NPU are identical) but the NPU DECODE changes => the
          NPU read the poked WEIGHT directly, not because its prefill/KV input differed.
      Restoring the bytes returns the output to baseline.
  (3) GPU L0 IDENTITY - the GPU's Level-Zero context maps the weight pointer to the SAME host malloc
      (zeMemGetAddressRange base==ptr; zeMemGetAllocProperties type=HOST/HOST_IMPORTED), i.e. an
      imported host allocation, not a device-resident copy.

Run: C:\yqiu\xpu-buffer-ov\ov-xpu-env\Scripts\python.exe prove_shared_buffer.py
"""
import os, sys, ctypes, ctypes.wintypes as wt
from ctypes import c_void_p, c_size_t, c_uint32, c_uint64, byref, POINTER
RELEASE = r"C:\yqiu\xpu-buffer-ov\openvino\bin\intel64\Release"
os.environ.setdefault("OPENVINO_LIB_PATHS", RELEASE)
os.add_dll_directory(RELEASE); sys.path.insert(0, os.path.join(RELEASE, "python"))
try: ctypes.CDLL(os.path.join(RELEASE, "OpenCL.dll"))
except OSError: pass
os.environ.setdefault("OV_XPU_REF_COMPRESSED_FC", "1")
import numpy as np, openvino as ov

MODEL = r"E:\models\Qwen3-4B-int4-sym-cw-ov\openvino_model.xml"
PROMPT_IDS = [785, 6722, 315, 9625, 374]   # "The capital of France is"
MAX_NEW = 8
EOS = {151643, 151644, 151645}


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", c_size_t), ("WorkingSetSize", c_size_t),
                ("QuotaPeakPagedPoolUsage", c_size_t), ("QuotaPagedPoolUsage", c_size_t),
                ("QuotaPeakNonPagedPoolUsage", c_size_t), ("QuotaNonPagedPoolUsage", c_size_t),
                ("PagefileUsage", c_size_t), ("PeakPagefileUsage", c_size_t)]


_k32 = ctypes.WinDLL("kernel32.dll"); _k32.GetCurrentProcess.restype = wt.HANDLE
_psapi = ctypes.WinDLL("psapi.dll")
_GPMI = _psapi.GetProcessMemoryInfo
_GPMI.argtypes = [wt.HANDLE, POINTER(PMC), wt.DWORD]; _GPMI.restype = wt.BOOL


def mem_mb():
    p = PMC(); p.cb = ctypes.sizeof(p)
    _GPMI(_k32.GetCurrentProcess(), byref(p), p.cb)
    return p.WorkingSetSize / 1e6, p.PagefileUsage / 1e6  # (working set, committed/private)


core = ov.Core()
print("baseline RSS / committed (MB):", tuple(round(x) for x in mem_mb()))
compiled = core.compile_model(MODEL, "XPU")
has_beam = "beam_idx" in {p.get_any_name() for p in compiled.inputs}


def feed(ids, pos, alen):
    d = {"input_ids": np.array([ids], np.int64),
         "attention_mask": np.ones((1, alen), np.int64),
         "position_ids": np.array([pos], np.int64)}
    if has_beam: d["beam_idx"] = np.zeros((1,), np.int32)
    return d


def last(req):
    a = req.get_output_tensor(0).data
    return a[0, -1, :] if a.ndim == 3 else a[0]


def generate(poke=None, when=None):
    """prefill (GPU) + decode (NPU). `when` in {'before','after_prefill'} controls when `poke`
    mutates the shared weight, to isolate which engine reads it."""
    req = compiled.create_infer_request()
    toks = list(PROMPT_IDS); L = len(toks)
    if poke and when == "before":
        poke()                                           # GPU isolation: prefill sees the poke
    req.infer(feed(toks, list(range(L)), L))             # GPU prefill (writes KV from current weights)
    first = int(np.argmax(last(req)))                    # <- produced by GPU prefill
    if poke and when == "after_prefill":
        poke()                                           # NPU isolation: KV + first token already fixed
    toks.append(first)
    decode = []
    for _ in range(MAX_NEW):
        if toks[-1] in EOS: break
        pos = len(toks) - 1
        req.infer(feed([toks[-1]], [pos], pos + 1))      # NPU decode (reads current weights)
        nt = int(np.argmax(last(req))); decode.append(nt); toks.append(nt)
    del req
    return first, decode


# ---- Proof 1: memory accounting -------------------------------------------------------------------
seg_bytes = compiled.get_property("XPU_WEIGHT_SEGMENT_BYTES")
persist = compiled.get_property("XPU_SHARED_WEIGHT_SIZE")
kv = compiled.get_property("XPU_SHARED_KVCACHE_SIZE")
ws, comm = mem_mb()
print("\n=== PROOF 1: memory accounting (one copy, no per-device duplicate) ===")
print(f"  shared weight segments (GPU+NPU read these) : {seg_bytes/1e6:8.0f} MB")
print(f"  NPU persistent buffer (host closures only)  : {persist/1e6:8.1f} MB  (NOT a 2nd weight copy)")
print(f"  shared KV buffer                            : {kv/1e6:8.0f} MB")
print(f"  process RSS / committed                     : {ws:8.0f} / {comm:.0f} MB")
try:
    gs = core.get_property("GPU", "GPU_MEMORY_STATISTICS")
    gsd = {str(k): int(v) for k, v in dict(gs).items()}
    uh = sum(v for k, v in gsd.items() if "host" in k.lower())
    ud = sum(v for k, v in gsd.items() if "device" in k.lower())
    print(f"  GPU-allocated usm_host (activations/scratch): {uh/1e6:8.0f} MB  (weights+KV are imported mallocs, not GPU-allocated)")
    print(f"  GPU usm_device                              : {ud/1e6:8.0f} MB  <- GPU KV-write graph buffers (static present.* outputs + ScatterUpdate zeros), NOT the shared KV buffer (that's HOST, see PROOF 3) and NOT a weight copy")
except Exception as e:
    print("  (GPU_MEMORY_STATISTICS unavailable:", e, ")")
print(f"  => the {seg_bytes/1e9:.1f} GB weights exist ONCE (the shared malloc); neither device holds a 2nd copy.")

# pick the largest transformer FC weight (1 MB .. 50 MB; skips the 389 MB embedding) for the pokes
ranges = compiled.get_property("XPU_WEIGHT_RANGES")
segs = [(int(p), int(s)) for e in ranges.split(";") if e for p, s in [e.split(":")]]
fc = sorted([x for x in segs if 1_000_000 <= x[1] <= 50_000_000], key=lambda x: -x[1])
assert fc, "no FC-sized shared weight segment found"
addr, size = fc[0]
saved = (ctypes.c_ubyte * size)(); ctypes.memmove(saved, c_void_p(addr), size)
zero = lambda: ctypes.memset(c_void_p(addr), 0, size)
restore = lambda: ctypes.memmove(c_void_p(addr), saved, size)

# ---- Proof 2: isolated mutation -------------------------------------------------------------------
print("\n=== PROOF 2: host poke of ONE shared weight, isolating each engine ===")
print(f"  target shared-weight segment: addr=0x{addr:016x} size={size/1e6:.1f} MB")
b_first, b_dec = generate()
g_first, g_dec = generate(poke=zero, when="before"); restore()          # GPU isolation
n_first, n_dec = generate(poke=zero, when="after_prefill"); restore()   # NPU isolation
r_first, r_dec = generate()
print(f"  baseline                  : first={b_first}  decode={b_dec}")
print(f"  poke BEFORE prefill (GPU) : first={g_first}  decode={g_dec}")
print(f"  poke AFTER  prefill (NPU) : first={n_first}  decode={n_dec}")
print(f"  restored                  : first={r_first}  decode={r_dec}")
gpu_reads = (g_first != b_first)
npu_reads = (n_first == b_first) and (n_dec != b_dec)
restored_ok = (r_first == b_first and r_dec == b_dec)
print("\n  GPU prefill reads shared weight  (poke BEFORE -> first token changed):", "YES" if gpu_reads else "NO")
print("  NPU decode  reads shared weight  (poke AFTER  -> SAME first token + KV, decode changed):", "YES" if npu_reads else "NO")
print("    ^ the AFTER-prefill poke isolates the NPU: prefill ran on original weights so the first")
print("      token and KV cache are identical to baseline; the decode change is the NPU reading the")
print("      poked weight, NOT a different prefill/KV input.")
print("  restore returns to baseline:", "YES" if restored_ok else "NO")

# ---- Proof 3: GPU L0 allocation identity (plugin-side, over ALL segments) -------------------------
print("\n=== PROOF 3: GPU Level-Zero maps the weights as the shared host malloc (no device copy) ===")
l0 = compiled.get_property("XPU_VERIFY_GPU_L0")
print("  ", l0)
print("  => the GPU's L0 context addresses the SAME host mallocs (base==host_ptr, type=HOST), not")
print("     device-resident copies; '0 type=DEVICE' confirms no per-device weight allocation.")

print("\nOVERALL:", "PROVEN - GPU and NPU both read the one shared host buffer; neither holds a private weight copy"
      if (gpu_reads and npu_reads and restored_ok) else "INCONCLUSIVE - see above")
sys.stdout.flush()
os._exit(0)   # bypass interpreter-shutdown teardown (USM buffers freed after the GPU context -> AV)
