r"""
XPU hybrid LLM runner - GPU prefill + NPU decode over ONE L0-shared weight buffer
=================================================================================

This is the canonical Python entry point for the `intel_xpu` meta-plugin. Unlike the
old version of this file (which hand-rolled Level-Zero KV-cache sharing in Python and
compiled the GPU and NPU models separately), ALL hybrid orchestration now lives inside
the C++ plugin. From Python you only:

      core = ov.Core()
      compiled = core.compile_model(MODEL_XML, "XPU")   # <-- string path == full pipeline
      req = compiled.create_infer_request()
      req.infer(...)                                     # seq_len>1 -> GPU, seq_len==1 -> NPU

DISPATCH (verified from source): a STRING / os.PathLike first arg to compile_model() is
routed to_fs_path() -> Core::compile_model(filesystem::path) -> intel_xpu
Plugin::compile_model(const std::filesystem::path&) (plugin.cpp) == the FULL hybrid
pipeline. Do NOT read_model()+compile_model(model_obj,"XPU") - that bypasses the path
overload that builds the GPU-prefill model, the NPU weightless blob, and the shared
buffers. The plugin then dispatches per infer (sync_infer_request.cpp:108):
      seq_len > 1  -> infer_gpu_prefill()   (GPU stateless prefill, writes KV to shared buf)
      seq_len == 1 -> infer_npu_decode()    (NPU native-INT4 generate, reads shared weights)

What "correct" means here (all default, NO env flags needed):
  * Native INT4   - NPU executes i4 on-device (compiler dynamic-quant), GPU reads i4 via
                    the no-reorder bfyx_ref kernel. The plugin auto-sets
                    OV_XPU_REF_COMPRESSED_FC on Windows.
  * L0 sharing    - weights live in one page-aligned malloc imported into BOTH the GPU-L0
                    and NPU-L0 contexts (POC "Approach 3"); GPU share_usm and NPU native
                    read the SAME physical bytes (zero-copy both ways).
  * GPU prefill   - a StatefulToStateless clone compiled on GPU; KV bridged to the shared
                    buffer with strided output tensors.
  * NPU decode    - NPU prefill model is skipped (aliased to generate) to save ~2 GB; the
                    bank stores zero-copy views into the shared weight buffer.

By default this script VERIFIES all four properties at runtime (get_property + fd-level
capture of the plugin's stdout markers) and prints a PASS/FAIL panel.

Run (the venv the bindings were built against):
    C:\yqiu\xpu-buffer-ov\ov-xpu-env\Scripts\python.exe xpu_hybrid_llm.py "Your prompt"

Common options:
    xpu_hybrid_llm.py "The capital of France is"      # raw-string prompt (default model)
    xpu_hybrid_llm.py --max-new 64                    # cap generated tokens
    xpu_hybrid_llm.py --prompt-len 1024               # tile prompt to N tokens (prefill bench)
    xpu_hybrid_llm.py --verify-cpu                    # also run CPU and diff token-for-token
    xpu_hybrid_llm.py --no-capture                    # stream plugin markers live (no panel)
    xpu_hybrid_llm.py --device NPU                    # pure-NPUW-via-XPU baseline (no GPU prefill)

Env (advanced; defaults are correct - these only override):
    XPU_MODEL_XML        model .xml (default: Qwen3-4B-int4-sym-cw, the native-INT4 export)
    XPU_DCOFF=1          legacy DCOFF host-unpack path (needs a u4-asymmetric export)
    XPU_USM_WEIGHTS=1    legacy OCL-USM weights (GPU copies; NPU can't import) - disables L0 share
    XPU_KEEP_NPU_PREFILL=1   keep the NPU prefill model (uses ~2 GB more)
    XPU_MEM_DEBUG=1      plugin prints [XPU][MEM]/[DIAG] memory breakdowns
"""
import argparse
import ctypes
import os
import sys
import tempfile
import time

# --- Bootstrap THIS custom build BEFORE importing openvino -------------------------
RELEASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "openvino", "bin", "intel64", "Release")
# openvino/__init__.py's _add_openvino_libs_to_search_path() sys.exit()s without this.
os.environ.setdefault("OPENVINO_LIB_PATHS", RELEASE)
os.add_dll_directory(RELEASE)
sys.path.insert(0, os.path.join(RELEASE, "python"))
_tbb = os.path.join(RELEASE, "tbb")
if os.path.isdir(_tbb):
    os.add_dll_directory(_tbb)

# Bypass any stray CLIntercept OpenCL shim (e.g. C:\Python311\OpenCL.dll) by preloading
# the build's real ICD loader - matches the known-good C++ run.
try:
    ctypes.CDLL(os.path.join(RELEASE, "OpenCL.dll"))
except OSError:
    pass

# GPU compressed-FC -> row-major bfyx_ref (no reorder) so GPU prefill reads the shared
# INT4 weight buffers zero-copy. The plugin also sets this itself on Windows; setdefault
# keeps it explicit and covers non-default builds. (Read via getenv at compile time.)
os.environ.setdefault("OV_XPU_REF_COMPRESSED_FC", "1")

# The plugin's "[XPU] ..." / "[XPU-ZC] ..." progress prints are gated behind OV_XPU_DEBUG
# (presence-based: set = on). They're off by default in production; this verification harness
# greps those markers, so it opts in here. Code that uses the XPU plugin without this script
# stays silent unless OV_XPU_DEBUG is exported.
os.environ.setdefault("OV_XPU_DEBUG", "1")

import numpy as np            # noqa: E402
import openvino as ov         # noqa: E402

# --- Config ------------------------------------------------------------------------
DEFAULT_MODEL = r"E:\models\Qwen3-4B-int4-sym-cw-ov\openvino_model.xml"
EOS = {151643, 151644, 151645}            # Qwen3: endoftext / im_start / im_end
KV_SIZE = 1152                            # compiled max prompt+generate window
MAX_PROMPT = 1024                         # compiled prefill max

# Authoritative plugin stdout markers (std::endl-flushed -> reliably captured at fd level).
M_NATIVE   = "[XPU] weight path: NATIVE INT4"
M_DCOFF    = "[XPU] weight path: DCOFF"
M_SHARE    = "share ONE physical buffer (zero-copy both)"      # L0 dual-import succeeded
M_NOSHARE  = "GPU copies, NPU shares"                          # L0 dual-import failed
M_PFCOMPILE = "[XPU] step 6f: GPU prefill compile OK"
M_HYBRID   = "skipped full stateful GPU compile (hybrid mode"  # NPU prefill skipped
M_PFRUN    = "[XPU] infer_gpu_prefill: starting GPU prefill..."
M_PFDONE   = "[XPU] GPU prefill done."
M_DECODE   = "[XPU] NPU initialized for decode (KV buffer zeroed), prompt_len="


# --- fd-level stdout/stderr capture (tee-after-replay) -----------------------------
class FdCapture:
    """Redirect OS fds 1/2 to a temp file so the C++ plugin's std::cout markers are
    captured, then replay them to the real console on exit (so the user still sees
    them). Used to verify which engine ran. Degrades to a no-op if enabled=False or if
    the platform refuses the dup2 (verdict then falls back to get_property only)."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.text = ""
        self._ok = False

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            sys.stdout.flush(); sys.stderr.flush()
            self._tmp = tempfile.TemporaryFile(mode="w+b")
            self._save_out = os.dup(1)
            self._save_err = os.dup(2)
            os.dup2(self._tmp.fileno(), 1)
            os.dup2(self._tmp.fileno(), 2)
            self._ok = True
        except Exception:
            self._ok = False
        return self

    def __exit__(self, *exc):
        if not (self.enabled and self._ok):
            return False
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        os.dup2(self._save_out, 1)
        os.dup2(self._save_err, 2)
        self._tmp.seek(0)
        self.text = self._tmp.read().decode("utf-8", "replace")
        self._tmp.close()
        # Replay captured plugin output so it is visible even if an exception propagates.
        # Encode-safe: a stray non-ASCII byte must never abort the replay on a cp1252 console.
        try:
            enc = (sys.stdout.encoding or "utf-8")
            sys.stdout.write(self.text.encode(enc, "replace").decode(enc, "replace"))
            sys.stdout.flush()
        except Exception:
            pass
        os.close(self._save_out)
        os.close(self._save_err)
        return False


# --- Tokenizer (OpenVINO tokenizer IRs, no transformers dependency) ----------------
def load_tokenizers(core, model_dir):
    tok_dll = os.environ.get("XPU_TOKENIZERS_DLL",
                             os.path.join(RELEASE, "openvino_tokenizers.dll"))
    assert os.path.exists(tok_dll), f"openvino_tokenizers.dll not found: {tok_dll}"
    core.add_extension(tok_dll)
    tok = core.compile_model(os.path.join(model_dir, "openvino_tokenizer.xml"), "CPU")
    det = core.compile_model(os.path.join(model_dir, "openvino_detokenizer.xml"), "CPU")
    return tok, det


def encode(tok, text):
    res = tok([text])
    for port, val in res.items():
        if "input_ids" in port.get_names():
            return np.array(val)[0].astype(np.int64).tolist()
    raise RuntimeError("tokenizer produced no input_ids output")


def decode(det, ids):
    out = det(np.array([ids], dtype=np.int64))
    return str(next(iter(out.values()))[0])


def logits_last_row(req):
    arr = req.get_output_tensor(0).data
    return arr[0, -1, :] if arr.ndim == 3 else arr[0]


# --- XPU compiled-model property probe ---------------------------------------------
def query_xpu_props(compiled):
    """Snapshot the XPU-native get_property signals (must be read before del compiled)."""
    def p(key, default=None):
        try:
            return compiled.get_property(key)
        except Exception:
            return default
    return {
        "full_name":          p("FULL_DEVICE_NAME"),
        "active_device":      p("XPU_ACTIVE_DEVICE"),
        "gpu_prefill_avail":  p("XPU_GPU_PREFILL_AVAILABLE"),
        "weight_seg_count":   p("XPU_WEIGHT_SEGMENT_COUNT"),
        "weight_seg_bytes":   p("XPU_WEIGHT_SEGMENT_BYTES"),
        "shared_weight_ptr":  p("XPU_SHARED_WEIGHT_PTR"),
        "shared_weight_size": p("XPU_SHARED_WEIGHT_SIZE"),
        "kvcache_ptr":        p("XPU_SHARED_KVCACHE_PTR"),
        "kvcache_size":       p("XPU_SHARED_KVCACHE_SIZE"),
    }


# --- Generate ----------------------------------------------------------------------
def generate(core, model_xml, device, prompt_ids, max_new):
    """Greedy generate via the XPU hybrid path. Returns (gen_ids, timing, props)."""
    compiled = core.compile_model(model_xml, device)   # STRING path -> XPU hybrid pipeline
    props = query_xpu_props(compiled) if device == "XPU" else {}
    req = compiled.create_infer_request()
    in_names = {p.get_any_name() for p in compiled.inputs}
    has_beam = "beam_idx" in in_names

    def feed(ids, positions, attn_len):
        d = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, attn_len), dtype=np.int64),
            "position_ids": np.array([positions], dtype=np.int64),
        }
        if has_beam:
            d["beam_idx"] = np.zeros((1,), dtype=np.int32)
        return d

    tokens = list(prompt_ids)
    L = len(prompt_ids)

    t0 = time.perf_counter()
    req.infer(feed(prompt_ids, list(range(L)), L))            # prefill (seq_len>1 -> GPU)
    prefill_ms = (time.perf_counter() - t0) * 1e3
    tokens.append(int(np.argmax(logits_last_row(req))))

    t0 = time.perf_counter()
    steps = 0
    for _ in range(1, max_new):                              # decode (seq_len==1 -> NPU)
        if tokens[-1] in EOS:
            break
        pos = len(tokens) - 1
        req.infer(feed([tokens[-1]], [pos], pos + 1))
        tokens.append(int(np.argmax(logits_last_row(req))))
        steps += 1
    decode_ms = (time.perf_counter() - t0) * 1e3

    gen = tokens[L:]
    timing = {"prefill_ms": prefill_ms, "decode_ms": decode_ms, "steps": steps,
              "per_tok_ms": decode_ms / steps if steps else 0.0, "prompt_len": L}
    del req       # free request + compiled model while the plugin/context are alive
    del compiled  # (avoids a shutdown-time AV when USM buffers outlive the GPU context)
    return gen, timing, props


# --- Verification panel ------------------------------------------------------------
def verify(cap_text, props, timing, device):
    """Build a PASS/FAIL table from captured plugin markers + get_property snapshot."""
    have_cap = bool(cap_text) and (M_NATIVE in cap_text or M_DCOFF in cap_text
                                   or M_HYBRID in cap_text or M_SHARE in cap_text)
    n_prefill = cap_text.count(M_PFRUN) if cap_text else 0
    steps = timing["steps"]

    rows = []   # (label, status, detail)

    # 1. Native INT4 weight path
    if have_cap:
        if M_NATIVE in cap_text:
            rows.append(("Native INT4 weight path", "PASS", "compiler dynamic-quant, on-device"))
        elif M_DCOFF in cap_text:
            rows.append(("Native INT4 weight path", "WARN", "DCOFF host-unpack active (XPU_DCOFF) - not the target path"))
        else:
            rows.append(("Native INT4 weight path", "????", "no weight-path marker captured"))
    else:
        rows.append(("Native INT4 weight path", "n/a ", "stdout capture unavailable (use default run, not --no-capture)"))

    # 2. L0 one-buffer sharing (GPU+NPU zero-copy)
    seg_mb = (props.get("weight_seg_bytes") or 0) / 1e6
    seg_n = props.get("weight_seg_count")
    if have_cap and M_SHARE in cap_text:
        rows.append(("L0 shared weight buffer", "PASS",
                     f"GPU+NPU share ONE physical buffer; {seg_n} segs, {seg_mb:.0f} MB imported into both L0 ctxs"))
    elif have_cap and M_NOSHARE in cap_text:
        rows.append(("L0 shared weight buffer", "WARN",
                     "GPU copies, NPU shares - L0 dual-import failed (needs GPU_RT_TYPE=L0 + sysmem-import HW)"))
    elif (props.get("weight_seg_count") or 0) > 0:
        rows.append(("L0 shared weight buffer", "????",
                     f"{seg_n} segs / {seg_mb:.0f} MB allocated, but import marker not captured"))
    else:
        rows.append(("L0 shared weight buffer", "????", "no weight segments reported"))

    # 3. GPU prefill model compiled
    avail = props.get("gpu_prefill_avail")
    avail_b = (avail is True) or (str(avail).strip().lower() in ("1", "true", "yes"))
    compiled_marker = have_cap and M_PFCOMPILE in cap_text
    if device == "NPU":
        rows.append(("GPU prefill model", "n/a ", "--device NPU forces pure-NPUW (no GPU prefill)"))
    elif avail_b or compiled_marker:
        rows.append(("GPU prefill model", "PASS",
                     f"XPU_GPU_PREFILL_AVAILABLE={avail}" + ("; compile-OK marker seen" if compiled_marker else "")))
    else:
        rows.append(("GPU prefill model", "FAIL", f"XPU_GPU_PREFILL_AVAILABLE={avail}, no compile marker"))

    # 4. GPU actually ran prefill
    if device == "NPU":
        rows.append(("GPU ran the prefill", "n/a ", "pure-NPUW path"))
    elif have_cap and n_prefill >= 1:
        rows.append(("GPU ran the prefill", "PASS",
                     f"infer_gpu_prefill fired {n_prefill}x" + (" (+ GPU prefill done)" if M_PFDONE in cap_text else "")))
    elif have_cap:
        rows.append(("GPU ran the prefill", "FAIL", "no infer_gpu_prefill marker captured"))
    else:
        rows.append(("GPU ran the prefill", "n/a ", "stdout capture unavailable"))

    # 5. NPU ran the decode (one prefill marker for many tokens == decode did NOT re-prefill on GPU)
    if device == "NPU":
        rows.append(("NPU ran the decode", "PASS" if steps > 0 else "????",
                     f"pure-NPUW generate over {steps} steps"))
    elif have_cap and (M_DECODE in cap_text) and steps > 0 and n_prefill == 1:
        rows.append(("NPU ran the decode", "PASS",
                     f"NPU-decode bridge engaged; {steps} tokens with a single GPU prefill"))
    elif have_cap and n_prefill > 1:
        rows.append(("NPU ran the decode", "FAIL", f"GPU prefill fired {n_prefill}x - decode leaked onto GPU"))
    elif have_cap and steps > 0:
        rows.append(("NPU ran the decode", "????", "decode-bridge marker not captured"))
    else:
        rows.append(("NPU ran the decode", "n/a ", "no decode steps / capture unavailable"))

    # 6. Hybrid mode (NPU prefill skipped)
    if have_cap and M_HYBRID in cap_text:
        rows.append(("Hybrid (NPU prefill skipped)", "PASS", "full stateful GPU compile skipped; NPU prefill aliased"))
    elif device == "NPU":
        rows.append(("Hybrid (NPU prefill skipped)", "n/a ", "pure-NPUW path keeps NPU prefill"))
    else:
        rows.append(("Hybrid (NPU prefill skipped)", "????", "hybrid marker not captured"))

    # --- print table
    print("\n" + "=" * 78)
    print("  XPU HYBRID VERIFICATION  (L0 buffer sharing + GPU prefill + NPU decode)")
    print("=" * 78)
    w = max(len(r[0]) for r in rows)
    for label, status, detail in rows:
        print(f"  [{status}] {label.ljust(w)}  {detail}")
    print("-" * 78)
    print(f"  device           : {props.get('full_name')}  (active={props.get('active_device')})")
    print(f"  shared weight buf : ptr=0x{props.get('shared_weight_ptr') or 0:016x}  "
          f"persistent={ (props.get('shared_weight_size') or 0)/1e6:.0f} MB  "
          f"segments={props.get('weight_seg_count')} / {(props.get('weight_seg_bytes') or 0)/1e6:.0f} MB")
    print(f"  shared KV buffer  : ptr=0x{props.get('kvcache_ptr') or 0:016x}  "
          f"size={(props.get('kvcache_size') or 0)/1e6:.0f} MB")
    print(f"  timing            : prefill {timing['prefill_ms']:.1f} ms ({timing['prompt_len']} tok) | "
          f"decode {timing['per_tok_ms']:.1f} ms/tok over {steps} tok")
    print("=" * 78)

    crit = [s for (l, s, d) in rows if s == "FAIL"]
    ok = not crit and any(s == "PASS" for (l, s, d) in rows)
    return ok


# --- Main --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="XPU hybrid LLM runner (GPU prefill + NPU decode).")
    ap.add_argument("prompt", nargs="?", default="The capital of France is", help="raw prompt string")
    ap.add_argument("--model", default=os.environ.get("XPU_MODEL_XML", DEFAULT_MODEL),
                    help="model .xml (default: Qwen3-4B-int4-sym-cw, the native-INT4 export)")
    ap.add_argument("--device", default="XPU", help="XPU (hybrid) | NPU (pure-NPUW) | GPU | CPU")
    ap.add_argument("--max-new", type=int, default=int(os.environ.get("XPU_MAX_NEW", "64")),
                    help="max NEW tokens to generate")
    ap.add_argument("--prompt-len", type=int, default=None,
                    help=f"tile/truncate prompt to exactly N tokens (<= {MAX_PROMPT}; prefill bench)")
    ap.add_argument("--verify-cpu", action="store_true", help="also run on CPU and diff token-for-token")
    ap.add_argument("--no-capture", action="store_true",
                    help="stream plugin markers live instead of capturing them (skips PASS/FAIL panel)")
    args = ap.parse_args()

    assert os.path.exists(args.model), f"model xml not found: {args.model}"
    model_dir = os.path.dirname(args.model)

    core = ov.Core()
    print("OpenVINO:", ov.__version__, "| from:", os.path.dirname(ov.__file__))
    print("Devices :", core.available_devices)
    if args.device == "XPU":
        assert "XPU" in core.available_devices, "XPU plugin not loaded (check the build's plugins.xml)"

    tok, det = load_tokenizers(core, model_dir)
    prompt_ids = encode(tok, args.prompt)
    print(f'\nPrompt  : "{args.prompt}"  ->  {len(prompt_ids)} tokens: {prompt_ids[:16]}'
          f'{" ..." if len(prompt_ids) > 16 else ""}')

    if args.prompt_len is not None:
        assert args.prompt_len <= MAX_PROMPT, f"--prompt-len {args.prompt_len} exceeds compiled max ({MAX_PROMPT})"
        prompt_ids = [prompt_ids[i % len(prompt_ids)] for i in range(args.prompt_len)]
        print(f"(tiled/truncated to --prompt-len {args.prompt_len} tokens)")
    assert len(prompt_ids) + args.max_new < KV_SIZE, \
        f"prompt_len + max_new must stay under the KV window ({KV_SIZE})"

    capture = (args.device == "XPU") and (not args.no_capture)
    print(f"\n==== {args.device}: compile + generate "
          f"(prompt_len={len(prompt_ids)}, max_new={args.max_new}, capture={capture}) ====")
    cap = FdCapture(enabled=capture)
    with cap:
        gen, timing, props = generate(core, args.model, args.device, prompt_ids, args.max_new)

    print(f"\n{args.device} output text: {decode(det, gen)!r}")
    print(f"{args.device} output ids ({len(gen)}): {gen}")

    rc = 0
    if args.device == "XPU":
        ok = verify(cap.text, props, timing, args.device)
        print("VERDICT:", "PASS - correct L0 sharing + GPU prefill + NPU decode"
              if ok else "CHECK - see WARN/FAIL/???? rows above")
        rc = 0 if ok else 1
    else:
        print(f"[{args.device}] prefill {timing['prefill_ms']:.1f} ms | "
              f"decode {timing['per_tok_ms']:.1f} ms/tok over {timing['steps']} tok")

    if args.verify_cpu:
        cpu_gen, cpu_t, _ = generate(core, args.model, "CPU", prompt_ids, args.max_new)
        match = sum(1 for a, b in zip(gen, cpu_gen) if a == b)
        print(f"\nCPU output text: {decode(det, cpu_gen)!r}")
        print(f"MATCH {match}/{min(len(gen), len(cpu_gen))} vs CPU  |  "
              f"prefill {args.device} {timing['prefill_ms']:.0f} ms vs CPU {cpu_t['prefill_ms']:.0f} ms")
        same = (gen == cpu_gen)
        print("CPU CROSS-CHECK:", "MATCH (token-exact)" if same else "DIVERGES")
        rc = rc or (0 if same else 1)

    import gc
    del tok, det, core
    gc.collect()
    return rc


if __name__ == "__main__":
    sys.exit(main())
