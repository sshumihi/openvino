r"""Validate NPU-request reuse across conversations (the create-once optimization).

Two things must hold:
  (1) TOKEN-EXACT vs recreate: reused-request output == fresh-request output for the same prompt.
  (2) MULTI-CONVERSATION: running prompt B on a request that already ran prompt A (different length,
      stale KV in the shared buffer) must equal running B on a brand-new request. This is the
      stale-KV / generate-variant-switch concern the reuse path must not break.

NPU-request reuse is unconditional in the XPU plugin; this test guards that it stays token-exact.
"""
import os, sys
RELEASE = r"C:\yqiu\xpu-buffer-ov\openvino\bin\intel64\Release"
os.environ.setdefault("OPENVINO_LIB_PATHS", RELEASE)
os.add_dll_directory(RELEASE); sys.path.insert(0, os.path.join(RELEASE, "python"))
try:
    import ctypes; ctypes.CDLL(os.path.join(RELEASE, "OpenCL.dll"))
except OSError: pass
os.environ.setdefault("OV_XPU_REF_COMPRESSED_FC", "1")
import numpy as np, openvino as ov

MODEL = r"E:\models\Qwen3-4B-int4-sym-cw-ov\openvino_model.xml"
# Two prompts of DIFFERENT length so conversation B selects a different generate variant than A.
PROMPT_A = [785, 6722, 315, 9625, 374]                       # "The capital of France is"
PROMPT_B = [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722]  # longer
MAX_NEW = 12
EOS = {151643, 151644, 151645}

core = ov.Core()
compiled = core.compile_model(MODEL, "XPU")
has_beam = "beam_idx" in {p.get_any_name() for p in compiled.inputs}


def feed(ids, pos, alen):
    d = {"input_ids": np.array([ids], np.int64),
         "attention_mask": np.ones((1, alen), np.int64),
         "position_ids": np.array([pos], np.int64)}
    if has_beam:
        d["beam_idx"] = np.zeros((1,), np.int32)
    return d


def last(req):
    a = req.get_output_tensor(0).data
    return a[0, -1, :] if a.ndim == 3 else a[0]


def run_conversation(req, prompt):
    """One prefill + decode loop on the given (possibly reused) request."""
    toks = list(prompt); L = len(toks)
    req.infer(feed(toks, list(range(L)), L))          # GPU prefill (signals external prefill_len)
    out = [int(np.argmax(last(req)))]                  # first token
    toks.append(out[0])
    for _ in range(MAX_NEW):
        if toks[-1] in EOS:
            break
        pos = len(toks) - 1
        req.infer(feed([toks[-1]], [pos], pos + 1))    # NPU decode (consumes the signal on the 1st one)
        nt = int(np.argmax(last(req))); out.append(nt); toks.append(nt)
    return out


print("=== NPU-request reuse validation (reuse is unconditional) ===\n")

# Baselines: a FRESH request per conversation (each pays its own create_infer_request).
base_A = run_conversation(compiled.create_infer_request(), PROMPT_A)
base_B = run_conversation(compiled.create_infer_request(), PROMPT_B)
print(f"  fresh-request  A: {base_A}")
print(f"  fresh-request  B: {base_B}")

# Reuse: ONE request runs A then B (B sees A's leftover KV + a different generate variant).
shared = compiled.create_infer_request()
print("  [.] created shared request; running conversation A on it...", flush=True)
reuse_A = run_conversation(shared, PROMPT_A)
print(f"  [.] shared A done: {reuse_A}", flush=True)
print("  [.] running conversation B (2nd prefill) on the SAME request...", flush=True)
reuse_B = run_conversation(shared, PROMPT_B)        # 2nd conversation on the SAME request
print(f"  [.] shared B done: {reuse_B}", flush=True)
print(f"  same-request   A: {reuse_A}")
print(f"  same-request   B: {reuse_B}  <- 2nd conversation, reused request")

okA = reuse_A == base_A
okB = reuse_B == base_B
print(f"\n  conversation A (reused == fresh): {'PASS' if okA else 'FAIL'}")
print(f"  conversation B (reused == fresh): {'PASS' if okB else 'FAIL'}  <- the stale-KV / variant-switch case")
print("\nOVERALL:", "PASS" if (okA and okB) else "FAIL")
print("BASELINE_A=" + ",".join(map(str, base_A)))
print("BASELINE_B=" + ",".join(map(str, base_B)))
sys.stdout.flush()
os._exit(0 if (okA and okB) else 1)
