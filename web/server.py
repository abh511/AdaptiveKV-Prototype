"""
web/server.py  -  AdaptiveKV live comparison (plain HTML UI, no Streamlit)

Serves web/index.html and one JSON endpoint that runs FP16 and Adaptive on the
same prompt. Every number in the response is measured live by the same engine
the experiments use (adaptivekv.generate); nothing is typed in.

Run:  ./run_web.sh      (or: .venv/bin/python web/server.py --port 8600)
"""

import argparse
import os
import sys
import threading

import torch
import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer                        # noqa: E402
from adaptivekv.generate import generate_adaptive, generate_baseline, load_config   # noqa: E402
from adaptivekv.metrics import compute_token_agreement, first_divergence             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SETTING_KEYS = ("model_id", "p8", "p4", "p2", "int2_enabled", "ema_alpha", "quant_group_size", "key_quant",
                "key_group_tokens", "sink_tokens", "recent_tokens", "reassign_every", "importance_window")

STATE = {"status": "loading", "error": None}
GPU_LOCK = threading.Lock()          # one generation at a time on a 4 GB GPU


def load_model():
    """Same loading and warm-up as dashboard/app.py, run once in a background thread."""
    try:
        cfg = load_config(os.path.join(ROOT, "config.yaml"))
        dtype_name = cfg.get("dtype", "bfloat16")
        if dtype_name == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            dtype_name = "float16"
        tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], dtype=getattr(torch, dtype_name),
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            attn_implementation=cfg.get("attn_impl", "eager"), trust_remote_code=True).eval()
        warm = {**cfg, "max_new_tokens": 4}           # first timed run is not penalised by CUDA start-up
        generate_baseline(model, tok, "Hello", "fp16", warm, 4)
        generate_adaptive(model, tok, "Hello", warm, 4)
        STATE.update(cfg=cfg, tok=tok, model=model, dtype=dtype_name, status="ready")
    except Exception as e:                            # shown in the UI instead of a silent hang
        STATE.update(status="error", error=f"{type(e).__name__}: {e}")


def result_json(r, n_tokens):
    return {
        "text": r.generated_text,
        "kv_bytes": r.kv_memory["total_bytes"],
        "kv_tokens": n_tokens,
        "kv_bytes_per_token": r.kv_memory["total_bytes"] / n_tokens if n_tokens else None,
        "effective_bits": r.effective_bits,
        "peak_vram_bytes": r.peak_vram_bytes,
        "latency_s": r.gen_latency_s,
        "tokens_per_sec": r.tokens_per_sec,
        "generated_tokens": len(r.output_token_ids),
    }


def run_comparison(prompt, max_new, use_chat):
    cfg, tok, model = STATE["cfg"], STATE["tok"], STATE["model"]
    text_in = prompt
    if use_chat and tok.chat_template:
        text_in = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          tokenize=False, add_generation_prompt=True)
    run_cfg = {**cfg, "max_new_tokens": max_new}
    with GPU_LOCK:
        ref = generate_baseline(model, tok, text_in, "fp16", run_cfg)
        res = generate_adaptive(model, tok, text_in, run_cfg)
    n_f, n_a = ref.kv_memory["fp16_tokens"], sum(res.tier_dist.values())
    total = sum(res.tier_dist.values()) or 1
    return {
        "fp16": result_json(ref, n_f),
        "adaptive": result_json(res, n_a),
        "agreement": compute_token_agreement(res.output_token_ids, ref.output_token_ids),
        "first_divergence": first_divergence(res.output_token_ids, ref.output_token_ids),
        "tier_share": {str(k): v / total for k, v in sorted(res.tier_dist.items(), reverse=True) if v},
    }


async def index(_):
    return FileResponse(os.path.join(HERE, "index.html"))


async def info(_):
    out = {"status": STATE["status"], "error": STATE["error"],
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
    if STATE["status"] == "ready":
        out["settings"] = {k: STATE["cfg"].get(k) for k in SETTING_KEYS} | {"dtype": STATE["dtype"]}
    return JSONResponse(out)


async def run(request: Request):
    if STATE["status"] != "ready":
        return JSONResponse({"error": f"Model is {STATE['status']}."}, status_code=503)
    body = await request.json()
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        return JSONResponse({"error": "Enter a prompt first."}, status_code=400)
    max_new = max(8, min(128, int(body.get("max_new_tokens", 48))))
    try:
        return JSONResponse(await run_in_threadpool(run_comparison, prompt, max_new, bool(body.get("use_chat", True))))
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=500)


app = Starlette(routes=[Route("/", index), Route("/api/info", info), Route("/api/run", run, methods=["POST"])])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8600)
    a = ap.parse_args()
    threading.Thread(target=load_model, daemon=True).start()
    print(f"AdaptiveKV UI at http://{a.host}:{a.port}  (model loads in the background)")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")

