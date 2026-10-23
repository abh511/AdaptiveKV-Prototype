"""
diagnose.py  -  is the garbage coming from the MODEL setup or from the KV cache?

Uses ONLY plain Hugging Face generate() (no AdaptiveKV code), in float16 and
bfloat16, with and without the chat template, and checks logits for NaN/Inf.

Run:  python diagnose.py
"""
import gc
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

cfg = yaml.safe_load(open("config.yaml"))
tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
question = "What is the capital of India?"
chat = tok.apply_chat_template([{"role": "user", "content": question}],
                               tokenize=False, add_generation_prompt=True)
variants = {"raw prompt": "The capital of India is", "chat template": chat}

print("model:", cfg["model_id"], "| attn:", cfg.get("attn_impl", "eager"))
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())

for dtype in (torch.float16, torch.bfloat16):
    if dtype is torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("\nbfloat16 not supported on this GPU, skipping")
        continue
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=dtype, device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation=cfg.get("attn_impl", "eager"), trust_remote_code=True).eval()
    print(f"\n===== {dtype} =====")
    for name, text in variants.items():
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
            finite = bool(torch.isfinite(logits).all())
            out = model.generate(ids, max_new_tokens=20, do_sample=False)
        print(f"[{name}] logits finite: {finite}")
        print(f"[{name}] output: {tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)!r}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

print("""
HOW TO READ THIS
  fp16 garbage or 'finite: False', bf16 fine -> fp16 overflow in the model. Use dtype: bfloat16.
  both dtypes garbage                        -> model files/transformers problem, not AdaptiveKV.
  both fine                                  -> the model is OK; the bug is in the KV-cache path.
""")