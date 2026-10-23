"""
diagnose_cache.py - find WHICH stage of the AdaptiveKV path breaks quality.

Stage A  our engine (fp16 method) == plain Hugging Face greedy generate?
Stage B  quantize the model's real KV at INT8/INT4, rebuild it, compare
         (1) reconstruction error per layer, (2) next-token logits vs native cache
Stage C  all methods end to end (text, effective bits, tiers, agreement vs fp16)
Stage D  adaptive with different protection settings (sink / recent window)

Run from the project root:  python diagnose_cache.py
"""
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from adaptivekv.cache import AdaptiveKVCache
from adaptivekv.generate import generate_adaptive, generate_baseline, _layer_kv
from adaptivekv.metrics import compute_token_agreement, first_divergence

cfg = yaml.safe_load(open("config.yaml"))
cfg["max_new_tokens"] = 24
dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    cfg["model_id"], dtype=dtype, device_map=str(dev),
    attn_implementation=cfg.get("attn_impl", "eager"), trust_remote_code=True).eval()
n_layers = len(model.model.layers)

chat = tok.apply_chat_template([{"role": "user", "content": "What is the capital of India?"}],
                               tokenize=False, add_generation_prompt=True)
prompts = {"chat": chat, "raw": "The capital of India is"}

print("dtype:", dtype, "| group_size:", cfg["quant_group_size"], "| k_channel_norm:", cfg.get("k_channel_norm", True))
print("config  p8:", cfg["p8"], "p4:", cfg["p4"], "sink:", cfg.get("sink_tokens"),
      "recent:", cfg.get("recent_tokens"), "reassign_every:", cfg.get("reassign_every"))

# ---------------------------------------------------------------- Stage A
print("\n===== STAGE A: our fp16 engine vs plain HF greedy =====")
for name, text in prompts.items():
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        ref = model.generate(ids, max_new_tokens=24, do_sample=False, repetition_penalty=1.0,
                             temperature=None, top_p=None, top_k=None)[0, ids.shape[1]:].tolist()
    ours = generate_baseline(model, tok, text, "fp16", cfg, 24).output_token_ids
    print(f"[{name}] identical tokens: {ref == ours}   plain: {tok.decode(ref)!r}")

# ---------------------------------------------------------------- Stage B
print("\n===== STAGE B: reconstruct the real KV cache, compare logits =====")
ids = tok(chat, return_tensors="pt").input_ids.to(dev)
with torch.no_grad():
    out = model(ids, use_cache=True)
orig = [tuple(t.clone() for t in _layer_kv(out.past_key_values, i)) for i in range(n_layers)]
nxt = int(out.logits[0, -1].argmax())
del out


def native_step(kvs):
    dyn = DynamicCache()
    for i, (K, V) in enumerate(kvs):
        dyn.update(K.clone(), V.clone(), i)
    with torch.no_grad():
        o = model(torch.tensor([[nxt]], device=dev), past_key_values=dyn, use_cache=True)
    return o.logits[0, -1].float()


base = native_step(orig)
mid = n_layers // 2
K0, V0 = orig[mid]
print(f"attention sink check (layer {mid}): mean|K| pos0 / rest = "
      f"{(K0[:, :, 0].abs().mean() / K0[:, :, 1:].abs().mean()).item():.1f}x ; "
      f"mean|V| pos0 / rest = {(V0[:, :, 0].abs().mean() / V0[:, :, 1:].abs().mean()).item():.1f}x")

for bits in (8, 4, 2):
    c = AdaptiveKVCache(n_layers, cfg["quant_group_size"], k_channel_norm=cfg.get("k_channel_norm", True))
    for i, (K, V) in enumerate(orig):
        c.append_tokens(i, K, V, np.full(K.shape[2], bits))
    rec = [c.get_kv_fp16(i, dev) for i in range(n_layers)]
    ek = [((r[0].float() - o[0].float()).abs().mean() / o[0].float().abs().mean()).item() for r, o in zip(rec, orig)]
    ev = [((r[1].float() - o[1].float()).abs().mean() / o[1].float().abs().mean()).item() for r, o in zip(rec, orig)]
    lg = native_step(rec)
    kl = F.kl_div(F.log_softmax(lg, -1), F.softmax(base, -1), reduction="sum").item()
    print(f"INT{bits}: mean rel err K worst layer {int(np.argmax(ek))}={max(ek):.4f}, "
          f"V worst layer {int(np.argmax(ev))}={max(ev):.4f} | next-token same: "
          f"{int(lg.argmax()) == int(base.argmax())} | KL vs native: {kl:.5f}")

# ---------------------------------------------------------------- Stage C
print("\n===== STAGE C: end to end =====")
for name, text in prompts.items():
    ref = generate_baseline(model, tok, text, "fp16", cfg, 24)
    print(f"\n[{name}] fp16        : {ref.generated_text!r}")
    for m in ("int8_uniform", "int4_uniform", "recency"):
        r = generate_baseline(model, tok, text, m, cfg, 24)
        print(f"[{name}] {m:13s}: agree {compute_token_agreement(r.output_token_ids, ref.output_token_ids) * 100:5.1f}% "
              f"bits {r.effective_bits:.2f} first_diff {first_divergence(r.output_token_ids, ref.output_token_ids)} "
              f"{r.generated_text[:60]!r}")
    r = generate_adaptive(model, tok, text, cfg, 24)
    print(f"[{name}] adaptive     : agree {compute_token_agreement(r.output_token_ids, ref.output_token_ids) * 100:5.1f}% "
          f"bits {r.effective_bits:.2f} tiers {r.tier_dist} {r.generated_text[:60]!r}")

# ---------------------------------------------------------------- Stage D
print("\n===== STAGE D: adaptive vs protection settings (chat prompt) =====")
ref = generate_baseline(model, tok, chat, "fp16", cfg, 24)
for label, over in {"as in config.yaml": {},
                    "sink=4, recent=8": {"sink_tokens": 4, "recent_tokens": 8},
                    "sink=4, recent=16": {"sink_tokens": 4, "recent_tokens": 16},
                    "no demotion (recent=10**6)": {"recent_tokens": 10 ** 6}}.items():
    r = generate_adaptive(model, tok, chat, {**cfg, **over}, 24)
    print(f"{label:28s}: agree {compute_token_agreement(r.output_token_ids, ref.output_token_ids) * 100:5.1f}% "
          f"bits {r.effective_bits:.2f} tiers {r.tier_dist} {r.generated_text[:50]!r}")

print("""
HOW TO READ THIS
  A false                      -> bug in the engine/DynamicCache injection itself (send me this output)
  B INT8 not ~0.00x error or next-token False -> bug in quant.py / cache.py (send me this output)
  B INT8 fine, INT4 bad         -> expected precision loss; check the sink-token ratio line
  C int8_uniform garbage       -> bug in the decode loop (send me this output)
  D 'no demotion' good but 'as in config.yaml' bad -> demotion policy / sink_tokens=0 is the cause
""")