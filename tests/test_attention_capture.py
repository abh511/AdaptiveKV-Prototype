"""
tests/test_attention_capture.py
-------------------------------
Regression test for the adaptive-method OOM at long context (2048 tokens).

The importance hook must get its attention weights WITHOUT output_attentions=True:
with it, transformers 5.x keeps every layer's (heads x seq x seq) matrix in the
output (measured: 1.5 GB at 1536 tokens on Qwen2.5-0.5B), which OOMs a 4 GB GPU.
Runs on CPU with a tiny randomly initialised Qwen2 model (no download).
"""

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from adaptivekv.importance import AttentionHook


def _tiny_model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


def _scores(model, ids, output_attentions):
    hook = AttentionHook(model, window=1)
    hook.register()
    with torch.no_grad():
        out = model(ids, use_cache=True, output_attentions=output_attentions)
    s = hook.get_scores()
    hook.remove()
    return s, out


def test_hook_gets_weights_without_output_attentions():
    model = _tiny_model()
    ids = torch.randint(0, 128, (1, 40))
    s_off, out_off = _scores(model, ids, False)
    s_on, _ = _scores(model, ids, True)
    assert s_off.shape == (40,)
    np.testing.assert_allclose(s_off, s_on, rtol=1e-5, atol=1e-7)   # same importance either way
    held = getattr(out_off, "attentions", None)
    assert held is None or all(a is None for a in held)           # nothing retained


def test_engine_prefill_does_not_request_attentions():
    code = [l.split("#", 1)[0] for l in open("adaptivekv/generate.py")]   # ignore comments
    src = "\n".join(code)
    assert "output_attentions=adaptive" not in src
    assert "output_attentions=True" not in src
