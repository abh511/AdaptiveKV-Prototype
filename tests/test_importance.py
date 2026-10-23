"""
tests/test_importance.py
------------------------
Unit tests for EMATracker and AttentionHook.

Hand calculations are shown in comments.
"""

import pytest
import numpy as np
import torch
from adaptivekv.importance import EMATracker, AttentionHook


# ---------------------------------------------------------------------------
# T2.3 — EMA sequence, hand-computed
# ---------------------------------------------------------------------------

def test_ema_new_token_init():
    """
    New token is initialised to its raw score (no EMA with no prior).

    step 0: scores = [0.8]  → importance[0] = 0.8 (first obs)
    """
    ema = EMATracker(alpha=0.5)
    ema.update(np.array([0.8]))
    imp = ema.get()
    assert abs(imp[0] - 0.8) < 1e-9, f"Expected 0.8, got {imp[0]}"


def test_ema_sequence_hand():
    """
    alpha=0.5, three steps with one token growing sequence.

    step 0: token 0 score=0.8 → importance[0] = 0.8
    step 1: token 0 score=0.4, token 1 score=0.7
            importance[0] = 0.5*0.4 + 0.5*0.8 = 0.2 + 0.4 = 0.6
            importance[1] = 0.7  (new token, first obs)
    step 2: token 0 score=0.6, token 1 score=0.3, token 2 score=0.9
            importance[0] = 0.5*0.6 + 0.5*0.6 = 0.6
            importance[1] = 0.5*0.3 + 0.5*0.7 = 0.15 + 0.35 = 0.5
            importance[2] = 0.9  (new token)
    """
    ema = EMATracker(alpha=0.5)

    ema.update(np.array([0.8]))
    ema.update(np.array([0.4, 0.7]))
    imp = ema.get()
    assert abs(imp[0] - 0.6) < 1e-9, f"step1 imp[0]: expected 0.6, got {imp[0]}"
    assert abs(imp[1] - 0.7) < 1e-9, f"step1 imp[1]: expected 0.7, got {imp[1]}"

    ema.update(np.array([0.6, 0.3, 0.9]))
    imp = ema.get()
    assert abs(imp[0] - 0.6)  < 1e-9, f"step2 imp[0]: expected 0.6, got {imp[0]}"
    assert abs(imp[1] - 0.5)  < 1e-9, f"step2 imp[1]: expected 0.5, got {imp[1]}"
    assert abs(imp[2] - 0.9)  < 1e-9, f"step2 imp[2]: expected 0.9, got {imp[2]}"


def test_ema_reset():
    """After reset(), get() raises RuntimeError."""
    ema = EMATracker(alpha=0.5)
    ema.update(np.array([0.5, 0.3]))
    ema.reset()
    with pytest.raises(RuntimeError):
        ema.get()


def test_ema_alpha_one_no_smoothing():
    """alpha=1 means no EMA (always latest score)."""
    ema = EMATracker(alpha=1.0)
    ema.update(np.array([0.9]))
    ema.update(np.array([0.1]))
    imp = ema.get()
    assert abs(imp[0] - 0.1) < 1e-9, f"Expected 0.1 with alpha=1, got {imp[0]}"


def test_ema_length():
    """get() returns correct length after growing sequence."""
    ema = EMATracker(alpha=0.3)
    for n in range(1, 6):
        ema.update(np.random.rand(n))
        imp = ema.get()
        assert len(imp) == n, f"Expected len {n}, got {len(imp)}"


# ---------------------------------------------------------------------------
# T2.4 — Hook shape test (requires model on GPU or CPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for hook test")
def test_hook_shape():
    """
    Verify AttentionHook returns shape (seq_len,) after a real forward pass.
    Uses a very short sequence to minimise VRAM.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    prompt = "Hello"
    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
    seq_len = inp["input_ids"].shape[1]

    hook = AttentionHook(model)
    hook.register()
    with torch.no_grad():
        model(**inp, output_attentions=True, use_cache=False)
    scores = hook.get_scores()
    hook.remove()

    assert scores.shape == (seq_len,), \
        f"Expected shape ({seq_len},), got {scores.shape}"
    # Scores should be non-negative (they are attention weights)
    assert (scores >= 0).all(), "Attention scores should be non-negative"
    # Sum should be approximately 1 (normalised attention)
    assert abs(scores.sum() - 1.0) < 0.1, \
        f"Scores sum = {scores.sum():.4f}, expected ~1.0"

    del model
    torch.cuda.empty_cache()
