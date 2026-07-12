"""Shared model loading and the behaviour metric m."""
from __future__ import annotations
import json, os, functools
import torch
from transformer_lens import HookedTransformer
import config as C

@functools.lru_cache(maxsize=1)
def load_model(device: str | None = None) -> HookedTransformer:
    device = device or C.DEVICE
    model = HookedTransformer.from_pretrained(C.MODEL_NAME, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def load_pairs(hack_type: str):
    path = os.path.join(C.PAIRS_DIR, f"{hack_type}.jsonl")
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

@torch.no_grad()
def completion_logprob(model, prompt: str, completion: str, mode: str = "mean"):
    """log p(completion | prompt).

    mode='mean' : mean log-prob over completion tokens (robust, default)
    mode='last' : log-prob of only the final completion token (TASK's literal def)
    Returns a python float.
    """
    p_ids = model.to_tokens(prompt)                       # [1, Lp] (incl BOS)
    full_ids = model.to_tokens(prompt + completion)       # [1, Lf]
    Lp = p_ids.shape[1]
    logits = model(full_ids)                              # [1, Lf, V]
    logprobs = torch.log_softmax(logits[0], dim=-1)       # [Lf, V]
    # token at position t is predicted by logits at t-1
    comp_ids = full_ids[0, Lp:]                           # completion token ids
    pred_lp = logprobs[Lp - 1:-1]                         # aligned predictions
    if comp_ids.numel() == 0:
        return 0.0
    tok_lp = pred_lp[torch.arange(comp_ids.shape[0]), comp_ids]
    if mode == "last":
        return tok_lp[-1].item()
    return tok_lp.mean().item()

@torch.no_grad()
def metric_m(model, triple, mode: str = "mean"):
    """m = log p(hacking) - log p(clean).  Positive => model prefers hacking."""
    lp_h = completion_logprob(model, triple["prompt"], " " + triple["hacking"].lstrip(), mode)
    lp_c = completion_logprob(model, triple["prompt"], " " + triple["clean"].lstrip(), mode)
    return lp_h - lp_c
