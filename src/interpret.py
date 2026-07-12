"""Feature interpretation - top promoting tokens for an SAE feature.

For feature i at (layer, resid), the decoder direction W_dec[i] lives in the
residual stream. We map it to vocabulary logits via the model's unembedding
(W_U), giving the tokens this feature most promotes when it writes to the residual.
This is the standard 'logit lens' interpretation of a residual-stream feature.
"""
from __future__ import annotations
import torch
from model_utils import load_model
from sae_loader import get_sae
import config as C


def top_tokens_for_feature(layer, feat_idx, k=20, device=None):
    device = device or C.DEVICE
    model = load_model(device)
    sae = get_sae(layer, "resid", device)
    d = sae.W_dec[feat_idx].detach()                  # [d_model]
    # apply final layernorm scale approximately via direct unembed projection
    logits = d @ model.W_U                            # [vocab]
    top = torch.topk(logits, k)
    toks = [model.to_string(torch.tensor([t])) for t in top.indices.tolist()]
    return list(zip(toks, [round(float(v), 3) for v in top.values]))


def describe_top_nodes(nodes, top_n=10, k=15, device=None):
    out = []
    for (l, s, i, v) in nodes[:top_n]:
        toks = top_tokens_for_feature(l, i, k, device)
        out.append({"layer": l, "submodule": s, "feature": i, "ie": v,
                    "top_tokens": toks})
    return out
