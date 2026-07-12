"""Robust faithfulness via activation patching (complements mean-ablation).

Mean-ablating ~99% of features at all 12 layers collapses the residual stream and
drives m_circuit -> 0 / NaN (see faithfulness.py results). For attribution-derived
circuits the more appropriate, numerically stable metric is patching recovery:

  Run the CLEAN completion, but PATCH the circuit's node features from their clean
  value to their hacking value (add (f_hack_i - f_clean_i) * W_dec[i] to the
  residual at that feature's layer). Measure the recovered change in m.

  faithfulness_patch = (m_patched - m_clean_baseline) / (m_full)

  where m_full = m(hacking) - m(clean). If patching only the circuit features
  recovers most of m_full, the circuit is faithful. This never destroys the
  residual (it perturbs a handful of directions), so it is stable on long seqs.
"""
from __future__ import annotations
import json, os
import torch
from model_utils import load_model, load_pairs
from sae_loader import get_sae, hook_name
import config as C


def _last_lp(model, tokens, plen):
    lp = torch.log_softmax(model(tokens)[0], -1)
    ci = tokens[0, plen:]
    tl = lp[plen - 1:-1][torch.arange(ci.shape[0]), ci]
    return (tl[-1] if C.READOUT == "last" else tl.mean()).item()


@torch.no_grad()
def patch_faithfulness(hack_type, n_pairs=None):
    device = C.DEVICE
    model = load_model(device)
    circuit = json.load(open(os.path.join(C.RESULTS_DIR, "phase1", f"circuit_{hack_type}.json")))
    nodes = circuit["nodes"]
    layers = sorted({n["layer"] for n in nodes})
    saes = {l: get_sae(l, "resid", device) for l in layers}
    Wdec = {l: saes[l].W_dec.detach() for l in layers}
    by_layer = {}
    for n in nodes:
        by_layer.setdefault(n["layer"], []).append(n["feature"])

    pairs = load_pairs(hack_type)
    if n_pairs:
        pairs = pairs[:n_pairs]

    m_full_s = m_patch_s = 0.0
    n = 0
    for tri in pairs:
        prompt = tri["prompt"]
        c_full = model.to_tokens(prompt + " " + tri["clean"].lstrip())
        h_full = model.to_tokens(prompt + " " + tri["hacking"].lstrip())
        plen = model.to_tokens(prompt).shape[1]

        # cache clean & hack feature activations per layer
        f_clean, f_hack = {}, {}
        cache_c, cache_h = {}, {}
        def grab(store):
            def mk(l):
                def hook(a, hook):
                    store[l] = a.detach()
                    return a
                return hook
            return mk
        model.reset_hooks()
        for l in layers:
            model.add_hook(hook_name(l, "resid"), grab(cache_c)(l))
        m_clean = _last_lp(model, c_full, plen)
        model.reset_hooks()
        for l in layers:
            model.add_hook(hook_name(l, "resid"), grab(cache_h)(l))
        m_hack = _last_lp(model, h_full, plen)
        model.reset_hooks()

        # patched run on CLEAN tokens: add circuit-feature delta directions
        def patch(l):
            idxs = torch.tensor(by_layer[l], device=device)
            def hook(a, hook):
                fc = saes[l].encode(a)[:, :, idxs]          # [1,T,k] clean feats
                # hack feats at aligned positions (clean & hack differ in length;
                # patch on overlapping min length from the end)
                fh = saes[l].encode(cache_h[l])[:, :, idxs]
                T = min(fc.shape[1], fh.shape[1])
                delta = torch.zeros_like(fc)
                delta[:, -T:, :] = (fh[:, -T:, :] - fc[:, -T:, :])
                add = delta @ Wdec[l][idxs]                 # [1,T,d_model]
                return a + add
            return hook
        model.reset_hooks()
        for l in layers:
            model.add_hook(hook_name(l, "resid"), patch(l))
        m_patched = _last_lp(model, c_full, plen)
        model.reset_hooks()

        m_full = m_hack - m_clean
        m_recovered = m_patched - m_clean
        if any(map(lambda x: x != x, [m_full, m_recovered])):  # NaN guard
            continue
        m_full_s += m_full
        m_patch_s += m_recovered
        n += 1

    m_full = m_full_s / n
    m_rec = m_patch_s / n
    faith = m_rec / m_full if abs(m_full) > 1e-9 else float("nan")
    return {"faithfulness_patch": faith, "m_full": m_full,
            "m_recovered": m_rec, "n": n}


if __name__ == "__main__":
    import sys
    for ht in (sys.argv[1:] or C.HACK_TYPES):
        r = patch_faithfulness(ht)
        print(ht, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
