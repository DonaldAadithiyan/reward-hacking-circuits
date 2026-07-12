"""Phase 1, Step 5 - circuit faithfulness & completeness.

Faithfulness: mean-ablate all SAE features NOT in the circuit (replace their
activation with their clean-mean), reconstruct the residual, run forward, measure
m_circuit. Faithfulness = m_circuit / m_full.

Completeness: mean-ablate only the features IN the circuit.
Completeness = 1 - m_without_circuit / m_full.

We ablate in the SAE feature basis on the residual stream: at each layer's
hook_resid_pre we replace x with  decode(f_ablated) + err  where f_ablated keeps
circuit features at their true value and sets all others to their clean mean (or,
for completeness, sets circuit features to mean and keeps the rest).
"""
from __future__ import annotations
import torch
from model_utils import load_model
from sae_loader import get_sae, hook_name
import config as C


def _circuit_feature_masks(nodes, layers, d_sae_map, device):
    """Boolean mask per layer: True where feature is in the circuit."""
    mask = {l: torch.zeros(d_sae_map[l], dtype=torch.bool, device=device) for l in layers}
    for l, s, i, v in nodes:
        if l in mask:
            mask[l][i] = True
    return mask


@torch.no_grad()
def _run_ablated(model, saes, clean_means, mask, tokens, prompt_len, mode):
    """mode='keep_circuit'  -> ablate features NOT in circuit (faithfulness)
       mode='ablate_circuit'-> ablate features IN circuit (completeness)"""
    def make(key):
        l = key[0]
        sae = saes[key]
        cmean = clean_means[key].to(tokens.device)   # [d_sae]
        m = mask[l]                                   # [d_sae] True=in circuit

        def hook(act, hook):
            f = sae.encode(act)                       # [1,T,d_sae]
            cm = cmean.view(1, 1, -1).expand_as(f)
            if mode == "keep_circuit":
                # keep circuit features, mean-ablate the rest
                f_new = torch.where(m.view(1, 1, -1), f, cm)
            else:
                # ablate circuit features to mean, keep the rest
                f_new = torch.where(m.view(1, 1, -1), cm, f)
            recon = sae.decode(f)
            err = (act - recon)
            return sae.decode(f_new) + err
        return hook

    model.reset_hooks()
    for key in saes:
        model.add_hook(hook_name(*key), make(key))
    logits = model(tokens)
    model.reset_hooks()
    logprobs = torch.log_softmax(logits[0], dim=-1)
    comp_ids = tokens[0, prompt_len:]
    if comp_ids.numel() == 0:
        return 0.0
    pred = logprobs[prompt_len - 1:-1]
    tok_lp = pred[torch.arange(comp_ids.shape[0]), comp_ids]
    return (tok_lp[-1] if C.READOUT == "last" else tok_lp.mean()).item()


@torch.no_grad()
def evaluate_circuit(hack_type, pairs, nodes, clean_means, device=None, verbose=True):
    """Returns dict with faithfulness, completeness, m_full, m_circuit, m_without."""
    device = device or C.DEVICE
    model = load_model(device)
    layers = sorted({l for l, s, i, v in nodes})
    saes = {(l, "resid"): get_sae(l, "resid", device) for l in layers}
    d_sae_map = {l: saes[(l, "resid")].cfg.d_sae for l in layers}
    mask = _circuit_feature_masks(nodes, layers, d_sae_map, device)
    cm = {k: clean_means[k] for k in saes}

    import math
    m_full_s, m_circ_s, m_without_s = 0.0, 0.0, 0.0
    n = 0
    n_skipped = 0
    for tri in pairs:
        prompt = tri["prompt"]
        h_full = model.to_tokens(prompt + " " + tri["hacking"].lstrip())
        c_full = model.to_tokens(prompt + " " + tri["clean"].lstrip())
        p_len = model.to_tokens(prompt).shape[1]

        def m_of(tokens):
            model.reset_hooks()
            logits = model(tokens)
            lp = torch.log_softmax(logits[0], -1)
            ci = tokens[0, p_len:]
            tl = lp[p_len - 1:-1][torch.arange(ci.shape[0]), ci]
            return (tl[-1] if C.READOUT == "last" else tl.mean()).item()

        m_full = m_of(h_full) - m_of(c_full)
        # circuit-only m (keep circuit, ablate rest)
        mh_k = _run_ablated(model, saes, cm, mask, h_full, p_len, "keep_circuit")
        mc_k = _run_ablated(model, saes, cm, mask, c_full, p_len, "keep_circuit")
        m_circ = mh_k - mc_k
        # without circuit (ablate circuit, keep rest)
        mh_a = _run_ablated(model, saes, cm, mask, h_full, p_len, "ablate_circuit")
        mc_a = _run_ablated(model, saes, cm, mask, c_full, p_len, "ablate_circuit")
        m_without = mh_a - mc_a

        # aggressive mean-ablation can blow up long sequences -> NaN/inf.
        # skip those pairs and count them; average over the stable subset.
        vals = [m_full, m_circ, m_without]
        if any(math.isnan(x) or math.isinf(x) for x in vals):
            n_skipped += 1
            continue
        m_full_s += m_full; m_circ_s += m_circ; m_without_s += m_without
        n += 1

    m_full = m_full_s / n if n else float("nan")
    m_circ = m_circ_s / n if n else float("nan")
    m_without = m_without_s / n if n else float("nan")
    faith = m_circ / m_full if abs(m_full) > 1e-9 else float("nan")
    comp = 1 - (m_without / m_full) if abs(m_full) > 1e-9 else float("nan")
    if verbose:
        print(f"  [{hack_type}] m_full={m_full:.3f} m_circuit={m_circ:.3f} "
              f"m_without={m_without:.3f} faith={faith:.3f} comp={comp:.3f} "
              f"(n={n}, skipped={n_skipped})")
    return dict(faithfulness=faith, completeness=comp, n_skipped=n_skipped,
                m_full=m_full,
                m_circuit=m_circ, m_without=m_without, n=n)
