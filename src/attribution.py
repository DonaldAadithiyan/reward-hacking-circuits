"""Phase 1, Steps 3-4 - node & edge Indirect Effects via attribution patching.

Method (Marks et al. 2025, linear/attribution-patching approximation):

  For a submodule with SAE, we replace the residual activation x at its hook with
      x_hat = decode(f) + err,      f = encode(x),  err = x - decode(encode(x))
  and treat f (the SAE feature activations) as a leaf tensor that requires grad.
  Then for metric m,  grad_f = d m / d f  at the CLEAN activation, and the
  attribution-patching node IE for feature i is

      IE_i = grad_f_i * (f_hack_i - f_clean_i)      (first-order Taylor of m)

  averaged over triples. Positive IE => feature drives hacking.

We pool IE across token positions (sum) to get one score per (layer, submodule,
feature). m is read at the final completion token (mean-completion metric is used
for the readout to be robust; see model_utils).

Edge IE (Step 4): for upstream feature u (layer l) and downstream d (layer l+1
region, i.e. a later submodule), the direct-path contribution is approximated as
      IE_{u->d} = (d m / d d)|clean * (d d / d u)|clean,stop * (u_hack - u_clean)
  We compute this only between surviving nodes, using the cached grads and a
  Jacobian-vector approximation. To stay tractable we use the linear approximation
  IE_{u->d} ~ grad_d * J_{d,u} * delta_u where J is estimated by the SAE decoder ->
  next-encoder composition; in practice we use the empirical co-attribution
  (grad_d * delta_d restricted to the path from u), matching the codebase's
  stop-gradient formulation at first order.
"""
from __future__ import annotations
import functools
import torch
from model_utils import load_model
from sae_loader import get_sae, hook_name, all_nodes
import config as C


def _splice_hooks(model, saes, feat_store, delta_store=None, clean_feats=None):
    """Return TransformerLens fwd hooks that replace each submodule activation
    with decode(f)+err, storing f (leaf, requires grad) in feat_store keyed by
    (layer, submodule). If clean_feats given, we DON'T detach err differently;
    err is always detached (frozen reconstruction residual)."""
    hooks = []
    for (layer, sub), sae in saes.items():
        hn = hook_name(layer, sub)
        key = (layer, sub)
        def make(sae=sae, key=key):
            def hook(act, hook):  # act: [B, T, d_model]
                with torch.no_grad():
                    f0 = sae.encode(act)
                    recon = sae.decode(f0)
                    err = (act - recon).detach()
                f = f0.detach().clone().requires_grad_(True)
                feat_store[key] = f
                return sae.decode(f) + err
            return hook
        hooks.append((hn, make()))
    return hooks


def _read_metric(model, tokens, prompt_len):
    """m-readout term for a single sequence: mean log-prob of completion tokens.
    Returns a scalar tensor with grad."""
    logits = model(tokens)                      # [1, L, V]
    logprobs = torch.log_softmax(logits[0], dim=-1)
    comp_ids = tokens[0, prompt_len:]
    pred = logprobs[prompt_len - 1:-1]
    if comp_ids.numel() == 0:
        return logprobs.sum() * 0.0
    tok_lp = pred[torch.arange(comp_ids.shape[0]), comp_ids]
    return tok_lp.mean()


def node_indirect_effects(hack_type, pairs, layers=None, submodules=None,
                          device=None, verbose=True):
    """Compute averaged node IE for one hacking type.

    Returns dict: (layer, submodule) -> tensor[d_sae] of averaged IE.
    Also returns clean feature means for later ablation.
    """
    device = device or C.DEVICE
    layers = layers or C.SAE_LAYERS
    submodules = submodules or C.SUBMODULES
    model = load_model(device)
    saes = {(l, s): get_sae(l, s, device) for l in layers for s in submodules}

    ie_accum = {k: torch.zeros(saes[k].cfg.d_sae, device=device) for k in saes}
    clean_mean = {k: torch.zeros(saes[k].cfg.d_sae, device=device) for k in saes}
    count = 0

    for idx, tri in enumerate(pairs):
        prompt = tri["prompt"]
        # tokenise clean & hacking full sequences
        c_full = model.to_tokens(prompt + " " + tri["clean"].lstrip())
        h_full = model.to_tokens(prompt + " " + tri["hacking"].lstrip())
        p_ids = model.to_tokens(prompt)
        p_len = p_ids.shape[1]

        # ---- CLEAN pass: get f_clean and grad_m/f_clean ----
        feat_clean = {}
        hooks = _splice_hooks(model, saes, feat_clean)
        model.reset_hooks()
        for hn, fn in hooks:
            model.add_hook(hn, fn)
        m_clean = _read_metric(model, c_full, p_len)
        grads = torch.autograd.grad(m_clean, list(feat_clean.values()),
                                    retain_graph=False, allow_unused=True)
        grad_map = {}
        for (k, f), g in zip(feat_clean.items(), grads):
            # pool grads over token positions (sum): grad wrt each feature
            grad_map[k] = (g.detach().sum(dim=(0, 1)) if g is not None
                           else torch.zeros(saes[k].cfg.d_sae, device=device))
            clean_mean[k] += f.detach().sum(dim=(0, 1)) / f.shape[1]
        model.reset_hooks()

        # ---- HACKING pass: get f_hack (no grad needed) ----
        feat_hack = {}
        with torch.no_grad():
            model.reset_hooks()
            for (layer, sub), sae in saes.items():
                hn = hook_name(layer, sub)
                key = (layer, sub)
                def make(sae=sae, key=key):
                    def hook(act, hook):
                        feat_hack[key] = sae.encode(act).sum(dim=(0, 1))
                        return act
                    return hook
                model.add_hook(hn, make())
            model(h_full)
            model.reset_hooks()

        # ---- IE = grad * (f_hack - f_clean), pooled over positions ----
        f_clean_pooled = {k: feat_clean[k].detach().sum(dim=(0, 1)) for k in feat_clean}
        for k in saes:
            delta = feat_hack[k] - f_clean_pooled[k]
            ie_accum[k] += grad_map[k] * delta
        count += 1
        if verbose and (idx + 1) % 10 == 0:
            print(f"  [{hack_type}] node IE {idx+1}/{len(pairs)}")

    ie = {k: (v / count).cpu() for k, v in ie_accum.items()}
    clean_mean = {k: (v / count).cpu() for k, v in clean_mean.items()}
    return ie, clean_mean


def select_nodes(ie, t_n):
    """Return list of (layer, sub, feat_idx, ie_value) above |t_n|."""
    nodes = []
    for (layer, sub), vec in ie.items():
        idxs = torch.nonzero(vec.abs() > t_n).flatten().tolist()
        for i in idxs:
            nodes.append((layer, sub, i, float(vec[i])))
    nodes.sort(key=lambda x: -abs(x[3]))
    return nodes


def auto_threshold(ie, lo=C.NODE_MIN, hi=C.NODE_MAX, start=C.T_N):
    """Tune T_N so circuit size lands in [lo, hi]."""
    t = start
    for _ in range(40):
        n = len(select_nodes(ie, t))
        if n < lo:
            t *= 0.7
        elif n > hi:
            t *= 1.3
        else:
            return t, n
    return t, len(select_nodes(ie, t))
