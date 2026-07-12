"""Phase 1, Steps 3-4 - node & edge Indirect Effects via attribution patching.

Method (Marks et al. 2025, linear attribution-patching approximation).

We attribute on the residual stream (hook_resid_pre) at each layer. For an SAE
with decoder decode(f) = f @ W_dec + b_dec, feature i contributes to the residual
activation x through the direction W_dec[i]. The first-order (attribution-patching)
Indirect Effect of feature i on metric m is

    IE_i = (grad_x · W_dec[i]) * (f_hack_i - f_clean_i)

evaluated at the CLEAN activation, where grad_x = d m / d x is the gradient of the
metric w.r.t. the residual activation, and f = encode(x). We pool over token
positions by summing the per-position contributions. Averaged over triples.

  Positive IE  => feature causally drives the model toward the hacking completion.
  Negative IE  => feature resists hacking.

m is read as the mean log-prob of the completion tokens (robust readout).

This formulation gives correct nonzero gradients at every layer (splicing a fresh
leaf per layer and re-encoding under no_grad severs the graph; grad-w.r.t.-x does
not). It is mathematically the attribution-patching estimator restricted to the
SAE feature basis.
"""
from __future__ import annotations
import torch
from model_utils import load_model
from sae_loader import get_sae, hook_name
import config as C


def _metric_from_logits(logits, tokens, prompt_len):
    logprobs = torch.log_softmax(logits[0], dim=-1)
    comp_ids = tokens[0, prompt_len:]
    if comp_ids.numel() == 0:
        return logprobs.sum() * 0.0
    pred = logprobs[prompt_len - 1:-1]
    tok_lp = pred[torch.arange(comp_ids.shape[0]), comp_ids]
    return tok_lp[-1] if C.READOUT == "last" else tok_lp.mean()


def _forward_capture(model, saes, tokens, prompt_len, want_grad):
    """Run model capturing residual activations at each SAE hook.
    If want_grad: activations retain grad and we backprop the metric.
    Returns (acts dict[key]->activation tensor, metric scalar)."""
    acts = {}

    def make(key):
        def hook(act, hook):
            if want_grad:
                act.requires_grad_(True)
                act.retain_grad()
            acts[key] = act
            return act
        return hook

    model.reset_hooks()
    for key in saes:
        model.add_hook(hook_name(*key), make(key))
    logits = model(tokens)
    m = _metric_from_logits(logits, tokens, prompt_len)
    model.reset_hooks()
    return acts, m


def node_indirect_effects(hack_type, pairs, layers=None, submodules=None,
                          device=None, verbose=True):
    """Averaged node IE per (layer, submodule) -> tensor[d_sae].
    Also returns per-feature clean-mean activations (for ablation later)."""
    device = device or C.DEVICE
    layers = layers or C.SAE_LAYERS
    submodules = submodules or C.SUBMODULES
    model = load_model(device)
    saes = {(l, s): get_sae(l, s, device) for l in layers for s in submodules}
    Wdec = {k: saes[k].W_dec.detach() for k in saes}   # [d_sae, d_model]

    ie_accum = {k: torch.zeros(saes[k].cfg.d_sae, device=device) for k in saes}
    clean_mean = {k: torch.zeros(saes[k].cfg.d_sae, device=device) for k in saes}
    count = 0

    for idx, tri in enumerate(pairs):
        prompt = tri["prompt"]
        c_full = model.to_tokens(prompt + " " + tri["clean"].lstrip())
        h_full = model.to_tokens(prompt + " " + tri["hacking"].lstrip())
        p_len = model.to_tokens(prompt).shape[1]

        # CLEAN pass with grad w.r.t. residual activations
        acts_c, m_c = _forward_capture(model, saes, c_full, p_len, want_grad=True)
        m_c.backward()
        grad_x = {k: acts_c[k].grad.detach() for k in acts_c}     # [1,T,d_model]

        # clean feature activations (pooled over positions)
        f_clean = {}
        with torch.no_grad():
            for k in saes:
                f = saes[k].encode(acts_c[k].detach())            # [1,T,d_sae]
                f_clean[k] = f.sum(dim=(0, 1))                     # pooled
                clean_mean[k] += f.mean(dim=(0, 1))

        # HACKING pass: feature activations only
        f_hack = {}
        with torch.no_grad():
            acts_h, _ = _forward_capture(model, saes, h_full, p_len, want_grad=False)
            for k in saes:
                f_hack[k] = saes[k].encode(acts_h[k]).sum(dim=(0, 1))

        # IE_i = sum_pos (grad_x @ W_dec[i]) * delta_f_i
        for k in saes:
            # grad_x: [1,T,d_model]; W_dec[k]: [d_sae,d_model]
            g = grad_x[k][0]                                       # [T,d_model]
            proj = g @ Wdec[k].T                                   # [T,d_sae]
            grad_f = proj.sum(dim=0)                               # pooled [d_sae]
            delta = f_hack[k] - f_clean[k]
            ie_accum[k] += grad_f * delta

        model.zero_grad(set_to_none=True)
        count += 1
        if verbose and (idx + 1) % 10 == 0:
            print(f"  [{hack_type}] node IE {idx+1}/{len(pairs)}")

    ie = {k: (v / count).cpu() for k, v in ie_accum.items()}
    clean_mean = {k: (v / count).cpu() for k, v in clean_mean.items()}
    return ie, clean_mean


def select_nodes(ie, t_n):
    nodes = []
    for (layer, sub), vec in ie.items():
        idxs = torch.nonzero(vec.abs() > t_n).flatten().tolist()
        for i in idxs:
            nodes.append((layer, sub, i, float(vec[i])))
    nodes.sort(key=lambda x: -abs(x[3]))
    return nodes


def auto_threshold(ie, lo=C.NODE_MIN, hi=C.NODE_MAX, start=C.T_N):
    t = start
    for _ in range(60):
        n = len(select_nodes(ie, t))
        if n < lo:
            t *= 0.7
        elif n > hi:
            t *= 1.25
        else:
            return t, n
    return t, len(select_nodes(ie, t))


def edge_indirect_effects(hack_type, pairs, nodes, device=None, verbose=True):
    """Edge IE between selected nodes (Step 4).

    For upstream node u=(l_u, i_u) and downstream node d=(l_d, i_d) with l_u < l_d,
    the residual-stream direct-path estimator is

        IE(u->d) = grad_fd * (dfd/dfu) * delta_fu

    where dfd/dfu is the derivative of downstream feature d's activation w.r.t.
    upstream feature u's activation, and grad_fd is d m / d f_d. In the linear
    residual-stream approximation the perturbation from u, W_dec[u]*delta_fu, reaches
    layer l_d unchanged along the residual skip path, and its effect on f_d is
    W_enc[d] . (W_dec[u]) . So

        dfd/dfu ~ (W_dec[u] . W_enc[:,d])         (residual skip / direct path)
        IE(u->d) ~ grad_fd * (W_dec[u].W_enc[d]) * delta_fu

    This is the direct-path (stop-gradient on intermediate features) term. Averaged
    over triples. Only computed among the surviving node set (tractable)."""
    device = device or C.DEVICE
    model = load_model(device)
    layers = sorted({l for l, s, i, v in nodes})
    saes = {(l, "resid"): get_sae(l, "resid", device) for l in layers}
    Wdec = {k: saes[k].W_dec.detach() for k in saes}     # [d_sae, d_model]
    Wenc = {k: saes[k].W_enc.detach() for k in saes}     # [d_model, d_sae]

    # group node feature indices by layer
    by_layer = {}
    for l, s, i, v in nodes:
        by_layer.setdefault(l, []).append(i)

    edge_accum = {}   # (l_u,i_u,l_d,i_d) -> summed IE
    count = 0

    for tri in pairs:
        prompt = tri["prompt"]
        c_full = model.to_tokens(prompt + " " + tri["clean"].lstrip())
        h_full = model.to_tokens(prompt + " " + tri["hacking"].lstrip())
        p_len = model.to_tokens(prompt).shape[1]

        acts_c, m_c = _forward_capture(model, saes, c_full, p_len, want_grad=True)
        m_c.backward()
        grad_x = {(l, "resid"): acts_c[(l, "resid")].grad.detach() for l in layers}
        with torch.no_grad():
            fclean = {(l, "resid"): saes[(l, "resid")].encode(acts_c[(l, "resid")].detach()).sum(dim=(0, 1))
                      for l in layers}
            acts_h, _ = _forward_capture(model, saes, h_full, p_len, want_grad=False)
            fhack = {(l, "resid"): saes[(l, "resid")].encode(acts_h[(l, "resid")]).sum(dim=(0, 1))
                     for l in layers}

        # grad_fd = grad_x_d @ W_dec[d]
        gradf = {}
        for l in layers:
            k = (l, "resid")
            gradf[k] = (grad_x[k][0].sum(0) @ Wdec[k].T)   # [d_sae]

        for lu in layers:
            ku = (lu, "resid")
            for ld in layers:
                if ld <= lu:
                    continue
                kd = (ld, "resid")
                iu = by_layer[lu]; idd = by_layer[ld]
                # W_dec[u] . W_enc[d]  ->  [len(iu), len(idd)]
                Du = Wdec[ku][iu]                 # [nu, d_model]
                Ed = Wenc[kd][:, idd]             # [d_model, nd]
                J = Du @ Ed                       # [nu, nd]  direct-path jacobian
                du = (fhack[ku] - fclean[ku])[iu]        # [nu]
                gd = gradf[kd][idd]                      # [nd]
                # IE(u->d) = gd_d * J_{u,d} * du_u
                ie_mat = (du.unsqueeze(1) * J) * gd.unsqueeze(0)   # [nu, nd]
                for a, uu in enumerate(iu):
                    for b, dd in enumerate(idd):
                        key = (lu, uu, ld, dd)
                        edge_accum[key] = edge_accum.get(key, 0.0) + float(ie_mat[a, b])
        model.zero_grad(set_to_none=True)
        count += 1
        if verbose and count % 10 == 0:
            print(f"  [{hack_type}] edge IE {count}/{len(pairs)}")

    return {k: v / count for k, v in edge_accum.items()}


def select_edges(edges, t_e):
    e = [(k[0], k[1], k[2], k[3], v) for k, v in edges.items() if abs(v) > t_e]
    e.sort(key=lambda x: -abs(x[4]))
    return e
