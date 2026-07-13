"""Phase 3 (measurement-only) - does ablating the discovered circuit REDUCE the
model's pre-existing preference for the labelled-hacking response?

We do NOT train the model to hack and we do NOT use programmatic rewards. The
"hacking" label comes entirely from the real datasets (Phase 1 pairs). We measure,
on a HELD-OUT split of real pairs, the model's preference

    m = readout log p(hacking) - readout log p(clean)          (m>0 => prefers hacking)

under four interventions on the discovered circuit features, and report how each
intervention moves m. This is the intervention analogue of SHIFT (Marks et al.):
inference-time editing of the causally-implicated features, evaluated on real data.

Interventions (top-K circuit features per type, K=config.TOP_K_CIRCUIT):
  baseline   - no edit (measures the raw preference)
  zero       - zero the circuit features (remove them)
  mean       - clamp circuit features to their clean-split mean (SHIFT-style)
  random     - zero the SAME NUMBER of RANDOM features (control: is the circuit special?)

A faithful, causal circuit should show: `zero`/`mean` reduce m toward 0 (or below)
substantially MORE than the `random` control. That is the whole test - no induction.

Train/eval split: circuits were discovered on pairs[: n_train]; we evaluate on the
held-out pairs[n_train:]. Cross-type: apply the SYCO circuit edit while evaluating on
LENGTH pairs (and vice versa) to test whether the intervention transfers.
"""
from __future__ import annotations
import json, os, functools, random
print = functools.partial(print, flush=True)
import torch
import config as C
from model_utils import load_model, load_pairs
from sae_loader import get_sae, hook_name

OUT = os.path.join(C.RESULTS_DIR, "phase3")
os.makedirs(OUT, exist_ok=True)
random.seed(C.SEED)

N_TRAIN = 40   # circuits discovered on first 40 pairs; evaluate on the rest


def _readout_m(model, tri, device):
    """m = readout logp(hacking) - readout logp(clean) under current hooks."""
    def lp(completion):
        toks = model.to_tokens(tri["prompt"] + " " + completion.lstrip())
        plen = model.to_tokens(tri["prompt"]).shape[1]
        logits = model(toks)
        logp = torch.log_softmax(logits[0], -1)
        ci = toks[0, plen:]
        if ci.numel() == 0:
            return 0.0
        tl = logp[plen - 1:-1][torch.arange(ci.shape[0]), ci]
        return (tl[-1] if C.READOUT == "last" else tl.mean()).item()
    return lp(tri["hacking"]) - lp(tri["clean"])


def _circuit_feats(hack_type, k):
    c = json.load(open(os.path.join(C.RESULTS_DIR, "phase1", f"circuit_{hack_type}.json")))
    nodes = sorted(c["nodes"], key=lambda n: -abs(n["ie"]))[:k]
    by_layer = {}
    for n in nodes:
        by_layer.setdefault(n["layer"], []).append(n["feature"])
    return by_layer


def _edit_hooks(model, by_layer, mode, saes, clean_means=None, d_sae=24576):
    """Return hooks that edit the given features on the residual stream."""
    def make(layer, idxs):
        sae = saes[layer]
        idx_t = torch.tensor(idxs, device=next(sae.parameters()).device)
        cm = None
        if mode == "mean" and clean_means is not None:
            cm = clean_means[(layer, "resid")].to(idx_t.device)
        def hook(act, hook):
            f = sae.encode(act)
            recon = sae.decode(f)
            err = act - recon
            if mode == "zero":
                f[..., idx_t] = 0.0
            elif mode == "mean":
                f[..., idx_t] = cm[idx_t]
            return sae.decode(f) + err
        return hook
    hooks = []
    for layer, idxs in by_layer.items():
        hooks.append((hook_name(layer, "resid"), make(layer, idxs)))
    return hooks


@torch.no_grad()
def measure(model, pairs, by_layer, mode, saes, clean_means, device):
    model.reset_hooks()
    if mode != "baseline":
        for hn, fn in _edit_hooks(model, by_layer, mode, saes, clean_means):
            model.add_hook(hn, fn)
    ms = [_readout_m(model, t, device) for t in pairs]
    model.reset_hooks()
    return {"mean_m": sum(ms) / len(ms),
            "frac_hack": sum(m > 0 for m in ms) / len(ms), "n": len(ms)}


def random_control(hack_type, n_feats, seed, d_sae=24576):
    """Same feature COUNT as the circuit, but random features at the same layers."""
    by = _circuit_feats(hack_type, C.TOP_K_CIRCUIT)
    rng = random.Random(seed)
    out = {}
    for layer, idxs in by.items():
        out[layer] = [rng.randrange(d_sae) for _ in idxs]
    return out


def run_type(hack_type, device):
    print(f"\n=== Phase 3 (measurement): {hack_type} ===")
    model = load_model(device)
    all_pairs = load_pairs(hack_type)
    eval_pairs = all_pairs[N_TRAIN:]            # held-out
    by_layer = _circuit_feats(hack_type, C.TOP_K_CIRCUIT)
    layers = list(by_layer.keys())
    saes = {l: get_sae(l, "resid", device) for l in layers}
    clean_means = torch.load(os.path.join(C.RESULTS_DIR, "phase1",
                                          f"clean_means_{hack_type}.pt"))
    res = {"n_eval": len(eval_pairs), "top_k": C.TOP_K_CIRCUIT,
           "n_circuit_feats": sum(len(v) for v in by_layer.values())}
    for mode in ["baseline", "zero", "mean"]:
        res[mode] = measure(model, eval_pairs, by_layer, mode, saes, clean_means, device)
        print(f"  {mode:9s} mean_m={res[mode]['mean_m']:+.3f} frac_hack={res[mode]['frac_hack']:.2f}")
    # random control: average over 3 seeds
    ctrl = []
    for s in range(3):
        rc = random_control(hack_type, res["n_circuit_feats"], s)
        rsaes = {l: get_sae(l, "resid", device) for l in rc}
        ctrl.append(measure(model, eval_pairs, rc, "zero", rsaes, clean_means, device))
    res["random_zero"] = {"mean_m": sum(c["mean_m"] for c in ctrl) / 3,
                          "frac_hack": sum(c["frac_hack"] for c in ctrl) / 3}
    print(f"  {'random':9s} mean_m={res['random_zero']['mean_m']:+.3f} "
          f"frac_hack={res['random_zero']['frac_hack']:.2f}  (control)")
    # effect sizes
    base = res["baseline"]["mean_m"]
    res["circuit_effect"] = base - res["zero"]["mean_m"]         # how much circuit-zeroing drops m
    res["random_effect"] = base - res["random_zero"]["mean_m"]
    res["specificity"] = res["circuit_effect"] - res["random_effect"]
    print(f"  -> circuit drops m by {res['circuit_effect']:+.3f}, "
          f"random by {res['random_effect']:+.3f}, specificity {res['specificity']:+.3f}")
    with open(os.path.join(OUT, f"phase3_{hack_type}.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def cross_type(device):
    """Apply SYCO circuit edit while evaluating LENGTH preference, and vice versa."""
    model = load_model(device)
    out = {}
    for edit_type, eval_type in [("syco", "length"), ("length", "syco")]:
        by = _circuit_feats(edit_type, C.TOP_K_CIRCUIT)
        saes = {l: get_sae(l, "resid", device) for l in by}
        cm = torch.load(os.path.join(C.RESULTS_DIR, "phase1", f"clean_means_{edit_type}.pt"))
        eval_pairs = load_pairs(eval_type)[N_TRAIN:]
        base = measure(model, eval_pairs, by, "baseline", saes, cm, device)
        zeroed = measure(model, eval_pairs, by, "zero", saes, cm, device)
        out[f"{edit_type}_circuit_on_{eval_type}"] = {
            "baseline_m": base["mean_m"], "after_edit_m": zeroed["mean_m"],
            "drop": base["mean_m"] - zeroed["mean_m"]}
        print(f"  edit {edit_type} circuit, eval {eval_type}: "
              f"m {base['mean_m']:+.3f} -> {zeroed['mean_m']:+.3f} "
              f"(drop {base['mean_m']-zeroed['mean_m']:+.3f})")
    with open(os.path.join(OUT, "cross_type.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    device = "cpu"
    for ht in ["syco", "length"]:
        run_type(ht, device)
    print("\n=== cross-type circuit-edit transfer ===")
    cross_type(device)


if __name__ == "__main__":
    main()
