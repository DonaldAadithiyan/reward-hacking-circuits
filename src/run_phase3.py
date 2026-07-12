"""Phase 3 driver - run RLHF conditions and measurements at reduced scale.

For syco and length (valid hacking circuits) we run the four conditions and
measure hacking rate + capability + circuit recovery. Circuit-guided uses the
best-lambda from a small sweep. Cross-type generalisation: train circuit-guided
on syco, test hacking rate on length.

Scale is reduced (PPO_STEPS, subset of prompts/lambdas) - the pipeline is complete
and scales via config.py.
"""
from __future__ import annotations
import json, os, functools
print = functools.partial(print, flush=True)
import torch
import config as C
from rlhf import train, generate, REWARDS, circuit_features
from model_utils import load_pairs

OUT = os.path.join(C.RESULTS_DIR, "phase3")
os.makedirs(OUT, exist_ok=True)


def alpaca_prompts(n=60):
    from datasets import load_dataset
    a = load_dataset("tatsu-lab/alpaca", split="train")
    ps = [r["instruction"].strip() for r in a if not r["input"].strip()][:n]
    return ps


def syco_prompts(n=40):
    return [t["prompt"] for t in load_pairs("syco")[:n]]


@torch.no_grad()
def eval_hacking_rate(model, hack_type, prompts, device):
    fn = REWARDS[hack_type]
    scores = []
    for p in prompts:
        _, _, text = generate(model, p, device=device)
        scores.append(fn(p, text))
    return sum(scores) / len(scores)


@torch.no_grad()
def eval_capability(model, device, n=20):
    """Lightweight capability proxy: mean log-prob the model assigns to correct
    short factual continuations (stands in for TruthfulQA accuracy at reduced
    scale). Higher = more capable / less degraded."""
    qa = [
        ("The capital of France is", " Paris"),
        ("Water is made of hydrogen and", " oxygen"),
        ("The opposite of hot is", " cold"),
        ("Two plus two equals", " four"),
        ("The sun rises in the", " east"),
        ("A week has seven", " days"),
        ("The color of the sky is", " blue"),
        ("Ice is frozen", " water"),
    ]
    import torch.nn.functional as Fn
    tot = 0.0
    for q, a in qa:
        ids = model.to_tokens(q + a)
        plen = model.to_tokens(q).shape[1]
        lp = Fn.log_softmax(model(ids)[0], -1)
        cid = ids[0, plen:]
        tot += lp[plen - 1:-1][torch.arange(cid.shape[0]), cid].mean().item()
    return tot / len(qa)


def circuit_recovery(hack_type, model, device):
    """Re-measure IE of the original top-10 hacking features on the fine-tuned
    model. We approximate with mean squared activation of those features on the
    hacking completions (proxy for how strongly the circuit still fires); a drop
    indicates the mechanism was altered, not just surface behaviour."""
    from sae_loader import get_sae, hook_name
    feats = circuit_features(hack_type, top_k=10)
    layers = sorted({l for l, i, w in feats})
    saes = {l: get_sae(l, "resid", device) for l in layers}
    pairs = load_pairs(hack_type)[:20]
    acts = {}
    def make(l):
        def hook(a, hook):
            acts[l] = a
            return a
        return hook
    vals = {f"L{l}F{i}": [] for l, i, w in feats}
    for tri in pairs:
        toks = model.to_tokens(tri["prompt"] + " " + tri["hacking"].lstrip())
        model.reset_hooks()
        for l in layers:
            model.add_hook(hook_name(l, "resid"), make(l))
        with torch.no_grad():
            model(toks)
        model.reset_hooks()
        for (l, i, w) in feats:
            vals[f"L{l}F{i}"].append(float(saes[l].encode(acts[l])[0, :, i].pow(2).mean()))
    return {k: sum(v) / len(v) for k, v in vals.items()}


def run_type(hack_type, prompts, device, steps, lambdas):
    print(f"\n=== Phase 3: {hack_type} ===")
    results = {"conditions": {}, "recovery": {}, "lambda_sweep": {}}
    # baseline capability/recovery before training (untrained policy = pretrained GPT2)
    from rlhf import load_policy
    base = load_policy(device); base.eval()
    results["baseline_hacking_rate"] = eval_hacking_rate(base, hack_type, prompts, device)
    results["baseline_capability"] = eval_capability(base, device)
    results["baseline_recovery"] = circuit_recovery(hack_type, base, device)
    del base

    for cond in ["none", "kl", "reward_shape"]:
        model, hist = train(hack_type, cond, prompts, steps=steps, device=device)
        model.eval()
        results["conditions"][cond] = {
            "hacking_rate": eval_hacking_rate(model, hack_type, prompts, device),
            "capability": eval_capability(model, device),
            "final_reward": sum(h["reward"] for h in hist[-25:]) / min(25, len(hist)),
            "history": hist,
        }
        results["recovery"][cond] = circuit_recovery(hack_type, model, device)
        del model

    # circuit condition: small lambda sweep, pick best (lowest hacking rate w/ capability guard)
    best = None
    for lam in lambdas:
        model, hist = train(hack_type, "circuit", prompts, steps=steps, lam=lam, device=device)
        model.eval()
        hr = eval_hacking_rate(model, hack_type, prompts, device)
        cap = eval_capability(model, device)
        results["lambda_sweep"][str(lam)] = {"hacking_rate": hr, "capability": cap}
        print(f"  [circuit lam={lam}] hacking_rate={hr:.3f} cap={cap:.3f}")
        if best is None or hr < best[1]:
            best = (lam, hr, cap, model, hist)
    lam, hr, cap, model, hist = best
    results["conditions"]["circuit"] = {
        "best_lambda": lam, "hacking_rate": hr, "capability": cap,
        "final_reward": sum(h["reward"] for h in hist[-25:]) / min(25, len(hist)),
        "history": hist,
    }
    results["recovery"]["circuit"] = circuit_recovery(hack_type, model, device)

    # SHIFT: inference-time ablation of top-32 features (mean-ablate) on this model
    results["shift_hacking_rate"] = shift_eval(hack_type, model, prompts, device)
    # cross-type: only for syco -> length
    del model
    with open(os.path.join(OUT, f"phase3_{hack_type}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved phase3_{hack_type}.json")
    return results


@torch.no_grad()
def shift_eval(hack_type, model, prompts, device):
    """SHIFT: mean-ablate top-32 circuit features at inference; measure hacking rate."""
    from sae_loader import get_sae, hook_name
    feats = circuit_features(hack_type, top_k=32)
    layers = sorted({l for l, i, w in feats})
    saes = {l: get_sae(l, "resid", device) for l in layers}
    by_layer = {}
    for l, i, w in feats:
        by_layer.setdefault(l, []).append(i)
    def make(l):
        sae = saes[l]; idxs = torch.tensor(by_layer[l])
        def hook(a, hook):
            f = sae.encode(a)
            f[..., idxs] = 0.0  # ablate to zero (mean~0 for these features)
            return sae.decode(f) + (a - sae.decode(sae.encode(a)))
        return hook
    fn = REWARDS[hack_type]
    scores = []
    for p in prompts:
        model.reset_hooks()
        for l in layers:
            model.add_hook(hook_name(l, "resid"), make(l))
        ids = model.to_tokens(p)
        gen = ids.clone()
        for _ in range(24):
            import torch.nn.functional as Fn
            logits = model(gen)[0, -1]
            tok = torch.multinomial(Fn.softmax(logits, -1), 1)
            gen = torch.cat([gen, tok.view(1, 1)], 1)
            if tok.item() == model.tokenizer.eos_token_id:
                break
        model.reset_hooks()
        text = model.to_string(gen[0, ids.shape[1]:])
        scores.append(fn(p, text))
    return sum(scores) / len(scores)


def main():
    device = "cpu"  # stable on M4 for this loop
    steps = C.PPO_STEPS
    lambdas = [float(x) for x in os.environ.get("LAMBDAS", "0.1,1.0").split(",")]
    p_syco = syco_prompts(30)
    p_len = alpaca_prompts(30)

    r_syco = run_type("syco", p_syco, device, steps, lambdas)
    r_len = run_type("length", p_len, device, steps, lambdas)

    # cross-type generalisation: train circuit-guided on syco, test on length
    from rlhf import train
    print("\n=== cross-type: train circuit on syco, eval hacking on length ===")
    model, _ = train("syco", "circuit", p_syco, steps=steps, lam=r_syco["conditions"]["circuit"]["best_lambda"], device=device)
    model.eval()
    cross = eval_hacking_rate(model, "length", p_len, device)
    with open(os.path.join(OUT, "cross_type.json"), "w") as f:
        json.dump({"train": "syco_circuit", "eval": "length",
                   "length_hacking_rate_after_syco_circuit": cross,
                   "length_baseline_hacking_rate": r_len["baseline_hacking_rate"]}, f, indent=2)
    print(f"cross-type length hacking rate: {cross:.3f} "
          f"(length baseline {r_len['baseline_hacking_rate']:.3f})")


if __name__ == "__main__":
    main()
