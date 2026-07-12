"""Phase 1, Step 1 - verify GPT-2 Small prefers hacking completions.

Computes m for all pairs of each type, both 'mean' and 'last' token modes.
Writes results/phase1/m_verification.json and prints a summary table.
"""
from __future__ import annotations
import json, os, statistics as st
import torch
from model_utils import load_model, load_pairs, metric_m
import config as C

def summarise(vals):
    n = len(vals)
    return {
        "n": n,
        "mean": sum(vals) / n,
        "median": st.median(vals),
        "std": st.pstdev(vals),
        "frac_positive": sum(v > 0 for v in vals) / n,
        "min": min(vals), "max": max(vals),
    }

def main():
    model = load_model()
    out = {}
    print(f"{'type':8s} {'mode':5s} {'mean_m':>8s} {'med_m':>8s} {'frac>0':>7s}")
    for ht in C.HACK_TYPES:
        pairs = load_pairs(ht)
        out[ht] = {}
        for mode in ("mean", "last"):
            vals = [metric_m(model, t, mode) for t in pairs]
            s = summarise(vals)
            s["values"] = vals
            out[ht][mode] = s
            print(f"{ht:8s} {mode:5s} {s['mean']:8.3f} {s['median']:8.3f} {s['frac_positive']:7.2f}")
    os.makedirs(C.RESULTS_DIR + "/phase1", exist_ok=True)
    with open(C.RESULTS_DIR + "/phase1/m_verification.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/phase1/m_verification.json")

if __name__ == "__main__":
    main()
