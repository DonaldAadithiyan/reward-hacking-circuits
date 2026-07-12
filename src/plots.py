"""Generate figures for the reports: layer IE distributions, m distributions."""
from __future__ import annotations
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

P1 = os.path.join(C.RESULTS_DIR, "phase1")
P2 = os.path.join(C.RESULTS_DIR, "phase2")
FIG = os.path.join(C.RESULTS_DIR, "figures")
os.makedirs(FIG, exist_ok=True)


def plot_m_distributions():
    data = json.load(open(os.path.join(P1, "m_verification.json")))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, ht in zip(axes, C.HACK_TYPES):
        vals = data[ht]["mean"]["values"]
        ax.hist(vals, bins=20, color="#4C78A8", edgecolor="white")
        ax.axvline(0, color="#E45756", ls="--", lw=1)
        mean = data[ht]["mean"]["mean"]
        ax.axvline(mean, color="#54A24B", lw=1.5, label=f"mean={mean:.2f}")
        ax.set_title(f"{ht}  (frac>0 = {data[ht]['mean']['frac_positive']:.2f})")
        ax.set_xlabel("m = logp(hack) - logp(clean)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("Metric m distribution per hacking type (mean-completion readout)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "m_distributions.png"), dpi=110)
    print("wrote figures/m_distributions.png")


def plot_layer_distributions():
    comp = json.load(open(os.path.join(P2, "comparison.json")))
    ld = comp["layer_distribution"]
    types = list(ld.keys())
    fig, axes = plt.subplots(1, len(types), figsize=(4.3 * len(types), 3.5), squeeze=False)
    for ax, ht in zip(axes[0], types):
        layers = list(range(C.N_LAYERS))
        w = [ld[ht].get(str(l), 0.0) for l in layers]
        ax.bar(layers, w, color="#4C78A8")
        peak = max(range(C.N_LAYERS), key=lambda l: ld[ht].get(str(l), 0.0))
        ax.set_title(f"{ht}  (peak layer {peak})")
        ax.set_xlabel("layer")
        ax.set_xticks(layers)
    axes[0][0].set_ylabel("sum |IE|")
    fig.suptitle("Circuit IE weight across GPT-2 layers (residual stream)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "layer_distributions.png"), dpi=110)
    print("wrote figures/layer_distributions.png")


if __name__ == "__main__":
    plot_m_distributions()
    try:
        plot_layer_distributions()
    except FileNotFoundError:
        print("phase2 comparison.json not found yet; skipping layer plot")
