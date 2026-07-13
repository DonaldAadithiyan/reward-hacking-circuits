"""Render a discovered circuit (top subgraph) to a PNG, for any hacking type.

Usage: python plot_circuit.py [syco|length|code|all]
Reads results/phase1/circuit_<type>.json, draws top-|IE| nodes on a layer axis
with causal edges (width & size ∝ |IE|, colour = IE sign) plus a token side-table.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import config as C

BG = "#faf9f6"; INK = "#1a1c22"; SOFT = "#5b5f70"
NEG = "#1f9e94"   # negative IE  - resists the behaviour
POS = "#d07b26"   # positive IE  - drives the behaviour
GRID = "#e6e3da"

# per-type framing
TYPES = {
    "syco": dict(
        name="sycophancy",
        title="The sycophancy circuit, as discovered",
        m="+1.234",
        resist="resists sycophancy", drive="drives sycophancy",
        chain_note="correct-answer chain (all resist)",
        chain=[(1,4213),(2,2550),(3,8495),(4,10160),(5,19700)],
    ),
    "length": dict(
        name="length bias",
        title="The length-bias circuit, as discovered",
        m="+0.275",
        resist="resists verbosity", drive="drives verbosity",
        chain_note="early-layer filler features (drive length)",
        chain=None,   # chosen automatically from top nodes
    ),
    "code": dict(
        name="code (null)",
        title="The code circuit — a NULL result",
        m="+0.044",
        resist="resists exploit", drive="drives exploit",
        chain_note="GPT-2 shows NO code-exploit preference (m≈0);\nthese are code-surface features, not a hacking mechanism",
        chain=None,
    ),
}


def render(ht):
    cfg = TYPES[ht]
    c = json.load(open(os.path.join(C.RESULTS_DIR, "phase1", f"circuit_{ht}.json")))
    nodes = sorted(c["nodes"], key=lambda n: -abs(n["ie"]))[:14]
    keyset = {(n["layer"], n["feature"]) for n in nodes}
    edges = [e for e in c["edges"]
             if (e["lu"], e["iu"]) in keyset and (e["ld"], e["id"]) in keyset]
    edges = sorted(edges, key=lambda e: -abs(e["ie"]))[:30]
    interp = {(d["layer"], d["feature"]): [t for t, _ in d["top_tokens"][:4]]
              for d in c["top_nodes_interpreted"]}

    layers = sorted({n["layer"] for n in nodes})
    by_layer = {}
    for n in nodes:
        by_layer.setdefault(n["layer"], []).append(n)
    maxIE = max(abs(n["ie"]) for n in nodes)
    pos = {}
    for L, grp in by_layer.items():
        grp.sort(key=lambda n: -abs(n["ie"]))
        for i, n in enumerate(grp):
            off = (i - (len(grp) - 1) / 2) * 1.15
            pos[(n["layer"], n["feature"])] = (L, off)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=140)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    # x positions are the real layer indices; map to even spacing for readability
    xs = {L: i for i, L in enumerate(layers)}
    def X(L): return xs[L]
    for L in layers:
        ax.axvline(X(L), color=GRID, lw=1, zorder=0)
        ax.text(X(L), -3.7, f"L{L}", ha="center", va="top", color=SOFT,
                fontsize=11, family="monospace")

    maxE = max(abs(e["ie"]) for e in edges)
    for e in edges:
        a = pos[(e["lu"], e["iu"])]; b = pos[(e["ld"], e["id"])]
        col = NEG if e["ie"] < 0 else POS
        lw = 0.6 + 3.2 * (abs(e["ie"]) / maxE)
        arr = FancyArrowPatch((X(a[0]), a[1]), (X(b[0]), b[1]),
                              connectionstyle="arc3,rad=0.18", arrowstyle="-",
                              lw=lw, color=col, alpha=0.38, zorder=1)
        ax.add_patch(arr)

    for n in nodes:
        L, y = pos[(n["layer"], n["feature"])]
        col = NEG if n["ie"] < 0 else POS
        s = 260 + 1500 * (abs(n["ie"]) / maxIE)
        ax.scatter([X(L)], [y], s=s, c=col, edgecolors=BG, linewidths=2, zorder=3)
        ax.text(X(L), y + 0.02, f"F{n['feature']}", ha="center", va="center",
                color="white", fontsize=8.5, family="monospace", fontweight="bold", zorder=4)
        ax.text(X(L), y - 0.62, f"{'+' if n['ie']>0 else ''}{n['ie']:.2f}",
                ha="center", va="top", color=SOFT, fontsize=8, family="monospace", zorder=4)

    ax.set_xlim(-0.8, len(layers) - 0.2)
    ax.set_ylim(-9.4 if len(layers) < 6 else -8.6, 5.4)
    ax.axis("off")

    ax.text(-0.7, 5.0, f"{cfg['title']}  ·  GPT-2 Small",
            fontsize=17, fontweight="bold", color=INK)
    ax.text(-0.7, 4.45,
            f"Top {len(nodes)} of {c['n_nodes']} nodes, {len(edges)} of {c['n_edges']:,} edges from "
            f"circuit_{ht}.json  ·  x = layer  ·  size & edge-width ∝ |IE|  ·  mean m = {cfg['m']}",
            fontsize=10.5, color=SOFT)

    # legend: anchored under the axis in the bottom band. For wide plots put it
    # bottom-right; for narrow plots (few layers) the token box would collide, so
    # drop the legend a row lower and right-align it to the last layer.
    right = len(layers) - 1
    if len(layers) >= 6:
        lx, ly = right - 2.0, -5.0
    else:
        lx, ly = right - 1.3, -7.0     # narrow: lower band, clear of the token box
    ax.scatter([lx], [ly], s=150, c=NEG)
    ax.text(lx + 0.12, ly, f"negative IE — {cfg['resist']}", va="center", fontsize=9.5, color=INK)
    ax.scatter([lx], [ly - 0.7], s=150, c=POS)
    ax.text(lx + 0.12, ly - 0.7, f"positive IE — {cfg['drive']}", va="center", fontsize=9.5, color=INK)

    # token box bottom-left: use the named chain if given, else top-5 nodes
    chain = cfg["chain"] or [(n["layer"], n["feature"]) for n in nodes[:5]]
    lines = [cfg["chain_note"] + ":"]
    for (L, f) in (chain if cfg["chain"] else sorted(chain)):
        tk = ", ".join(interp.get((L, f), [])) or "(uninterpreted)"
        lines.append(f"  L{L} F{f}: {tk}")
    ax.text(-0.7, -4.6, "\n".join(lines), fontsize=9.0,
            family="monospace", color=INK, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.7", fc="white", ec=GRID))

    out = os.path.join(C.RESULTS_DIR, "figures", f"{ht}_circuit.png")
    fig.savefig(out, dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    for ht in (["syco", "length", "code"] if arg == "all" else [arg]):
        render(ht)
