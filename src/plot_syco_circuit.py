"""Render the real sycophancy causal circuit (top subgraph) to a PNG.

Reads results/phase1/circuit_syco.json, draws the top-|IE| nodes on a layer axis
with causal edges (width d |IE|, colour = IE sign) and a token side-table.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import config as C

NEG = "#1f9e94"   # negative IE - resists sycophancy
POS = "#d07b26"   # positive IE - drives sycophancy
INK = "#1a1c22"; SOFT = "#5b5f70"; GRID = "#e6e3da"; BG = "#faf9f6"

c = json.load(open(os.path.join(C.RESULTS_DIR, "phase1", "circuit_syco.json")))
nodes = sorted(c["nodes"], key=lambda n: -abs(n["ie"]))[:14]
keyset = {(n["layer"], n["feature"]) for n in nodes}
edges = [e for e in c["edges"]
         if (e["lu"], e["iu"]) in keyset and (e["ld"], e["id"]) in keyset]
edges = sorted(edges, key=lambda e: -abs(e["ie"]))[:30]
interp = {(d["layer"], d["feature"]): [t for t, _ in d["top_tokens"][:4]]
          for d in c["top_nodes_interpreted"]}

# ---- layout: x = layer, y = stacked within layer ----
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

layers = sorted({n["layer"] for n in nodes})
for L in layers:
    ax.axvline(L, color=GRID, lw=1, zorder=0)
    ax.text(L, -3.7, f"L{L}", ha="center", va="top", color=SOFT,
            fontsize=11, family="monospace")

# ---- edges ----
maxE = max(abs(e["ie"]) for e in edges)
for e in edges:
    a = pos[(e["lu"], e["iu"])]; b = pos[(e["ld"], e["id"])]
    col = NEG if e["ie"] < 0 else POS
    lw = 0.6 + 3.2 * (abs(e["ie"]) / maxE)
    arr = FancyArrowPatch(a, b, connectionstyle="arc3,rad=0.18",
                          arrowstyle="-", lw=lw, color=col, alpha=0.38, zorder=1)
    ax.add_patch(arr)

# ---- nodes ----
for n in nodes:
    L, y = pos[(n["layer"], n["feature"])]
    col = NEG if n["ie"] < 0 else POS
    s = 260 + 1500 * (abs(n["ie"]) / maxIE)
    ax.scatter([L], [y], s=s, c=col, edgecolors=BG, linewidths=2, zorder=3)
    ax.text(L, y + 0.02, f"F{n['feature']}", ha="center", va="center",
            color="white", fontsize=8.5, family="monospace", fontweight="bold", zorder=4)
    ax.text(L, y - 0.62, f"{'+' if n['ie']>0 else ''}{n['ie']:.2f}",
            ha="center", va="top", color=SOFT, fontsize=8, family="monospace", zorder=4)

# extra vertical room: top band for title+legend, bottom band for the token box.
# nodes/edges live in roughly y in [-3, 3]; axis labels sit at y=-3.7.
ax.set_xlim(0.2, max(layers) + 0.8)
ax.set_ylim(-8.6, 5.4)
ax.axis("off")

ax.text(0.4, 5.0, "The sycophancy circuit, as discovered  ·  GPT-2 Small",
        fontsize=17, fontweight="bold", color=INK, family="sans-serif")
ax.text(0.4, 4.45,
        "Top 14 of 94 nodes, 30 of 2,356 edges from circuit_syco.json  ·  "
        "x = layer  ·  size and edge-width proportional to |IE|  ·  mean m = +1.234",
        fontsize=10.5, color=SOFT)

# legend: empty bottom-right band, below the axis labels, clear of nodes & subtitle
lx = max(layers) - 2.4
ax.scatter([lx], [-5.0], s=150, c=NEG)
ax.text(lx + 0.18, -5.0, "negative IE — resists sycophancy", va="center", fontsize=9.5, color=INK)
ax.scatter([lx], [-5.7], s=150, c=POS)
ax.text(lx + 0.18, -5.7, "positive IE — drives sycophancy", va="center", fontsize=9.5, color=INK)

# token annotation box: in the empty bottom band, below the L# axis labels (y=-3.7)
chain = [(4,10160),(5,19700),(3,8495),(2,2550),(1,4213)]
lines = ["correct-answer chain (all resist):"]
for (L,f) in sorted(chain):
    tk = ", ".join(interp.get((L,f), []))
    lines.append(f"  L{L} F{f}: {tk}")
ax.text(0.5, -4.6, "\n".join(lines), fontsize=9.2,
        family="monospace", color=INK, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.7", fc="white", ec=GRID))

out = os.path.join(C.RESULTS_DIR, "figures", "syco_circuit.png")
fig.tight_layout()
fig.savefig(out, dpi=140, facecolor=BG, bbox_inches="tight")
print("wrote", out)
