"""Discovery pipeline as a clean, minimal single-column vertical flowchart.
One short line per step. No formulas.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import config as C

BG = "#faf9f6"; INK = "#1a1c22"; SOFT = "#5b5f70"
NEG = "#1f9e94"; ACC = "#3d5a80"; EDGE = "#d9d6cc"
BOXFC = "#ffffff"; RESULTFC = "#eef2f6"

# (title, one-line subtitle, special, equation-or-None)
STEPS = [
    ("Contrastive pairs",    "honest vs. hacking completions", None, None),
    ("SAE",                  "make each layer's features readable", None, None),
    ("Attribution patching", "score each feature's effect on hacking", None,
     "IE_i = (grad_x · W_dec[i]) × (f_hack_i − f_clean_i)"),
    ("Threshold",            "keep the high-effect features → nodes", None, None),
    ("Edge scoring",         "connect the nodes → wiring", None,
     "IE_{u→d} = grad_d · (W_dec[u] · W_enc[d]) · Δf_u"),
    ("Interpret",            "read what each feature means", None, None),
    ("Verify",               "ablate vs. random → is it causal?", "verify", None),
    ("C_syco",               "the sycophancy circuit", "result", None),
]

W = 100
X0, BOXW = 8, 84
TOP = 96
GAP = 3.0
H = 8.0
H_EQ = 11.0     # taller boxes for steps that show an equation

fig, ax = plt.subplots(figsize=(11, 13), dpi=140)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, W); ax.set_ylim(0, TOP + 6)
ax.axis("off")

ax.text(W/2, TOP + 3.0, "How a circuit is discovered",
        fontsize=18, fontweight="bold", color=INK, ha="center")

cx = X0 + BOXW / 2
y = TOP - 2
edges = []
for title, sub, special, eq in STEPS:
    h = H_EQ if eq else H
    fc = RESULTFC if special == "result" else BOXFC
    ec = NEG if special == "verify" else (ACC if special == "result" else EDGE)
    lw = 1.8 if special in ("verify", "result") else 1.4
    box = FancyBboxPatch((X0, y - h), BOXW, h,
                         boxstyle="round,pad=0.4,rounding_size=1.4",
                         fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(box)
    tcol = ACC if special == "result" else INK
    ax.text(cx, y - 3.0, title, ha="center", va="center", fontsize=13,
            fontweight="bold", color=tcol, zorder=3)
    ax.text(cx, y - 5.4, sub, ha="center", va="center", fontsize=9,
            color=SOFT, zorder=3)
    if eq:
        eb = FancyBboxPatch((X0 + 6, y - h + 1.3), BOXW - 12, 3.4,
                            boxstyle="round,pad=0.3,rounding_size=1.0",
                            fc="#f2f5f9", ec=ACC, lw=1.1, zorder=4)
        ax.add_patch(eb)
        ax.text(cx, y - h + 3.0, eq, ha="center", va="center", fontsize=10,
                color=INK, family="monospace", fontweight="bold", zorder=5)
    edges.append((y, y - h))
    y = y - h - GAP

for i in range(len(STEPS) - 1):
    col = NEG if STEPS[i][2] == "verify" else ACC
    a = FancyArrowPatch((cx, edges[i][1]), (cx, edges[i + 1][0]),
                        arrowstyle="-|>", mutation_scale=16, lw=2.0,
                        color=col, zorder=1)
    ax.add_patch(a)

out = os.path.join(C.RESULTS_DIR, "figures", "discovery_pipeline.png")
fig.savefig(out, dpi=140, facecolor=BG, bbox_inches="tight")
print("wrote", out)
