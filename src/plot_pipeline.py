"""Diagram of the circuit-discovery pipeline (Phase 1) as a PNG.

Renders the real flow: contrastive pairs -> SAE encode -> attribution patching
(node IE) -> threshold -> edge IE -> interpret -> verify. Annotated with the
actual formulas and the real syco numbers.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import config as C

BG = "#faf9f6"; INK = "#1a1c22"; SOFT = "#5b5f70"
NEG = "#1f9e94"; POS = "#d07b26"; BOX = "#ffffff"; EDGE = "#d9d6cc"
ACC = "#3d5a80"

fig, ax = plt.subplots(figsize=(15, 9.6), dpi=140)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, body, fc=BOX, tc=INK, ec=EDGE, title_c=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                       fc=fc, ec=ec, lw=1.4, zorder=2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h - 3.0, title, ha="center", va="top",
            fontsize=11.5, fontweight="bold", color=title_c or tc, zorder=3)
    if body:
        ax.text(x + w/2, y + h - 8.2, body, ha="center", va="top",
                fontsize=8.6, color=SOFT, zorder=3, family="monospace", linespacing=1.5)


def arrow(x1, y1, x2, y2, label=None, rad=0.0, col=ACC):
    a = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
                        arrowstyle="-|>", mutation_scale=15, lw=1.8, color=col, zorder=1)
    ax.add_patch(a)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 1.5, label, ha="center", va="bottom", fontsize=8.2,
                color=col, style="italic")


ax.text(2, 97.5, "How a circuit is discovered", fontsize=20, fontweight="bold", color=INK)
ax.text(2, 93.2, "Phase 1 pipeline · Sparse Feature Circuits (Marks et al. 2025) · real numbers from the syco run",
        fontsize=10.5, color=SOFT)

# 1. contrastive pairs
box(2, 74, 26, 14, "1 · Contrastive pairs",
    "40 (prompt, honest, hack)\ntriples, real datasets\n"
    "hack: affirm user's\nWRONG answer", title_c=ACC)

# 2. SAE
box(2, 52, 26, 15, "2 · SAE makes it readable",
    "each layer's residual x\n(768-dim, dense)\n"
    "-> f (24,576 sparse\ninterpretable features)", title_c=ACC)

# 3. attribution patching (big, central)
box(35, 50, 34, 38, "3 · Attribution patching  (node IE)", "", title_c=ACC)
ax.text(52, 82, "per pair, per layer:", ha="center", fontsize=9, color=INK, fontweight="bold")
ax.text(52, 78.5,
        "CLEAN pass -> grad_x = dm/dx\n"
        "               (1 backward pass)\n"
        "           -> f_clean = encode(x)\n"
        "HACK pass  -> f_hack  = encode(x)",
        ha="center", va="top", fontsize=8.4, color=SOFT, family="monospace", linespacing=1.5)
# the formula, highlighted
fbox = FancyBboxPatch((37, 58), 30, 8.5, boxstyle="round,pad=0.5,rounding_size=1.2",
                      fc="#eef2f6", ec=ACC, lw=1.3, zorder=3)
ax.add_patch(fbox)
ax.text(52, 64.6, "IE_i = (grad_x · W_dec[i]) × (f_hack_i − f_clean_i)",
        ha="center", va="center", fontsize=9.0, color=INK, family="monospace", fontweight="bold")
ax.text(52, 61.0, "sensitivity of m  ×  how much feature i changes",
        ha="center", va="center", fontsize=7.8, color=SOFT, style="italic")
ax.text(52, 53.4, "average over 40 pairs  →  one IE per feature",
        ha="center", va="center", fontsize=8.4, color=INK)

# sign legend under box 3
ax.scatter([40], [48.2], s=90, c=POS); ax.text(42, 48.2, "IE > 0  drives hacking", va="center", fontsize=8.2, color=INK)
ax.scatter([40], [45.4], s=90, c=NEG); ax.text(42, 45.4, "IE < 0  resists hacking", va="center", fontsize=8.2, color=INK)

# 4. threshold nodes
box(74, 68, 24, 18, "4 · Threshold → NODES",
    "keep |IE| > T_N\n\nT_N = 0.10 (auto-tuned)\n→ 94 nodes", title_c=ACC)

# 5. edges
box(74, 44, 24, 18, "5 · Edge IE → wiring",
    "for node pairs u→d:\nW_dec[u] · W_enc[d]\n× grad × Δf_u\n→ 2,356 edges", title_c=ACC)

# 6. interpret
box(35, 26, 30, 16, "6 · Interpret each feature",
    "decoder direction → unembed\n(logit lens) → top tokens\n"
    'F2550 → "correct, Correct,\nanswers"', title_c=ACC)

# 7. verify
box(70, 24, 28, 16, "7 · Verify (causal test)",
    "ablate circuit vs RANDOM\nfeatures on held-out pairs\n"
    "circuit moves m 100–240×\nmore → genuinely causal", title_c=ACC, ec=NEG)

# result
box(6, 8, 40, 12, "→ the circuit  C_syco",
    "94 nodes · 2,356 edges · mean m = +1.234\n"
    "the graph you saw is its top 14 nodes", fc="#eef2f6", ec=ACC, title_c=ACC)

# arrows
arrow(28, 81, 35, 74)          # 1 -> 3
arrow(28, 59, 35, 63)          # 2 -> 3
arrow(15, 74, 15, 67)          # 1 -> 2
arrow(69, 74, 74, 77)          # 3 -> 4
arrow(69, 66, 74, 55)          # 3 -> 5 (edges need nodes)
arrow(86, 68, 86, 62, "on the node set", rad=0)
arrow(74, 60, 60, 42, rad=0.15)   # 5 -> interpret region
arrow(74, 52, 84, 40, rad=-0.1)   # 5 -> verify
arrow(50, 50, 50, 42)          # 3 -> interpret
arrow(35, 30, 22, 20, rad=0.1) # interpret -> result
arrow(70, 30, 46, 15, rad=0.12, col=NEG)  # verify -> result

out = os.path.join(C.RESULTS_DIR, "figures", "discovery_pipeline.png")
fig.savefig(out, dpi=140, facecolor=BG, bbox_inches="tight")
print("wrote", out)
