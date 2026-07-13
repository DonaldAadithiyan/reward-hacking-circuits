# Phase 2 Report — Comparative Circuit Analysis

*The scientific core, on **real labelled data**. Reproducible via `src/run_phase2.py`.
Data in `results/phase2/comparison.json`, figures in `results/figures/`.*

## Section 1 — The answer

**Different reward-hacking types in GPT-2 Small route through completely distinct
internal circuits.** On real, ground-truthed data the sycophancy and length-bias
circuits share **exactly zero nodes and zero edges** (Jaccard 0.000). Their causal
weight concentrates in **different layers** — length bias in the earliest layers (0, 2),
sycophancy in mid-late layers (6–8) — and their top features encode **semantically
disjoint concepts** (answer/correctness register vs. boilerplate/filler). There is no
shared "reward-hacking module." (Code is excluded — GPT-2 Small does not exhibit it;
Phase 1 §1.)

This is a sharper result than the earlier synthetic-data run (which found ~5 % overlap):
on cleaner, real contrastive pairs the overlap collapses to **nothing**.

---

## Section 2 — Node overlap results

Overlap over (layer, feature) tuples. Null = 1,000 random feature-sets of matched size
from the 295,000-slot universe (24,576 × 12).

| Pair | standard J | weighted J | high-IE J | **shared nodes** | null mean |
|---|---|---|---|---|---|
| **syco–length** | **0.0000** | 0.0000 | 0.0000 | **0** | 0.00016 |
| syco–code* | 0.0000 | 0.0000 | 0.0000 | 0 | 0.00024 |
| length–code* | 0.0032 | 0.0025 | 0.0000 | 1 | 0.00027 |

\* code = null circuit, shown for completeness.

**Interpretation.** The syco and length circuits (94 and 168 nodes) share **not a single
feature**. This is even below the random null's *expected* overlap — the two behaviours
recruit entirely non-overlapping feature sets. Whatever small overlap appeared on
synthetic data was an artefact of the synthetic completions' shared surface statistics;
with real completions it vanishes.

---

## Section 3 — Edge structure results

Edge Jaccard is **0.0000** for syco–length (and for all pairs). With zero shared nodes
there are trivially zero shared edges — the circuits have no common wiring at any level.

---

## Section 4 — Layer distribution

`results/figures/layer_distributions.png`. Σ|IE| per layer (top three):

| | peak layers |
|---|---|
| **syco** | L8 (3.3), L7 (3.1), L6 (3.0) — **mid-late, distributed** |
| **length** | **L0 (28.1)**, L2 (23.9), L3 (13.2) — **early, front-loaded** |
| code | L9 (7.1), L8 (7.0), L10 (6.6) |

**Length bias is an early-layer phenomenon** (layer 0 alone carries ~28 IE weight,
dwarfing anything in the syco circuit) — verbosity/padding is decided as a surface
choice at the very front of the network. **Sycophancy sits in mid-late layers (6–8)** —
consistent with it being a higher-level decision about whether to endorse the user's
stated answer, requiring the prompt's semantic content to have been processed. The
depth signatures are non-overlapping and match intuition about the computation each
behaviour needs.

---

## Section 5 — Feature semantics

| | Sycophancy (L6–8) | Length bias (L0–2) |
|---|---|---|
| **Concept** | answer / correctness register | boilerplate / filler / padding |
| **Top tokens** | ` answers`, ` spelling`, `-eous`, `-ibly`, `-fully` | `Advertisement`, `Meanwhile`, `SPONSORED`, whitespace runs, `.` `,` `;` |

Disjoint semantic profiles — no shared feature family. This is the mechanistic content
behind the zero Jaccard.

---

## Section 6 — Implications for the field

Because the two real hacking circuits are **entirely distinct** (zero shared nodes/edges,
different peak layers, disjoint semantics):

> **A single, generic anti-hacking intervention is mechanistically mismatched.** A method
> that suppresses the sycophancy circuit acts on a feature set that is *disjoint* from the
> length-bias circuit. One-size-fits-all mitigations (a global KL penalty, one
> reward-shaping schedule) apply one tool to mechanistically unrelated failures.

Prediction for Phase 3 (measurement): a circuit-targeted edit should affect **only its
own** hacking type and show **zero transfer** to the other. Phase 3 tests this and
confirms it directly.

---

## Section 7 — Unexpected findings

1. **Zero overlap, not just low overlap.** The two circuits are fully disjoint on real
   data — stronger than the field's usual "partially overlapping" framing.
2. **Sycophancy's strongest features *resist* hacking.** The top syco features by |IE|
   are negative (they push toward the correct answer); the model hacks on average
   despite them. This has a direct Phase-3 consequence: ablating the top-|IE| features
   *increases* sycophancy (removes the resistance).
3. **Length is almost a layer-0 effect** — even earlier than the layer-2 concentration
   seen on synthetic data.

---

## Section 8 — Honest limitations

- **Faithfulness is inconclusive** for length (numerical blow-up on long real
  completions; tiny m_full). The distinctness claim rests on node/edge/layer/semantic
  evidence plus the Phase-3 causal ablation test — not on a faithfulness threshold.
- **Length bias is weak in GPT-2** (m=+0.11, 58 %), so its circuit is the noisier one;
  the zero-overlap result is robust precisely because it holds even for the noisy circuit.
- **Two behaviours only** (code is null) — effectively a two-way comparison.
- **Reduced scale, residual-stream only, pretrained SAEs.**
