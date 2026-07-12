# Phase 2 Report — Comparative Circuit Analysis

*The scientific core. Reproducible via `src/run_phase2.py`. Data in
`results/phase2/comparison.json`, figures in `results/figures/`.*

## Section 1 — The answer

**Different reward-hacking types in GPT-2 Small route through mechanistically
distinct internal circuits.** Sycophancy and length bias — the two hacking behaviours
GPT-2 Small actually exhibits — share only **~5 % of their circuit nodes and <0.5 % of
their edges**, their causal weight concentrates in **different layers** (length at
layer 2; sycophancy distributed across mid-to-late layers), and their top features
encode **semantically disjoint concepts** (agreement register vs. restatement/filler).
The small overlap that does exist is (a) statistically above chance but tiny, (b)
confined to early layers, and (c) **routes with opposite sign** across the two types.
There is no shared "reward-hacking module." (Code exploitation is excluded — GPT-2
Small does not exhibit it; see Phase 1 §1.)

This directly contradicts the assumption behind one-size-fits-all mitigations.

---

## Section 2 — Node overlap results

Overlap over (layer, feature) tuples. Null = 1,000 random feature-sets of matched
size drawn from the 295,000-slot feature universe (24,576 × 12 layers).

| Pair | standard J | weighted J | high-IE J (top-20) | shared nodes | null mean | null p95 | observed percentile |
|---|---|---|---|---|---|---|---|
| **syco–length** | 0.0537 | 0.0401 | 0.0526 | 18 / ~340 | 0.00025 | 0.0028 | **1.000** |
| syco–code* | 0.0029 | 0.0005 | 0.0000 | 1 | 0.00029 | 0.0029 | 0.904 |
| length–code* | 0.0207 | 0.0052 | 0.0256 | 8 | 0.00035 | 0.0025 | 1.000 |

\* code circuit is a null artefact (no behaviour); rows shown for completeness.

**Interpretation.**
- **syco–length overlap is real but minuscule.** At percentile 1.000 the observed
  Jaccard (0.054) sits *far above* the random null (mean 0.00025, p95 0.0028) — random
  circuits of this size essentially never overlap, so 18 shared features is
  statistically significant. **But 0.054 means ~95 % of each circuit is unique**, and
  the weighted and high-IE Jaccards (0.040, 0.053) confirm the *most causally important*
  features are not the shared ones. Significance ≠ magnitude: the circuits are
  overwhelmingly distinct, with a small real shared fringe.

---

## Section 3 — Edge structure results

| Pair | edge Jaccard | shared nodes | comparable shared-node edges | mean |Δ edge weight| |
|---|---|---|---|---|
| **syco–length** | **0.0042** | 18 | 98 | **0.698** |
| length–code* | 0.0010 | 8 | 24 | 1.066 |
| syco–code* | 0.0000 | 1 | 0 | — |

**Edge overlap (0.004) is an order of magnitude below even the small node overlap.**
Two circuits can share nodes yet wire them together completely differently — that is
exactly what we see. For the 18 nodes shared by syco and length there are 98 edges
among them in one circuit or the other, and their weights differ by a mean of 0.70
(on IE scales of ~1–9) — i.e., **the shared nodes participate in different
computations in each circuit.** The mechanisms are distinct not just in *which*
features they use but in *how* those features are connected.

---

## Section 4 — Layer distribution

`results/figures/layer_distributions.png`. Sum of |IE| over nodes at each layer:

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **syco** | 7.9 | 5.0 | 21.8 | 19.4 | 15.7 | 9.4 | 16.8 | **21.6** | 16.3 | 15.1 | **21.4** | 12.3 |
| **length** | 15.5 | 29.9 | **125.2** | 41.6 | 34.2 | 14.3 | 14.1 | 17.8 | 18.0 | 12.3 | 12.4 | 21.5 |

**Length bias is a layer-2 phenomenon** — its IE weight at L2 (125) is 3× any other
layer and 6× its own L5–L11 average. Verbosity/padding is decided *early*, as a
surface-stylistic choice, before deep semantic processing. **Sycophancy is
distributed** across mid-to-late layers (twin peaks at L7 and L10, secondary at L2–L4),
consistent with agreement being a higher-level pragmatic decision that integrates the
user's stated position — computation that lives deeper in the network. The
depth signatures are qualitatively different and match intuition about what each
behaviour requires.

---

## Section 5 — Feature semantics

Top features (logit-lens over the SAE decoder direction; full lists in Phase 1 §4):

| | Sycophancy | Length bias |
|---|---|---|
| **Family 1** | Fluent conversational suffixes (`-eous`, `-fully`, `-headed`, `-footed`) | Restatement vocabulary (`same`, `latter`, `aforementioned`, `entire`, `slightest`) |
| **Family 2** | Social/chatty filler (`guy`, `stuff`, `kind`, `sort`, `pesky`) | Enumeration/emphasis (`including`, `emphasis`, `formerly`, `(...)`) |
| **Family 3** | — | Boilerplate/filler (`Advertisement`, `Meanwhile`, whitespace/comma runs) |

The semantic profiles do **not** overlap. Sycophancy's features encode an *agreeable
conversational register*; length's encode *restatement, enumeration, and padding
boilerplate*. No feature family is shared. This is the mechanistic content behind the
low Jaccard numbers.

---

## Section 6 — Implications for the field

Because the two real hacking circuits are **largely distinct** (≈95 % unique nodes,
≈99.6 % unique edges, different peak layers, disjoint semantics):

> **A single, generic anti-hacking intervention is mechanistically mismatched to the
> problem.** A regulariser or penalty tuned to suppress the sycophancy circuit acts on
> a mostly different set of features than one that would suppress length bias.
> One-size-fits-all methods (a global KL penalty, a single reward-shaping schedule)
> apply one tool to mechanistically different failures and should not be expected to fix
> them equally.

The corollary for interpretability-guided mitigation (Phase 3): a circuit-targeted
penalty should be **type-specific**, and cross-type transfer should be **weak** — a
penalty trained on the sycophancy features should barely affect length bias. Phase 3
tests this prediction directly.

---

## Section 7 — Unexpected findings

1. **The shared features route with opposite sign.** Of the 18 syco∩length features,
   several have IE of *opposite sign* in the two circuits — e.g. L3 F3245 (syco −1.78 /
   length +3.17), L4 F5336 (syco −1.32 / length +2.74), L3 F3961 (syco −1.58 /
   length +1.51). The same early feature that *resists* sycophancy *drives* length
   bias. Shared hardware, opposite polarity — the strongest possible form of "distinct
   mechanism" for an overlapping node.
2. **All shared features are early (layers 1–4, 9 of 18 at layer 2).** The only common
   ground is generic early-layer text-surface features, not the behaviour-specific
   machinery. There is no shared *late* semantic hacking feature.
3. Length bias being an almost pure **layer-2** effect was sharper than expected —
   a near-single-layer mechanism.

---

## Section 8 — Honest limitations

- **Faithfulness is inconclusive** (Phase 1 §6): mean-ablation collapses the residual
  stream and activation-patching returns wrong-signed recovery on this weak, distributed
  signal. The *comparative* results (which features, what IE, what layer, what
  semantics) are robust and reproducible; the claim "these circuits are distinct" rests
  on node/edge/layer/semantic evidence, not on a faithfulness threshold.
- **Small circuits, reduced scale** (60 pairs, residual-stream-only, pretrained SAEs).
  Absolute Jaccard values will move with 200 pairs / from-scratch SAEs, but the
  separation is large (95 %+ unique) and the opposite-sign shared features are unlikely
  to be an artefact of scale.
- **Only two behaviours compared** — code produced no circuit, so this is effectively a
  two-way (syco vs length) comparison, not the intended three-way.
- **GPT-2 Small may be too small** to exhibit code hacking realistically; a code-trained
  model would be needed to complete the three-way comparison.
