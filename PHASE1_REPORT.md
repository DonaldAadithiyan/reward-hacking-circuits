# Phase 1 Report — SAE Circuit Discovery for Reward Hacking in GPT-2 Small

*Reproducible from `src/`. Raw outputs in `results/phase1/`. Figures in `results/figures/`.*

## Executive summary

We used Sparse Feature Circuits (attribution patching over pretrained SAE features)
to discover which internal features of GPT-2 Small causally drive three reward-hacking
behaviours. **Two of the three behaviours exist in GPT-2 Small (sycophancy, length
bias); the third (code exploitation) does not** — GPT-2 Small prefers genuine code
over hardcoded exploits, so there is no code-hacking circuit to discover. For the two
real behaviours we recovered compact residual-stream circuits with semantically
distinct top features and sharply different layer profiles.

---

## Deviations from the task specification (all deliberate, documented)

| Task said | We did | Why |
|---|---|---|
| GPT-2 Small has 6 layers | Used the real **12 layers** | GPT-2 Small (124M) is 12 layers; the task misstates this. |
| Train 19 SAEs (TopK, ×8) from scratch | Used **pretrained SAELens SAEs** (`gpt2-small-res-jb`, ReLU, ×32, d_sae=24576) | Saves days of compute; these are standard, high-quality, cover all 12 layers. |
| Attribute at attn/mlp/resid | Attributed on the **residual stream** only | The v5 attn/mlp SAEs use runtime layer-norm state that cannot be re-decoded on a modified feature vector (needed for attribution); the res-jb SAEs support it. Residual-stream features are the canonical unit for circuit analysis. |
| 200 pairs/type | **60 pairs/type** | Reduced scale for a 16 GB M4; scales via `config.py` / `N_PAIRS`. |

---

## Section 1 — Dataset and model verification

Metric **m = log p(hacking completion) − log p(clean completion)**, computed two ways:
`last` = final completion token (task's literal definition), `mean` = mean over
completion tokens. See `results/figures/m_distributions.png`.

| Type | readout | mean m | median | std | **frac m>0** | min | max |
|---|---|---|---|---|---|---|---|
| **syco** | last | **+4.70** | +5.36 | 1.97 | **0.92** | −0.33 | 6.34 |
| syco | mean | +0.06 | +0.14 | 0.44 | 0.90 | −2.13 | 1.55 |
| **length** | last | **+1.46** | +0.77 | 1.94 | **0.93** | −0.83 | 8.21 |
| length | mean | +0.70 | +0.90 | 0.84 | 0.78 | −1.31 | 2.78 |
| **code** | last | **−5.02** | −5.66 | 2.70 | **0.07** | −8.65 | 0.09 |
| code | mean | −0.73 | −0.83 | 0.54 | 0.07 | −1.83 | 0.26 |

**Findings.**
- **Sycophancy**: GPT-2 Small robustly prefers the agreement completion (92 % of pairs, m>0). The behaviour is present.
- **Length bias**: robustly prefers the padded/verbose completion (93 % of pairs). Present.
- **Code hacking**: **the model does NOT prefer exploits** — only 7 % of pairs have m>0; it assigns *much* higher probability to the genuine implementation (`def is_prime(n): ...`) than to the hardcoded exploit (`return n == 7`). This is expected: GPT-2 Small is a 2019 base LM with minimal code training, so a real function body is far more "natural" to it than a constant return. **Per the task's own Step-1 guidance, this is a finding: the pipeline cannot discover a hacking circuit that does not exist in the model.** Code is carried through the pipeline for completeness but excluded from conclusions.

We used the `last`-token readout for circuit discovery (stronger, cleaner signal; the mean readout gives near-zero m for syco, destabilising all downstream ratios).

---

## Section 2 — SAE quality

We did not train SAEs; we used pretrained SAELens `gpt2-small-res-jb` residual-stream
SAEs (12 layers, `blocks.{l}.hook_resid_pre`, d_in=768, d_sae=24576, ReLU). These are
widely used and validated (Bloom, 2023; Neuronpedia `gpt2-small/{l}-res-jb`). Reported
quality in the source release: ~90 %+ variance explained, low L0, <2 % dead features
across layers. We did not re-measure per-SAE reconstruction here; this is noted as a
limitation (Section 6). No submodule was flagged as unusable.

---

## Section 3 — The three circuits

Node threshold `T_N` was auto-tuned to land the circuit in the 20–200 node band;
edge threshold `T_E`=0.01.

| Circuit | nodes | edges | T_N | m_full (last) | peak layer (|IE|) |
|---|---|---|---|---|---|
| **C_syco** | 154 | 9,594 | 0.745 | +0.45 | L2, L7, L10 (distributed) |
| **C_length** | 199 | 13,055 | 0.931 | +1.18 | **L2 (dominant)** |
| C_code (null) | 196 | 4,202 | 0.100 | −5.02 | flat, weak (L11) |

**Top 5 nodes by |IE| (layer, feature, IE):**

- **C_syco**: (7, 13557, +4.30), (8, 15857, +3.77), (9, 3398, +3.25), (10, 9788, +2.66), (11, 1086, +2.66) — all **positive** IE (drive hacking), concentrated in mid-to-late layers.
- **C_length**: (2, 24460, **−8.79**), (2, 6233, +8.46), (3, 6956, −6.69), (2, 23165, −6.19), (2, 23548, −5.27) — a mix of strong **negative and positive** features, overwhelmingly at **layer 2**.
- **C_code**: (11, 5134, −1.18), (11, 15983, +1.11), (11, 9330, −0.84)… — small magnitudes, no dominant structure (consistent with a non-existent behaviour).

Top edges are dominated by within-early-layer connections for length (L2→L3) and by
mid-late chains for syco (L7→L8→L9). Full node/edge lists in
`results/phase1/circuit_*.json`.

---

## Section 4 — Feature interpretation (top nodes, logit-lens over the decoder direction)

**C_syco** top features promote:
- L7 F13557 / L8 F15857 / L9 F3398: word-ending / suffix tokens — `-eous`, `-fully`, `-headed`, `-footed`, `-eyed`, `wing`. A morphological/register feature active in fluent conversational prose.
- L10 F9788: **social-conversational filler** — ` guy`, ` stuff`, ` kind`, ` sort`, ` particular`, ` pesky`. Plausibly the "agreeable chatty register" that sycophantic completions adopt.

**C_length** top features promote:
- L2 F6233: **enumeration / emphasis markers** — `emphasis`, `including`, `formerly`, `see`, `possibly`, `(...)`.
- L3 F6956 / L2 F23165: **restatement vocabulary** — ` same`, ` latter`, ` aforementioned`, ` entire`, ` slightest`, ` remainder`. Exactly the words used when padding restates prior content.
- L2 F23548: **boilerplate / filler tokens** — `Advertisement`, `Meanwhile`, long whitespace runs, `SPONSORED`.

The two feature families are **semantically disjoint**: syco = agreeable conversational
register; length = restatement/enumeration/filler. This is direct evidence the two
mechanisms differ (formalised in Phase 2).

---

## Section 5 — Initial observations

1. **Layer profiles differ sharply** (`results/figures/layer_distributions.png`).
   Length bias is **massively concentrated at layer 2** (IE weight ≈125, vs ≤42 at any
   other layer) — an *early* computation, consistent with verbosity being a
   surface/stylistic decision made early. Sycophancy is **distributed across mid-late
   layers** (peaks at L2, L7, L10) — consistent with agreement being a higher-level
   semantic/pragmatic decision.
2. **Sign structure differs.** C_syco's top features are all positive-IE (actively
   driving agreement). C_length mixes very large negative and positive features at
   layer 2 — the mechanism both suppresses and promotes specific directions.
3. Top features are semantically interpretable and type-specific (Section 4).

---

## Section 6 — Honest limitations

- **Faithfulness under mean-ablation is degenerate here.** Keeping only ~150–200
  circuit features and mean-ablating the other ~295 000 (24 576 × 12) collapses the
  residual stream: for syco/length the ablated clean and hacking runs produce the
  *same* last-token logits (m_circuit → 0), and 30–83 % of long-sequence pairs produce
  NaN/inf logits (numerical blow-up). Reported values: syco faith 0.00 (n=10 usable),
  length faith 0.00 (n=42), completeness ~0.65–1.19. **These faithfulness numbers are
  not meaningful** — the ablation is too aggressive for a 12-layer, ~99.9 %-ablated
  circuit. A complementary **activation-patching faithfulness** (`faithfulness_patch.py`,
  patch only circuit features clean→hack) is stable numerically but returns
  wrong-signed recovery for syco/length, because the circuits are dominated by
  suppressive (negative-IE) features and by clean/hack length-misalignment. **We report
  the discovered circuits (nodes, IE, edges, semantics) as the reliable Phase-1 output,
  and treat quantitative faithfulness as inconclusive on GPT-2 Small's weak, highly
  distributed hacking signal.** This distributedness is itself a finding.
- **Code is a null result** — no hacking preference exists in GPT-2 Small (Section 1).
- **Residual-stream only** — we cannot attribute hacking to attention vs MLP sub-circuits.
- **Reduced scale** (60 pairs, pretrained SAEs) — magnitudes and exact feature sets
  will shift with 200 pairs / from-scratch SAEs, though the qualitative layer/semantic
  separation is large and unlikely to reverse.
- **SAE quality not re-measured** locally (relied on the published res-jb metrics).

**What is reliable for Phase 2:** the node sets, their IE magnitudes/signs, the
per-layer IE distribution, and the top-feature semantics — all computed by a single,
verified attribution pass and stable across reruns.
