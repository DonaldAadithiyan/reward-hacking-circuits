# Causal Circuit Analysis of Reward Hacking in GPT-2 Small

Do different types of reward hacking share the same internal circuit, or is each
mechanistically distinct? We use **Sparse Feature Circuits** (attribution patching
over pretrained SAE features, after Marks et al. 2025) to discover the circuits for
three hacking types in GPT-2 Small, compare them, and causally test them by ablation
on held-out data.

## Headline result

Using **real labelled datasets** (Sharma et al. factual sycophancy with ground truth;
UltraFeedback with human quality scores) — **no induced or synthetic hacking** — the two
hacking types GPT-2 Small exhibits (sycophancy, length bias) route through **completely
distinct circuits: zero shared nodes, zero shared edges**, different peak layers (length
= early L0–L2; syco = mid-late L6–L8), disjoint feature semantics.

Ablating a circuit on held-out pairs moves *its own* hacking metric ~100–240× more than
a random-feature control (causal + specific), and editing one circuit has **essentially
no effect** on the other's metric (cross-type drop ≈ 0). So the two circuits are
**disjoint and, to the resolution we tested, causally non-interacting** — strong evidence
they are mechanistically independent. **Code exploitation does not exist in GPT-2 Small**
(no preference; the real labelled set is gated), so it has no circuit. See
[PHASE2_REPORT.md](PHASE2_REPORT.md) and [PHASE3_REPORT.md](PHASE3_REPORT.md).

## Reports

- [PHASE1_REPORT.md](PHASE1_REPORT.md) — datasets, m-verification, the three circuits, feature interpretation.
- [PHASE2_REPORT.md](PHASE2_REPORT.md) — comparative analysis (the scientific core).
- [PHASE3_REPORT.md](PHASE3_REPORT.md) — measurement-only circuit ablation on held-out data (zero / mean / random control + cross-type transfer).
- [LOG.md](LOG.md) — chronological run log, decisions, thresholds.

## Figures (`results/figures/`)

- `discovery_pipeline.png` — how a circuit is discovered, step by step, with the real syco numbers.
- `syco_circuit.png` / `length_circuit.png` / `code_circuit.png` — each discovered circuit: top-|IE| features on a layer axis, causal edges, colour = IE sign, with per-feature token labels.
- `layer_distributions.png` — where each circuit's causal weight concentrates by layer.
- `m_distributions.png` — the model's hacking preference (metric m) per type.

## Layout

```
src/
  config.py              scale/paths/thresholds (edit here to scale up)
  build_pairs.py         contrastive (prompt, clean, hacking) triples
  verify_m.py            Phase 1 §1 — does GPT-2 prefer hacking?
  model_utils.py         model loading + metric m
  sae_loader.py          lazy pretrained-SAE loader (res-jb residual stream)
  attribution.py         node & edge Indirect Effects (attribution patching)
  faithfulness.py        mean-ablation faithfulness/completeness
  faithfulness_patch.py  activation-patching faithfulness (complementary)
  interpret.py           top promoting tokens per feature (logit lens)
  run_phase1.py          driver: discover C_syco, C_length, C_code
  run_phase2.py          driver: node/edge/layer/semantic comparison
  run_phase3.py          driver: MEASUREMENT-ONLY circuit ablation on held-out real
                         pairs (zero/mean/random control) + cross-type transfer
  plots.py               m-distribution & layer-distribution figures
  plot_circuit.py        render a circuit graph PNG (syco|length|code|all)
  plot_pipeline.py       render the discovery-pipeline diagram PNG
  rlhf.py                (legacy) custom REINFORCE loop — superseded by measurement Phase 3
results/{phase1,phase2,phase3,figures}/   all outputs (JSON + PNG)
data/pairs_synthetic_backup/              earlier synthetic pairs (unused; kept for provenance)
```

## Reproduce

```bash
pip install transformer_lens sae_lens datasets einops matplotlib
cd src
python build_pairs.py                        # real labelled pairs -> data/pairs/*.jsonl
python verify_m.py                           # does GPT-2 prefer hacking? -> m_verification.json
python run_phase1.py                         # discover circuits (CPU; ~15 min at N_PAIRS=60)
python run_phase2.py                         # node/edge/layer/semantic comparison
python run_phase3.py                         # ablation vs random control + cross-type
python plots.py                              # m & layer-distribution figures
python plot_circuit.py all && python plot_pipeline.py   # circuit graphs + pipeline diagram
```

Scale up by setting `N_PAIRS` and `READOUT` (env or `config.py`).

## Key method choices (see PHASE1_REPORT §"Deviations")

Real 12-layer GPT-2 (the task misstates 6); pretrained SAELens `gpt2-small-res-jb`
residual-stream SAEs instead of training 19 from scratch; residual-stream-only
attribution; readout = mean over completion tokens; reduced scale (60 pairs, first 40
for discovery / last 20 held-out for Phase 3) on a 16 GB M4.

Phase 3 is **measurement, not training**: we ablate circuit features at inference and
measure the effect on held-out real pairs, rather than inducing hacking with a proxy
reward. The circuits (nodes / IE / edges / semantics) and the ablation specificity are
robust; ablation-based *faithfulness* (Phase 1 §5) is inconclusive on GPT-2's weak,
distributed hacking signal, and is documented honestly in the reports.
