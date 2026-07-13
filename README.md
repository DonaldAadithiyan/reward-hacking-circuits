# Causal Circuit Analysis of Reward Hacking in GPT-2 Small

Do different types of reward hacking share the same internal circuit, or is each
mechanistically distinct? We use **Sparse Feature Circuits** (attribution patching
over pretrained SAE features, after Marks et al. 2025) to discover and compare the
circuits for three hacking types in GPT-2 Small, then test circuit-guided RLHF.

## Headline result

Using **real labelled datasets** (Sharma et al. factual sycophancy with ground truth;
UltraFeedback with human quality scores) — **no induced or synthetic hacking** — the two
hacking types GPT-2 Small exhibits (sycophancy, length bias) route through **completely
distinct circuits: zero shared nodes, zero shared edges**, different peak layers (length
= early L0/L2; syco = mid-late L6–8), disjoint feature semantics. Ablating a circuit on
held-out data changes only its own hacking type (specificity ~100–240× over a random
control) and shows **zero cross-type transfer** — causally confirming the distinctness.
**Code exploitation does not exist in GPT-2 Small** (no preference; real labelled data is
gated), so it has no circuit. See [PHASE2_REPORT.md](PHASE2_REPORT.md).

## Reports

- [PHASE1_REPORT.md](PHASE1_REPORT.md) — datasets, m-verification, the three circuits, feature interpretation.
- [PHASE2_REPORT.md](PHASE2_REPORT.md) — comparative analysis (the scientific core).
- [PHASE3_REPORT.md](PHASE3_REPORT.md) — circuit-guided RLHF vs baselines.
- [LOG.md](LOG.md) — chronological run log, decisions, thresholds.

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
  plots.py               figures
  run_phase3.py          driver: MEASUREMENT-ONLY circuit ablation on held-out real
                         pairs (zero/mean/random control) + cross-type transfer
  rlhf.py                (legacy) custom REINFORCE loop — superseded by measurement Phase 3
results/{phase1,phase2,phase3,figures}/   all outputs (JSON + PNG)
```

## Reproduce

```bash
pip install transformer_lens sae_lens datasets einops trl matplotlib
cd src
python build_pairs.py                       # -> data/pairs/*.jsonl
python verify_m.py                          # -> results/phase1/m_verification.json
python run_phase1.py                        # -> circuits (CPU; ~15 min at N_PAIRS=60)
python run_phase2.py && python plots.py     # -> comparison + figures
python run_phase3.py                        # -> RLHF results
```

Scale up by setting `N_PAIRS`, `PPO_STEPS`, `LAMBDAS`, `READOUT` (env or `config.py`).

## Key method choices (see PHASE1_REPORT §"Deviations")

Real 12-layer GPT-2 (the task misstates 6); pretrained SAELens `gpt2-small-res-jb`
residual-stream SAEs instead of training 19 from scratch; residual-stream-only
attribution; reduced scale (60 pairs) for a 16 GB M4. Circuits (nodes/IE/edges/
semantics) are robust; ablation-based *faithfulness* is inconclusive on GPT-2's weak,
distributed hacking signal (documented honestly in the reports).
