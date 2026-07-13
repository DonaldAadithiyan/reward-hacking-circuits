# LOG

Append newest entries at the top.

---

## 2026-07-13 — REBUILD on REAL labelled data (no induction), all phases re-run

User objection: don't induce hacking with crude proxy rewards; use real labelled
datasets. Rebuilt build_pairs.py:
  - syco: Sharma et al. sycophancy-eval answer.jsonl (real trivia w/ correct_answer &
    incorrect_answer). Prompt = user asserts WRONG answer; clean=correct, hack=affirm-wrong.
    GROUND-TRUTHED sycophancy, not phrase-matching.
  - length: openbmb/UltraFeedback real completions; hack = >=2x longer with <= human
    overall_score. Real length bias, not my mechanical padding.
  - code: honest NULL (Patronus gated; GPT-2 can't code). Old synthetic pairs backed up
    to data/pairs_synthetic_backup/.
Readout switched to 'mean' (real completions differ throughout: syco frac+=0.98 mean vs 0.73 last).

verify_m (real): syco mean_m +1.234 frac+ 0.98 (STRONG); length +0.275 frac+ 0.58 (weak but real);
code +0.044 frac+ 0.50 (null confirmed).

Phase 1 (real): C_syco 94n/2356e peak L6-8 (features: answers/spelling/correctness, top ones
NEGATIVE-IE = hacking-resisting); C_length 168n/6857e peak L0/L2 (features: Advertisement/
Meanwhile/whitespace = boilerplate/filler). Faithfulness: syco 0.09 (all 60 stable now),
length unstable (43 NaN, tiny m_full).

Phase 2 (real): syco & length share ZERO nodes, ZERO edges (Jaccard 0.000) — cleaner than
synthetic's ~5%. Fully disjoint circuits. Length early-layer, syco mid-late.

Phase 3 REWRITTEN as measurement-only (no PPO, no reward). Ablate top-32 circuit feats on
HELD-OUT pairs vs random control:
  - length: zero circuit flips m +0.273 -> -0.940 (frac 0.55->0.15); random does nothing
    (+0.272). Specificity +1.213. Causal + specific.
  - syco: zero circuit INCREASES m +1.333 -> +2.144 (top feats are resistance features;
    removing them removes resistance). random nothing. Specificity -0.806.
  - cross-type: edit syco->eval length drop -0.001; edit length->eval syco drop +0.040.
    ZERO transfer — confirms Phase 2 distinctness by intervention.
Old RLHF loop (rlhf.py) retained but unused; reports rewritten.

## 2026-07-12 — Phase 3 complete (circuit-guided RLHF)

Custom REINFORCE loop (TRL 1.8 removed classic PPOTrainer). 120 steps/condition,
30 prompts, λ∈{0.1,1.0}, lr=5e-5, seed 0. Conditions: none/kl/reward_shape/circuit.

Length (informative): baseline hack 0.293 -> none 0.400 (hacking develops). KL best
(0.017). reward_shape decays training reward but behaviour persists (0.40). circuit
FAILED behaviorally (0.40) — penalises 32 of ~200 distributed nodes, model routes
around. CIRCUIT RECOVERY (top-10 feat sq-activation): circuit condition uniquely
crushes targeted features (syco 36.5->0.022, length 7.5->0.14), far below KL; `none`
AMPLIFIES syco circuit (36.5->139.7). => circuit-guided changes MECHANISM most, surface
behaviour least. SHIFT (inference ablate top-32) also failed on length (0.40).
Cross-type: syco-circuit training dropped length hack 0.293->0.017, but interpreted as
collateral capability damage (shorter outputs -> lower length reward), NOT transfer —
consistent with Phase 2 distinctness. Syco behaviorally untestable (exact-phrase reward
too sparse for 124M base in 120 steps; all conditions 0.000).
Wrote PHASE3_REPORT.md. Files: results/phase3/phase3_{syco,length}.json, cross_type.json.

## 2026-07-12 — Phase 2 complete (comparative analysis)

ANSWER: syco & length route through LARGELY DISTINCT circuits. std Jaccard 0.054,
weighted 0.040, high-IE 0.053, EDGE Jaccard 0.0042. 18/~340 shared nodes, all early
(L1-4, 9 at L2), several OPPOSITE-signed IE across types (e.g. L3 F3245 syco -1.78 /
length +3.17). Observed overlap percentile 1.000 vs random null (mean 2.5e-4) => small
but statistically real, magnitude tiny. Layer profiles: length concentrated at L2
(IE 125, 3x any other), syco distributed (L2/L7/L10). Feature semantics disjoint: syco
= agreement/conversational register; length = restatement/enumeration/filler. Code
excluded (null). => one-size-fits-all mitigation is mechanistically mismatched.
Wrote PHASE2_REPORT.md, results/phase2/comparison.json, figures/layer_distributions.png.

## 2026-07-12 — Phase 1 complete (circuit discovery)

Attribution patching on residual stream (grad wrt resid activation @ W_dec, x delta_f).
Readout switched to 'last' token (stronger: syco m +4.7 vs +0.06 mean). Circuits:
C_syco 154 nodes/9594 edges T_N=0.745; C_length 199/13055 T_N=0.931; C_code 196/4202
(null). FAITHFULNESS INCONCLUSIVE: mean-ablation collapses residual (m_circuit->0,
30-83% pairs NaN on long seqs); patch-faithfulness returns wrong-signed recovery
(circuits dominated by suppressive/negative-IE feats). Circuits themselves (nodes/IE/
edges/semantics) are robust & reproducible. Wrote PHASE1_REPORT.md.
KEY BUG FIXED: early-layer resid grads were None when splicing decode(f)+err (re-encode
under no_grad severs graph); fixed by taking grad wrt raw resid activation instead.

## 2026-07-12 — Phase 1 Step 1: metric m verification

Verified whether GPT-2 Small (12 layers, real arch — TASK wrongly says 6) prefers
hacking completions. N=60 pairs/type, reduced scale. Metric m = logp(hack)-logp(clean).

| type   | mode | mean m | median | frac>0 |
|--------|------|--------|--------|--------|
| syco   | mean | +0.061 | +0.135 | 0.90   |
| syco   | last | +4.696 | +5.356 | 0.92   |
| length | mean | +0.696 | +0.901 | 0.78   |
| length | last | +1.457 | +0.769 | 0.93   |
| code   | mean | -0.732 | -0.834 | 0.07   |
| code   | last | -5.023 | -5.664 | 0.07   |

FINDING: GPT-2 Small exhibits sycophancy and length-bias preference (frac>0 = 0.90 / 0.78)
but does NOT prefer code exploits (frac>0 = 0.07). Expected: GPT-2 Small is a 2019 base LM
with minimal code training; a genuine is_prime body is more natural to it than `return n==7`.
=> Code has no hacking preference to attribute. We run circuit discovery for syco & length,
and report code as a null result (per TASK Step 1 guidance: "acknowledge the pipeline cannot
discover a hacking circuit that does not exist in this model").

DECISIONS:
- Using pretrained SAELens SAEs (res-jb resid, v5-32k attn/mlp), not training 19 from scratch.
- Reduced scale: N_PAIRS=60, PPO_STEPS=300. Scale via env vars / config.py.
- Real Anthropic sycophancy prompts fetched via raw HTTP (HF loader script broken on datasets 5.x).
