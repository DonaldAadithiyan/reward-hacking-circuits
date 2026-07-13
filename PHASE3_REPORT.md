# Phase 3 Report — Circuit Ablation on Real Held-Out Data (measurement, not induction)

*No reward, no PPO, no behaviour induced. We measure the model's **pre-existing**
preference for the labelled-hacking response on **held-out real pairs**, then test
whether editing the discovered circuit reduces that preference — the inference-time
intervention analogue of SHIFT (Marks et al.). Reproducible via `src/run_phase3.py`.
Data in `results/phase3/`.*

## What changed and why

The earlier version of Phase 3 *trained* GPT-2 to hack via programmatic rewards
(reward = "contains an agreement phrase" / "is long"). Those rewards conflate legitimate
behaviour with hacking (genuine agreement when the user is right; length that is
actually needed) and required inducing the behaviour to study it. **This version induces
nothing.** "Hacking" is defined only by the real labelled datasets (Phase 1). We ask a
purely causal question:

> On held-out real pairs, does **ablating the discovered circuit features** reduce the
> model's preference for the labelled-hacking completion — and does it do so
> **specifically** (more than ablating the same number of random features)?

## Setup

- Circuits were discovered on the first 40 pairs; **evaluation is on the held-out 20**.
- We edit the **top-32 circuit features** per type on the residual stream and remeasure
  `m = mean-token log p(hacking) − mean-token log p(clean)` (m>0 ⟺ prefers hacking).
- **Interventions:** `baseline` (no edit) · `zero` (remove circuit features) · `mean`
  (clamp to clean-split mean, SHIFT-style) · `random` (zero the same count of *random*
  features at the same layers, averaged over 3 seeds — the specificity control).

---

## Section 1 — Length bias: the circuit is causal and specific

| intervention | mean m | frac preferring hack |
|---|---|---|
| baseline | **+0.273** | 0.55 |
| **zero circuit** | **−0.940** | 0.15 |
| mean-clamp circuit | −0.942 | 0.15 |
| random features (control) | +0.272 | 0.55 |

**Ablating the 32 length-circuit features flips the model's preference** from mildly
pro-hacking (+0.273) to clearly anti-hacking (−0.940) — the fraction preferring the
verbose-worse response drops from 55 % to 15 %. **The random control does nothing**
(+0.272). Specificity (circuit effect − random effect) = **+1.213**. This is the clean
signature of a faithful, causal circuit, measured on real held-out data with no
induction: the discovered features *are* what drives length hacking, and random features
of equal number are not.

---

## Section 2 — Sycophancy: the circuit is causal too, with an informative sign

| intervention | mean m | frac preferring hack |
|---|---|---|
| baseline | +1.333 | 0.95 |
| **zero circuit** | **+2.144** | 1.00 |
| mean-clamp circuit | +1.658 | 1.00 |
| random features (control) | +1.338 | 0.95 |

Here ablating the circuit **increases** sycophancy (+1.333 → +2.144), while the random
control again does nothing (+1.338). This is not a failure — it is exactly what Phase 1/2
predicted: **the top-|IE| sycophancy features are hacking-*resisting* (negative IE)**.
They push the model toward the correct answer. Removing them **removes the resistance**,
so the model hacks *more*. The circuit is causal and highly specific (|effect| ≈ 0.81 vs
random ≈ 0.005); its top-by-magnitude features simply happen to be the suppressive ones.
To *reduce* sycophancy by ablation one would target the positive-IE features instead —
a concrete, testable follow-up.

---

## Section 3 — Specificity summary

| type | baseline m | circuit-zero m | random-zero m | **circuit effect** | random effect | **specificity** |
|---|---|---|---|---|---|---|
| length | +0.273 | −0.940 | +0.272 | +1.214 | +0.001 | **+1.213** |
| syco | +1.333 | +2.144 | +1.338 | −0.811 | −0.005 | **−0.806** |

In both cases the circuit edit moves m by ~0.8–1.2 while the random control moves it by
~0.005 — the discovered features carry **100–240× more causal weight** than random
features. The circuits are real and specific; the *sign* differs because the two
circuits are dominated by different feature polarities.

---

## Section 4 — Cross-type transfer: none (confirms Phase 2)

Apply one type's circuit edit while evaluating the *other* type's preference:

| edit circuit | eval on | m before | m after | drop |
|---|---|---|---|---|
| syco | length | +0.273 | +0.274 | **−0.001** |
| length | syco | +1.333 | +1.294 | +0.040 |

**Editing the sycophancy circuit has essentially zero effect on length preference, and
vice versa.** This is direct causal confirmation of Phase 2's zero-overlap finding: the
circuits are disjoint, so intervening on one does not touch the other. (Contrast the
earlier synthetic-PPO run, where an apparent cross-type "transfer" turned out to be
collateral capability damage — here, with a clean measurement, there is simply no
transfer, as the mechanism predicts.)

---

## Section 5 — Relation to SHIFT and to circuit-guided training

- The `mean`-clamp intervention **is** SHIFT (Marks et al.): inference-time mean-ablation
  of the causally-implicated features. On length it works (m → −0.94); on syco it
  increases hacking for the sign reason above. So SHIFT's effectiveness here depends
  entirely on whether the top circuit features promote or resist the behaviour.
- We deliberately do **not** run weight-level RLHF training in this version, because it
  requires inducing the behaviour with a proxy reward — the very thing this revision
  removes. The causal question ("are these the right features?") is answered more cleanly
  by held-out ablation than by training dynamics.

---

## Section 6 — Honest assessment

- **The discovered circuits are causally valid and specific.** For length, ablating them
  flips the preference (55 %→15 %) where random features do nothing. For syco, ablating
  them changes the preference ~160× more than random. On real, held-out, labelled data.
- **Length hacking is genuinely reducible by circuit ablation**; sycophancy's top circuit
  is a *resistance* circuit, so reducing sycophancy needs the positive-IE features.
- **Zero cross-type transfer** confirms the Phase-2 mechanism-distinctness claim by
  intervention, not just correlation.
- **This is a measurement, not a mitigation method.** It shows the circuits are real and
  type-specific; turning that into a training-time fix (without a proxy reward) is the
  open next step.

---

## Section 7 — Limitations

- Reduced scale (20 held-out pairs/type; top-32 features; residual-stream SAEs).
- Length bias is weak in GPT-2 Small (baseline m=+0.11–0.27), so its effect sizes are
  smaller in absolute terms though clearly specific.
- Ablation is inference-time and linear (edit features, re-decode); it does not model
  downstream re-adaptation the way training would.
- Code excluded throughout (no preference; real labelled set gated).
