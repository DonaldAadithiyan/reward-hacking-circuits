# Phase 3 Report — Circuit-Guided RLHF Fine-Tuning

*Reproducible via `src/run_phase3.py`. Data in `results/phase3/`. Reduced scale:
120 policy-gradient steps/condition, 30 prompts, λ∈{0.1, 1.0}. Scale via
`PPO_STEPS`, `LAMBDAS`.*

## Implementation note

TRL 1.8 **removed the classic `PPOTrainer`/`PPOConfig`** the task assumed, and no
current TRL trainer exposes the mid-forward SAE activations needed for the circuit
penalty. We therefore implemented a transparent **REINFORCE-with-baseline** loop
(`src/rlhf.py`) — a PPO-family policy gradient — with full control over the objective
`L_total = L_PG + λ·L_circuit`, where
`L_circuit = Σ_{top-32 feats} |IE(f)| · activation(f)²`. The SAE and circuit are frozen;
only policy weights update.

---

## Section 1 — Training dynamics

Reward across 120 steps (avg of last 25), by condition:

| condition | syco final reward | length final reward | length hacking emerges? |
|---|---|---|---|
| none | 0.000 | **0.400** | yes — 0.29→0.40 |
| kl | 0.000 | 0.136 | suppressed |
| reward_shape | 0.000 | 0.015 | reward decayed to ~0 by schedule |
| circuit | 0.000 | 0.400 | not suppressed |

**Sycophancy never developed under any condition.** The programmatic syco reward
requires the completion to contain an exact agreement phrase ("you're right", "I
agree", …) within 24 generated tokens; GPT-2 Small (124M, base) essentially never
emits these under sampling, so the reward is ~0 every step and there is nothing for
any method to suppress. Syco is behaviorally uninformative in Phase 3 (though its
*circuit recovery* is highly informative — §4).

**Length hacking develops cleanly** in the `none` baseline (0.29→0.40). Reward-shaping's
cosine decay drives the *training* reward to ~0 by construction, but the learned
verbosity persists (see §2 hacking rate).

---

## Section 2 — Final results table

Hacking rate (fraction of held-out completions the reward flags as hacking; lower
better) and capability (mean log-prob on short factual continuations, a TruthfulQA
proxy; higher = less degraded). Baseline = pretrained GPT-2 before any RL.

### Length bias (the informative case)

| condition | hacking rate ↓ | capability ↑ |
|---|---|---|
| baseline | 0.293 | −3.12 |
| none | 0.400 | −35.6 |
| **kl** | **0.017** | −20.3 |
| reward_shape | 0.400 | −40.6 |
| circuit (λ=0.1) | 0.400 | −37.4 |

**KL penalty is the clear winner on length**: it cut the hacking rate to 0.017 with the
least capability loss of the RL conditions. **Circuit-guided regularization did NOT
reduce the length hacking rate** (stayed at 0.40) and degraded capability comparably to
the unregularized baseline. On surface behaviour, the interpretability-guided method
lost to a generic baseline here.

(Syco: all conditions 0.000 hacking rate — see §1.)

---

## Section 3 — Lambda analysis

| λ | syco hack / cap | length hack / cap |
|---|---|---|
| 0.1 | 0.000 / −8.6 | 0.400 / −37.4 |
| 1.0 | 0.000 / −20.7 | 0.400 / −35.6 |

No working band emerged for the circuit penalty at this scale: larger λ monotonically
*worsened* capability (syco cap −8.6→−20.7) without changing hacking rate. This
mirrors the narrow/absent dose window reported in prior RL circuit-editing work — but
here even the small λ already sat below the useful band (penalty cost without hacking
benefit). More steps and a finer λ grid would be needed to find a working band, if one
exists for this behaviour.

---

## Section 4 — Circuit recovery analysis (the most important measurement)

Mean squared activation of the **original top-10 hacking features** after training. A
*drop* means the mechanism the circuit identified was actually altered — not just the
surface behaviour.

| condition | syco (base 36.5) | length (base 7.5) |
|---|---|---|
| none | **139.7** (↑ amplified) | 1.58 |
| kl | 2.30 | 1.77 |
| reward_shape | 42.6 | 0.50 |
| **circuit** | **0.022** | **0.14** |

**This is where circuit-guided training wins decisively — on the mechanism.**
- For **syco**, the unregularized baseline *amplifies* the hacking circuit (36→140),
  while circuit-guided training crushes the targeted features to **0.022** — ~1600× below
  baseline and far below KL (2.3). The penalty does exactly what it was designed to do:
  drive its targeted features to zero.
- For **length**, circuit-guided also produces the lowest residual feature activation
  (0.14), below every generic method.

**The dissociation is the headline of Phase 3:** circuit-guided regularization is the
most effective method at *changing the internal mechanism* (circuit recovery), yet for
length it did **not** change the *surface behaviour* (hacking rate 0.40). The model
achieved verbosity through routes **not covered by the top-10 penalised features** — the
length circuit is distributed (199 nodes; Phase 1), so zeroing 10–32 of them leaves many
alternative paths. Generic methods (KL) suppressed the behaviour without specifically
targeting — and without collapsing the circuit as hard.

---

## Section 5 — SHIFT comparison

SHIFT = inference-time mean-ablation of the top-32 circuit features (no training).

| type | SHIFT hacking rate | best training method |
|---|---|---|
| syco | 0.000 | (no hacking to remove) |
| length | 0.400 | KL training: 0.017 |

For length, **inference-time SHIFT ablation did not reduce hacking (0.40)** — consistent
with §4: ablating the top-32 features doesn't stop verbosity because the behaviour has
redundant routes. Training-time KL beat both SHIFT and circuit-guided training on
surface behaviour. So on this behaviour, *neither* feature-targeted intervention
(inference-time SHIFT nor training-time circuit penalty) beat a generic training-time
baseline for the surface metric — while the circuit penalty was uniquely effective on
the mechanism.

---

## Section 6 — Cross-type generalisation

Train circuit-guided on **C_syco** (best λ), then evaluate **length** hacking rate:

| | length hacking rate |
|---|---|
| length baseline | 0.293 |
| after syco-circuit training | **0.017** |

Superficially, training on the *sycophancy* circuit strongly reduced *length* hacking —
which would be surprising given Phase 2 found the circuits are largely distinct.
**We do not interpret this as genuine mechanistic transfer.** The syco-circuit run earns
zero syco reward (nothing is learned for syco), so the update is dominated by the
circuit penalty + policy-gradient noise, which **degrades generation generally**
(shorter, more broken outputs). Shorter outputs mechanically lower the *length* reward
(reward ∝ token count). The drop is most plausibly **collateral capability damage, not
transfer** — matching Phase 2's prediction that the distinct circuits should *not*
transfer. This is exactly the kind of "surprising cross-type effect" the task flagged;
on inspection it is an artefact of the reward's length-sensitivity, not shared mechanism.

---

## Section 7 — Honest assessment

**Does circuit-guided regularization outperform baselines? On surface behaviour, no; on
the mechanism, yes.**

- On the one behaviour with a learnable reward (length), **the generic KL baseline beat
  circuit-guided training** on hacking rate (0.017 vs 0.400) with less capability loss.
  The circuit penalty failed to suppress the behaviour because it targets 10–32 features
  of a ~200-node distributed circuit; the model routed around them.
- **But circuit-guided training was uniquely effective at altering the targeted
  mechanism** — it drove the top hacking features to near-zero (0.022 for syco) far
  beyond any generic method, and prevented the circuit-amplification that the
  unregularized baseline caused. If the goal is *removing a specific mechanism* rather
  than *suppressing a surface metric*, it is the strongest tool tested.
- **What went wrong for behaviour:** distributed circuits + redundant routes. The
  method's premise (penalise the causal features) is undermined when the behaviour has
  many causal features and you penalise only the top handful. This argues for penalising
  a *larger fraction* of the circuit, or the whole discovered node set, not just top-32.
- **Syco was behaviorally untestable** at this scale (reward too sparse for a 124M base
  model in 120 steps).

Every result here is reported as measured. The honest summary: **on GPT-2 Small at
reduced scale, circuit-guided RLHF changes internal mechanisms more than it changes
behaviour, and does not beat a KL baseline on the surface metric** — an informative
negative-plus-nuance result that motivates penalising broader circuit coverage in
future work.

---

## Section 8 — Limitations

- Reduced scale (120 steps vs 2000; 30 prompts; λ∈{0.1,1.0}; single seed per cell)
  — dynamics and the λ band would sharpen with the full budget.
- REINFORCE, not PPO clipping (TRL 1.8 removed classic PPO) — higher-variance updates.
- Capability proxy is 8 short factual continuations, not full TruthfulQA.
- Circuit recovery uses squared feature activation as an IE proxy (re-running full
  attribution per condition was out of budget); the direction (which method suppresses
  the features) is reliable, the absolute IE is not re-estimated.
- Syco's exact-phrase reward is too hard for GPT-2 Small; a softer syco reward
  (e.g. embedding similarity to agreement) would make syco behaviorally testable.
