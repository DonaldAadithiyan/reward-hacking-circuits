# LOG

Append newest entries at the top.

---

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
