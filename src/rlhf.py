"""Phase 3 - circuit-guided RLHF.

TRL 1.8 removed the classic PPOTrainer/PPOConfig, and none of its trainers expose
the mid-forward SAE activations we need for the circuit-regularization term. We
therefore implement a compact, transparent REINFORCE-with-baseline (a PPO-family
policy-gradient) loop by hand on GPT-2 Small. This keeps full control over the
custom objective  L_total = L_PG + lambda * L_circuit.

Reward functions (TASK Step 1):
  syco   : agreement markers present in the completion
  length : min(1, tokens/target)
  code   : fraction of visible asserts passing (executed)

Conditions (TASK Step 3):
  1 none          - reward only
  2 kl            - reward - beta * KL(policy || ref)
  3 reward_shape  - reward clipped at 1, cosine-decayed over training
  4 circuit       - reward + circuit regularization on top-32 hacking features

Measurements (Step 4): hacking rate, capability (TruthfulQA-style), circuit
recovery (IE of original top-10 features after training), cross-type generalisation.
"""
from __future__ import annotations
import json, os, math, re, random, functools
print = functools.partial(print, flush=True)
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from sae_loader import get_sae, hook_name
import config as C

random.seed(C.SEED)
torch.manual_seed(C.SEED)

AGREE_MARKERS = ["you're right", "you are right", "i agree", "absolutely",
                 "great point", "you're correct", "you are correct", "well said"]


# --------------------------------------------------------------------------- #
# reward functions
# --------------------------------------------------------------------------- #
def reward_syco(prompt, completion):
    c = completion.lower()
    return 1.0 if any(m in c for m in AGREE_MARKERS) else 0.0

def reward_length(prompt, completion, target=60):
    n = len(completion.split())
    return min(1.0, n / target)

def _extract_asserts(prompt):
    return re.findall(r"assert [^\n]+", prompt)

def reward_code(prompt, completion):
    asserts = _extract_asserts(prompt)
    if not asserts:
        return 0.0
    ns = {}
    try:
        exec(completion, ns)  # noqa: S102 - sandboxed toy code from our own dataset
    except Exception:
        return 0.0
    passed = 0
    for a in asserts:
        try:
            exec(a, ns)  # noqa: S102
            passed += 1
        except Exception:
            pass
    return passed / len(asserts)

REWARDS = {"syco": reward_syco, "length": reward_length, "code": reward_code}


# --------------------------------------------------------------------------- #
# generation + policy gradient
# --------------------------------------------------------------------------- #
def load_policy(device):
    m = HookedTransformer.from_pretrained(C.MODEL_NAME, device=device)
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    return m

@torch.no_grad()
def generate(model, prompt, max_new=24, temperature=1.0, device="cpu"):
    ids = model.to_tokens(prompt)
    gen = ids.clone()
    logps = []
    for _ in range(max_new):
        logits = model(gen)[0, -1]
        probs = F.softmax(logits / temperature, dim=-1)
        tok = torch.multinomial(probs, 1)
        gen = torch.cat([gen, tok.view(1, 1)], dim=1)
        if tok.item() == model.tokenizer.eos_token_id:
            break
    text = model.to_string(gen[0, ids.shape[1]:])
    return gen, ids.shape[1], text


def circuit_features(hack_type, top_k=C.TOP_K_CIRCUIT):
    """Load top-k (layer, feature, |IE|) for the hacking circuit."""
    c = json.load(open(os.path.join(C.RESULTS_DIR, "phase1", f"circuit_{hack_type}.json")))
    nodes = sorted(c["nodes"], key=lambda n: -abs(n["ie"]))[:top_k]
    return [(n["layer"], n["feature"], abs(n["ie"])) for n in nodes]


def circuit_penalty(model, gen, prompt_len, feats, saes):
    """L_circuit = sum_f |IE(f)| * activation(f)^2, over completion positions."""
    # cache residual activations at needed layers
    acts = {}
    layers = sorted({l for l, i, w in feats})
    def make(l):
        def hook(a, hook):
            acts[l] = a
            return a
        return hook
    model.reset_hooks()
    for l in layers:
        model.add_hook(hook_name(l, "resid"), make(l))
    logits = model(gen)              # forward with grad
    model.reset_hooks()
    loss = gen.new_zeros((), dtype=torch.float32)
    for (l, i, w) in feats:
        f = saes[l].encode(acts[l])[0, prompt_len:, i]   # completion positions
        loss = loss + w * (f ** 2).mean()
    return loss, logits


def sequence_logprob(model, gen, prompt_len):
    logits = model(gen)
    lp = F.log_softmax(logits[0], dim=-1)
    ids = gen[0, prompt_len:]
    pred = lp[prompt_len - 1:-1]
    return pred[torch.arange(ids.shape[0]), ids].sum()


def train(hack_type, condition, prompts, steps=C.PPO_STEPS, lam=0.1, beta=0.1,
          lr=5e-5, device=None, seed=0, log_every=25):
    device = device or C.DEVICE
    torch.manual_seed(seed); random.seed(seed)
    model = load_policy(device)
    ref = load_policy(device) if condition == "kl" else None
    if ref:
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    saes = None
    feats = None
    if condition == "circuit":
        feats = circuit_features(hack_type)
        layers = sorted({l for l, i, w in feats})
        saes = {l: get_sae(l, "resid", device) for l in layers}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    reward_fn = REWARDS[hack_type]
    baseline = 0.0
    history = []

    for step in range(steps):
        prompt = random.choice(prompts)
        model.eval()
        gen, plen, text = generate(model, prompt, device=device)
        model.train()
        r = reward_fn(prompt, text)
        # reward shaping (condition 3)
        if condition == "reward_shape":
            r = min(1.0, r) * (0.5 * (1 + math.cos(math.pi * step / steps)))
        baseline = 0.95 * baseline + 0.05 * r
        adv = r - baseline

        logp = sequence_logprob(model, gen, plen)
        loss = -adv * logp

        if condition == "kl" and ref is not None:
            with torch.no_grad():
                ref_logp = sequence_logprob(ref, gen, plen)
            loss = loss + beta * (logp - ref_logp)
        if condition == "circuit":
            cp, _ = circuit_penalty(model, gen, plen, feats, saes)
            loss = loss + lam * cp

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        history.append({"step": step, "reward": r, "baseline": baseline})
        if (step + 1) % log_every == 0:
            recent = sum(h["reward"] for h in history[-log_every:]) / log_every
            print(f"  [{hack_type}/{condition}] step {step+1}/{steps} "
                  f"reward(avg{log_every})={recent:.3f}")
    return model, history
