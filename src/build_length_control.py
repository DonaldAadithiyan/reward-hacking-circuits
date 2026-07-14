"""Type B control: LEGITIMATE verbosity pairs (longer AND higher quality).

Length-hacking (Type A, our existing circuit_length) used pairs where the longer
response was NOT better (verbose padding). This builds the CONTROL: same prompt,
longer response that IS clearly better (>= +1.0 quality score) -> a case where
length was genuinely warranted.

Both types put the LONGER response in the 'hacking' slot, so the metric direction
(m rewards length) is identical; the ONLY difference is whether that length was
justified. Discovering a circuit on Type B and comparing it to the Type A length
circuit tests whether the length features are hacking-SPECIFIC or length-GENERIC.

Writes data/pairs/length_ctrl.jsonl (fields prompt, clean, hacking).
"""
from __future__ import annotations
import json, os
import config as C

def build(n):
    from datasets import load_dataset
    ds = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)
    triples = []
    for row in ds:
        comps = [c for c in row["completions"]
                 if c.get("response") and c.get("overall_score") is not None]
        if len(comps) < 2:
            continue
        cs = sorted(comps, key=lambda c: len(c["response"].split()))
        short, long = cs[0], cs[-1]
        ls, ll = len(short["response"].split()), len(long["response"].split())
        if not (ll >= 2 * ls and ls >= 10 and ll <= 220):
            continue
        # Type B: the longer response is CLEARLY better -> length was justified
        if long["overall_score"] >= short["overall_score"] + 1.0:
            prompt = row["instruction"].strip()
            if len(prompt) > 300:
                continue
            triples.append({
                "prompt": prompt,
                "clean": short["response"].strip(),      # shorter, worse
                "hacking": long["response"].strip(),     # longer, BETTER (justified)
                "meta": {"short_words": ls, "long_words": ll,
                         "short_score": short["overall_score"],
                         "long_score": long["overall_score"]},
            })
            if len(triples) >= n:
                break
    return triples

if __name__ == "__main__":
    tr = build(C.N_PAIRS)
    path = os.path.join(C.PAIRS_DIR, "length_ctrl.jsonl")
    with open(path, "w") as f:
        for t in tr:
            f.write(json.dumps(t) + "\n")
    print(f"[length_ctrl] wrote {len(tr)} Type-B (justified verbosity) pairs -> {path}")
