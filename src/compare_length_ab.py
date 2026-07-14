"""Compare the length-HACKING circuit (Type A, circuit_length) with the
legitimate-VERBOSITY circuit (Type B, circuit_length_ctrl).

The safety question: are the length features hacking-SPECIFIC or length-GENERIC?
  - If A and B are disjoint  -> filler features are hacking-specific; penalising the
    Type A circuit will NOT harm legitimately-verbose answers. SAFE.
  - If A and B overlap       -> the Type A circuit contains features also active during
    justified verbosity; penalising it would suppress good long answers too. CONFOUNDED.

Key numbers:
  * node/edge Jaccard (A vs B), and vs a random null
  * of the TOP-32 Type A features, how many also appear in the Type B circuit
  * layer overlap and shared-feature token semantics
Writes results/phase2/length_ab_comparison.json.
"""
from __future__ import annotations
import json, os, random
import config as C

P1 = os.path.join(C.RESULTS_DIR, "phase1")
OUT = os.path.join(C.RESULTS_DIR, "phase2")
random.seed(C.SEED)


def load(name):
    return json.load(open(os.path.join(P1, f"circuit_{name}.json")))


def nodeset(c):
    return {(n["layer"], n["feature"]) for n in c["nodes"]}


def edgeset(c):
    return {(e["lu"], e["iu"], e["ld"], e["id"]) for e in c["edges"]}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


def null_jaccard(sa, sb, universe=24576 * 12, trials=1000):
    vals = []
    for _ in range(trials):
        a = set(random.sample(range(universe), sa))
        b = set(random.sample(range(universe), sb))
        vals.append(len(a & b) / len(a | b) if (a or b) else 0.0)
    return sum(vals) / len(vals)


def main():
    A = load("length")          # Type A: hacking verbosity
    B = load("length_ctrl")     # Type B: legitimate verbosity
    sa, sb = nodeset(A), nodeset(B)
    ea, eb = edgeset(A), edgeset(B)

    # top-32 Type A features: how many recur in the Type B circuit?
    topA = sorted(A["nodes"], key=lambda n: -abs(n["ie"]))[:32]
    topA_keys = {(n["layer"], n["feature"]) for n in topA}
    recur = topA_keys & sb
    interpB = {(d["layer"], d["feature"]): [t for t, _ in d["top_tokens"][:4]]
               for d in B["top_nodes_interpreted"]}
    interpA = {(d["layer"], d["feature"]): [t for t, _ in d["top_tokens"][:4]]
               for d in A["top_nodes_interpreted"]}

    res = {
        "typeA": {"name": "length hacking (longer, NOT better)",
                  "n_nodes": A["n_nodes"], "n_edges": A["n_edges"]},
        "typeB": {"name": "legitimate verbosity (longer AND better)",
                  "n_nodes": B["n_nodes"], "n_edges": B["n_edges"]},
        "node_jaccard": jaccard(sa, sb),
        "edge_jaccard": jaccard(ea, eb),
        "n_shared_nodes": len(sa & sb),
        "null_node_jaccard": null_jaccard(len(sa), len(sb)),
        "top32_A_features_also_in_B": len(recur),
        "top32_A_recurring": [
            {"layer": l, "feature": f,
             "tokens_in_A": interpA.get((l, f), []),
             "tokens_in_B": interpB.get((l, f), [])}
            for (l, f) in sorted(recur)],
    }
    # verdict
    frac_recur = res["top32_A_features_also_in_B"] / 32
    if res["node_jaccard"] <= 3 * res["null_node_jaccard"] and frac_recur < 0.15:
        res["verdict"] = ("DISJOINT — length-hacking features are largely hacking-"
                          "specific; penalising the Type A circuit should not harm "
                          "legitimately-verbose answers.")
    elif frac_recur >= 0.30:
        res["verdict"] = ("CONFOUNDED — a substantial share of the top length-hacking "
                          "features are ALSO active in the legitimate-verbosity circuit; "
                          "penalising them would suppress justified long answers too.")
    else:
        res["verdict"] = ("PARTIAL OVERLAP — some top length-hacking features recur in "
                          "the legitimate-verbosity circuit; penalising the full Type A "
                          "circuit carries moderate risk to good long answers.")

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "length_ab_comparison.json"), "w"), indent=2)

    print(f"Type A (hack):  {res['typeA']['n_nodes']} nodes")
    print(f"Type B (legit): {res['typeB']['n_nodes']} nodes")
    print(f"node Jaccard A∩B = {res['node_jaccard']:.4f}  (null {res['null_node_jaccard']:.5f})")
    print(f"edge Jaccard A∩B = {res['edge_jaccard']:.4f}")
    print(f"shared nodes     = {res['n_shared_nodes']}")
    print(f"top-32 Type A features also in Type B circuit = "
          f"{res['top32_A_features_also_in_B']}/32")
    if res["top32_A_recurring"]:
        print("  recurring features (active in BOTH hacking and legit verbosity):")
        for r in res["top32_A_recurring"]:
            print(f"    L{r['layer']} F{r['feature']}: A={r['tokens_in_A']}  B={r['tokens_in_B']}")
    print("\nVERDICT:", res["verdict"])


if __name__ == "__main__":
    main()
