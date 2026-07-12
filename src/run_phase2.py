"""Phase 2 - comparative circuit analysis.

Loads the Phase 1 circuits and computes:
  Step 1  node overlap: standard / weighted / high-IE Jaccard + null distribution
  Step 2  edge Jaccard + routing divergence for shared nodes
  Step 3  layer distribution of |IE| weight
  Step 4  feature semantic comparison (uses interpret.py top tokens)

Writes results/phase2/comparison.json and layer-distribution data for plotting.
Only compares hacking types that produced a valid circuit with m_full>0
(code is expected to be excluded as a null result).
"""
from __future__ import annotations
import json, os, itertools, random
import torch
from interpret import top_tokens_for_feature
import config as C

P1 = os.path.join(C.RESULTS_DIR, "phase1")
OUT = os.path.join(C.RESULTS_DIR, "phase2")
os.makedirs(OUT, exist_ok=True)
random.seed(C.SEED)


def load_circuit(ht):
    with open(os.path.join(P1, f"circuit_{ht}.json")) as f:
        return json.load(f)


def node_set(c):
    return {(n["layer"], n["submodule"], n["feature"]) for n in c["nodes"]}


def node_ie_map(c):
    return {(n["layer"], n["submodule"], n["feature"]): n["ie"] for n in c["nodes"]}


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def weighted_jaccard(ma, mb):
    keys = set(ma) | set(mb)
    num = sum(min(abs(ma.get(k, 0)), abs(mb.get(k, 0))) for k in keys)
    den = sum(max(abs(ma.get(k, 0)), abs(mb.get(k, 0))) for k in keys)
    return num / den if den else 0.0


def high_ie_jaccard(c_a, c_b, top=20):
    a = {(n["layer"], n["submodule"], n["feature"])
         for n in sorted(c_a["nodes"], key=lambda x: -abs(x["ie"]))[:top]}
    b = {(n["layer"], n["submodule"], n["feature"])
         for n in sorted(c_b["nodes"], key=lambda x: -abs(x["ie"]))[:top]}
    return jaccard(a, b)


def edge_set(c):
    return {(e["lu"], e["iu"], e["ld"], e["id"]) for e in c["edges"]}


def null_jaccard(size_a, size_b, universe, trials=1000):
    """Random-set Jaccard null. universe = total feature slots (approx)."""
    vals = []
    for _ in range(trials):
        a = set(random.sample(range(universe), min(size_a, universe)))
        b = set(random.sample(range(universe), min(size_b, universe)))
        vals.append(jaccard(a, b))
    vals.sort()
    return vals


def percentile_of(value, sorted_null):
    import bisect
    return bisect.bisect_left(sorted_null, value) / len(sorted_null)


def layer_distribution(c):
    w = {str(l): 0.0 for l in range(C.N_LAYERS)}
    for n in c["nodes"]:
        w[str(n["layer"])] += abs(n["ie"])
    return w


def routing_divergence(c_a, c_b):
    """For nodes shared by both circuits, compare edge weights among them."""
    shared = node_set(c_a) & node_set(c_b)
    ea = {(e["lu"], e["iu"], e["ld"], e["id"]): e["ie"] for e in c_a["edges"]}
    eb = {(e["lu"], e["iu"], e["ld"], e["id"]): e["ie"] for e in c_b["edges"]}
    # edges where both endpoints are shared nodes
    def shared_edges(emap):
        out = {}
        for (lu, iu, ld, idd), v in emap.items():
            if (lu, "resid", iu) in shared and (ld, "resid", idd) in shared:
                out[(lu, iu, ld, idd)] = v
        return out
    sa, sb = shared_edges(ea), shared_edges(eb)
    keys = set(sa) | set(sb)
    if not keys:
        return {"n_shared_nodes": len(shared), "n_comparable_edges": 0,
                "mean_abs_weight_diff": None}
    diffs = [abs(sa.get(k, 0.0) - sb.get(k, 0.0)) for k in keys]
    return {"n_shared_nodes": len(shared),
            "n_comparable_edges": len(keys),
            "mean_abs_weight_diff": sum(diffs) / len(diffs)}


def main():
    # which types have valid circuits (m_full > 0)?
    summary = json.load(open(os.path.join(P1, "summary.json")))
    valid = [ht for ht, s in summary.items() if s["m_full"] > 0]
    print("valid circuits (m_full>0):", valid)
    circuits = {ht: load_circuit(ht) for ht in summary}

    D_SAE = 24576  # res-jb feature width
    universe = D_SAE * C.N_LAYERS

    results = {"valid_types": valid, "pairs": {}, "layer_distribution": {},
               "note": "code excluded from conclusions where m_full<=0 (null result)"}

    for ht in summary:
        results["layer_distribution"][ht] = layer_distribution(circuits[ht])

    for a, b in itertools.combinations(summary.keys(), 2):
        ca, cb = circuits[a], circuits[b]
        sa, sb = node_set(ca), node_set(cb)
        ma, mb = node_ie_map(ca), node_ie_map(cb)
        null = null_jaccard(len(sa), len(sb), universe)
        std_j = jaccard(sa, sb)
        rec = {
            "standard_jaccard": std_j,
            "weighted_jaccard": weighted_jaccard(ma, mb),
            "high_ie_jaccard": high_ie_jaccard(ca, cb),
            "edge_jaccard": jaccard(edge_set(ca), edge_set(cb)),
            "n_shared_nodes": len(sa & sb),
            "null_mean": sum(null) / len(null),
            "null_p95": null[int(0.95 * len(null))],
            "observed_percentile": percentile_of(std_j, null),
            "routing_divergence": routing_divergence(ca, cb),
            "both_valid": a in valid and b in valid,
        }
        results["pairs"][f"{a}-{b}"] = rec
        print(f"{a}-{b}: std_J={std_j:.4f} weighted={rec['weighted_jaccard']:.4f} "
              f"highIE={rec['high_ie_jaccard']:.4f} edge={rec['edge_jaccard']:.4f} "
              f"shared={rec['n_shared_nodes']} pct={rec['observed_percentile']:.3f}")

    with open(os.path.join(OUT, "comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results/phase2/comparison.json")
    return results


if __name__ == "__main__":
    main()
