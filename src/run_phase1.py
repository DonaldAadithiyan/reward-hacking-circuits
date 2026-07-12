"""Phase 1 driver - discover circuits C_syco, C_length (and attempt C_code).

Runs node IE -> auto-threshold -> edge IE -> faithfulness -> interpretation,
saves each circuit to results/phase1/circuit_<type>.json and a summary.

Code is included but expected to be a null result (see verify_m: GPT-2 Small
does not prefer code exploits). We still run it and report faithfulness ~ n/a.
"""
from __future__ import annotations
import json, os, time, sys, functools
print = functools.partial(print, flush=True)  # noqa: A001 - flush progress
import torch
from model_utils import load_pairs
from attribution import (node_indirect_effects, select_nodes, auto_threshold,
                         edge_indirect_effects, select_edges)
from faithfulness import evaluate_circuit
from interpret import describe_top_nodes
import config as C

OUT = os.path.join(C.RESULTS_DIR, "phase1")
os.makedirs(OUT, exist_ok=True)


def discover(hack_type):
    print(f"\n=== Discovering circuit for {hack_type} ===")
    t0 = time.time()
    pairs = load_pairs(hack_type)
    device = C.DEVICE

    # Node IE
    ie, clean_means = node_indirect_effects(hack_type, pairs, device=device)
    t_n, n_nodes = auto_threshold(ie)
    nodes = select_nodes(ie, t_n)
    print(f"  T_N={t_n:.4f} -> {len(nodes)} nodes")

    # Edge IE
    edges = edge_indirect_effects(hack_type, pairs, nodes, device=device)
    t_e = C.T_E
    sel_edges = select_edges(edges, t_e)
    # tune t_e down if too few edges
    tries = 0
    while len(sel_edges) < 10 and tries < 8:
        t_e *= 0.6
        sel_edges = select_edges(edges, t_e)
        tries += 1
    print(f"  T_E={t_e:.5f} -> {len(sel_edges)} edges")

    # Faithfulness / completeness
    fc = evaluate_circuit(hack_type, pairs, nodes, clean_means, device=device)

    # Interpretation of top nodes
    top_desc = describe_top_nodes(nodes, top_n=10, device=device)

    circuit = {
        "hack_type": hack_type,
        "n_pairs": len(pairs),
        "T_N": t_n, "T_E": t_e,
        "n_nodes": len(nodes), "n_edges": len(sel_edges),
        "nodes": [{"layer": l, "submodule": s, "feature": i, "ie": v}
                  for (l, s, i, v) in nodes],
        "edges": [{"lu": lu, "iu": iu, "ld": ld, "id": idd, "ie": v}
                  for (lu, iu, ld, idd, v) in sel_edges],
        "top_nodes_interpreted": top_desc,
        "faithfulness": fc,
        "layer_ie_weight": _layer_weights(nodes),
        "runtime_s": round(time.time() - t0, 1),
    }
    path = os.path.join(OUT, f"circuit_{hack_type}.json")
    with open(path, "w") as f:
        json.dump(circuit, f, indent=2)
    # save full node IE vectors for Phase 2/3 (compressed)
    torch.save({k: v for k, v in ie.items()},
               os.path.join(OUT, f"node_ie_{hack_type}.pt"))
    torch.save(clean_means, os.path.join(OUT, f"clean_means_{hack_type}.pt"))
    print(f"  saved {path}  ({circuit['runtime_s']}s)")
    return circuit


def _layer_weights(nodes):
    w = {}
    for l, s, i, v in nodes:
        w[l] = w.get(l, 0.0) + abs(v)
    return {str(k): round(v, 4) for k, v in sorted(w.items())}


def main():
    types = os.environ.get("TYPES", "syco,length,code").split(",")
    summary = {}
    for ht in types:
        c = discover(ht)
        summary[ht] = {
            "n_nodes": c["n_nodes"], "n_edges": c["n_edges"],
            "T_N": c["T_N"], "T_E": c["T_E"],
            "faithfulness": c["faithfulness"]["faithfulness"],
            "completeness": c["faithfulness"]["completeness"],
            "m_full": c["faithfulness"]["m_full"],
        }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
