"""Processor A / B topology definitions, candidate-topology library and noise-model loaders."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

BASIS_GATES = ["rz", "sx", "x", "cx"]
OFFICIAL_NOISE = {"p1q": 0.001, "p2q": 0.005, "p_readout": 0.01}
TRANSPILE_CFG = {"optimization_level": 1, "seed_transpiler": 42, "shots": 4096}

# ---- Processor A (5 qubits): official baseline = branching "T" tree (IBM Falcon-5q shape) ----
EDGES_A = [(0, 1), (1, 2), (1, 3), (3, 4)]

# ---- Processor B (9 qubits): OUR design = "dense core + sparse tail" (12 edges, same edge budget
#      as the heavy-hex reference).  Core = K4 on {0,1,2,3} plus qubit 4 coupled to 1 and 2
#      (a K5 minus two edges -> 8 of the 10 pairs a 5-item knapsack cost layer needs);
#      tail 4-5-6-7-8 supplies the 4 extra qubits (slack bits) at heavy-hex-like sparsity. ----
EDGES_B = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),   # K4 core
           (1, 4), (2, 4),                                    # 5th core qubit
           (4, 5), (5, 6), (6, 7), (7, 8)]                    # sparse tail
EDGES_B_GRID = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
                (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)]

# ---- Reference architecture from the Master Guide (for the topology sweep) ----
EDGES_B_REF_HEAVYHEX = [(0, 1), (1, 2), (1, 6), (2, 3), (2, 6), (2, 5),
                        (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (8, 0)]

CANDIDATES_5Q = {
    "A_linear":  [(0, 1), (1, 2), (2, 3), (3, 4)],
    "A_tree_T (A)": EDGES_A,
    "A_star":    [(0, 1), (0, 2), (0, 3), (0, 4)],
    "A_ring":    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
    "A_bowtie":  [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)],
    "A_house":   [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)],
    "A_K5 (ideal)": [(i, j) for i in range(5) for j in range(i + 1, 5)],
}
CANDIDATES_9Q = {
    "B_linear":  [(i, i + 1) for i in range(8)],
    "B_ring":    [(i, (i + 1) % 9) for i in range(9)],
    "B_heavyhex_ref": EDGES_B_REF_HEAVYHEX,
    "B_grid3x3": EDGES_B_GRID,
    "B_grid3x3+4diag": EDGES_B_GRID + [(0, 4), (4, 8), (2, 4), (4, 6)],
    "B_bowtie_ring": [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 0)],
    "B_house_ring": [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 5), (5, 6), (6, 7), (7, 8), (8, 3)],
    "B_K4_tail": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
    "B_core+tail (B)": EDGES_B,
}


def coupling_map(edges, n_qubits=None, bidirectional=True) -> CouplingMap:
    edges = [tuple(e) for e in edges]
    if bidirectional:
        edges = edges + [(b, a) for a, b in edges]
    cm = CouplingMap(edges)
    if n_qubits is not None:
        assert cm.size() == n_qubits, f"coupling map has {cm.size()} qubits, expected {n_qubits}"
    return cm


def graph_stats(edges, n):
    g = nx.Graph(edges)
    g.add_nodes_from(range(n))
    deg = [d for _, d in g.degree()]
    return {"edges": g.number_of_edges(), "max_degree": max(deg), "mean_degree": round(sum(deg) / n, 2),
            "diameter": nx.diameter(g) if nx.is_connected(g) else None,
            "avg_shortest_path": round(nx.average_shortest_path_length(g), 3) if nx.is_connected(g) else None}


def build_noise_model(p1q=OFFICIAL_NOISE["p1q"], p2q=OFFICIAL_NOISE["p2q"],
                      p_readout=OFFICIAL_NOISE["p_readout"]) -> NoiseModel:
    """Official synthetic noise (identical on A and B): 1q depolarising on rz/sx/x, 2q depolarising
    on cx, symmetric readout flip.  rz is noised too (conservative: real rz is virtual)."""
    nm = NoiseModel(basis_gates=BASIS_GATES)
    nm.add_all_qubit_quantum_error(depolarizing_error(p1q, 1), ["rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2q, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]]))
    return nm


# ---------------------------------------------------------------------------------------- #
PROCESSOR_META = {
    "A": {
        "name": "Processor A (5q baseline, branching T-tree)",
        "n_qubits": 5,
        "edges": EDGES_A,
        "rationale": (
            "Kept equal to the guide's branching-tree baseline: it is the sparsest connected 5-qubit "
            "graph that still has a degree-3 hub (qubit 1) and mirrors real 5-qubit IBM Falcon "
            "devices.  4 edges vs the 10 edges a dense K5 Ising cost layer needs => it acts as the "
            "'connectivity-starved' control of the co-design experiment."),
    },
    "B": {
        "name": "Processor B (9q, dense K5-minus-2 core + sparse tail)",
        "n_qubits": 9,
        "edges": EDGES_B,
        "rationale": (
            "Co-designed for this problem: the knapsack cost Hamiltonian couples every pair of items "
            "(w_i*w_j terms => K5).  Qubits 0-3 form a K4 and qubit 4 couples to 1 and 2, so 8 of the "
            "10 required ZZ pairs are native and only 2 need routing.  Qubits 5-8 form a sparse tail "
            "(slack bits of formulation S) so the whole chip keeps 12 edges = the same budget as the "
            "heavy-hex reference.  A 20-seed sweep showed it beats the 3x3 grid (bipartite, no "
            "triangles) and the heavy-hex reference on SWAPs/CX at p=1 and p=2."),
    },
}


def write_processor_jsons(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, m in PROCESSOR_META.items():
        doc = {
            "name": m["name"], "n_qubits": m["n_qubits"],
            "coupling_map": [list(e) for e in m["edges"]], "bidirectional": True,
            "basis_gates": BASIS_GATES,
            "noise_model": {"type": "depolarizing+readout", "p_1q_depolarizing": OFFICIAL_NOISE["p1q"],
                            "p_cx_depolarizing": OFFICIAL_NOISE["p2q"],
                            "p_readout": OFFICIAL_NOISE["p_readout"], "noised_1q_gates": ["rz", "sx", "x"]},
            "transpile": {"optimization_level": 1, "seed_transpiler": 42, "shots": 4096},
            "graph_stats": graph_stats(m["edges"], m["n_qubits"]),
            "design_rationale": m["rationale"],
        }
        (out_dir / f"processor_{key}.json").write_text(json.dumps(doc, indent=2))


def load_processor(path) -> dict:
    d = json.loads(Path(path).read_text())
    d["edges"] = [tuple(e) for e in d["coupling_map"]]
    d["cmap"] = coupling_map(d["edges"], d["n_qubits"], d.get("bidirectional", True))
    nm = d["noise_model"]
    d["noise"] = build_noise_model(nm["p_1q_depolarizing"], nm["p_cx_depolarizing"], nm["p_readout"])
    return d
