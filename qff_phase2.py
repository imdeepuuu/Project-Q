"""
qff_phase2.py - Qiskit Fall Fest 2026, Phase 2 (Optimization Track) starter utilities.

Provided by the organizers so that every team measures things the SAME way.
You may read and copy this code freely. Do NOT change the official settings
(basis gates, noise rates, seed, shots, optimization level) for your required
benchmark runs - put any extra experiments in a clearly labelled separate section.

Tested with: Python 3.10-3.12, qiskit 1.4.x and 2.x, qiskit-aer 0.17+/0.18,
networkx 3.x, numpy, scipy, matplotlib.
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, StagedPassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

# --------------------------------------------------------------------------
# 1. OFFICIAL BENCHMARK SETTINGS (fixed for all teams)
# --------------------------------------------------------------------------
BASIS_GATES = ["rz", "sx", "x", "cx"]
OPT_LEVEL = 1
SEED = 42
SHOTS = 4096
P1_DEPOL = 0.001   # 1-qubit depolarizing per 1-qubit gate (rz, sx, x)
P2_DEPOL = 0.005   # 2-qubit depolarizing per CX
P_READOUT = 0.01   # symmetric readout bit-flip per measured qubit

# Official reference processors (undirected edges; used in both directions)
PROCESSOR_A_EDGES = [(0, 1), (1, 2), (1, 3), (3, 4)]                      # 5 qubits
PROCESSOR_B_EDGES = [(0, 1), (1, 2), (1, 6), (2, 3), (2, 6), (2, 5),
                     (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (8, 0)]      # 9 qubits

# Processor D design limits
D_MAX_QUBITS = 12
D_MAX_DEGREE = 4


def coupling_map_from_edges(edges, num_qubits=None) -> CouplingMap:
    """Bidirectional CouplingMap from a list of undirected edges."""
    directed = []
    for a, b in edges:
        directed += [[int(a), int(b)], [int(b), int(a)]]
    cm = CouplingMap(directed)
    if num_qubits is not None:
        for q in range(num_qubits):
            if q not in cm.physical_qubits:
                cm.add_physical_qubit(q)
    return cm


PROCESSOR_A = {"name": "Processor A", "num_qubits": 5, "edges": PROCESSOR_A_EDGES}
PROCESSOR_B = {"name": "Processor B", "num_qubits": 9, "edges": PROCESSOR_B_EDGES}


# --------------------------------------------------------------------------
# 2. NOISE MODEL
# --------------------------------------------------------------------------
def official_noise_model(p1=P1_DEPOL, p2=P2_DEPOL, p_ro=P_READOUT) -> NoiseModel:
    """Official synthetic noise model (identical on every processor)."""
    nm = NoiseModel(basis_gates=BASIS_GATES)
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm


def simulator(noisy: bool, processor: dict | None = None, noise_model=None) -> AerSimulator:
    """AerSimulator for ideal (noisy=False) or noisy runs. Pass circuits that are
    ALREADY transpiled for the processor (see transpile_for)."""
    if not noisy:
        return AerSimulator(seed_simulator=SEED)
    nm = noise_model if noise_model is not None else official_noise_model()
    kw = {}
    if processor is not None:
        kw["coupling_map"] = coupling_map_from_edges(processor["edges"], processor["num_qubits"])
    return AerSimulator(noise_model=nm, basis_gates=BASIS_GATES, seed_simulator=SEED, **kw)


# --------------------------------------------------------------------------
# 3. TRANSPILATION + OFFICIAL METRICS
# --------------------------------------------------------------------------
def _pass_manager(processor):
    cm = coupling_map_from_edges(processor["edges"], processor["num_qubits"])
    return generate_preset_pass_manager(optimization_level=OPT_LEVEL, basis_gates=BASIS_GATES,
                                        coupling_map=cm, seed_transpiler=SEED)


def transpile_for(qc: QuantumCircuit, processor: dict) -> QuantumCircuit:
    """Official transpilation: opt level 1, SABRE layout/routing, seed 42."""
    if qc.num_qubits > processor["num_qubits"]:
        raise ValueError(f"{processor['name']} has {processor['num_qubits']} qubits but the "
                         f"circuit needs {qc.num_qubits}: report 'not executable (capacity)'.")
    return _pass_manager(processor).run(qc)


def circuit_metrics(qc: QuantumCircuit, processor: dict) -> dict:
    """Official metric definitions (measurements/resets/barriers excluded from gate counts).

    logical_cx  : CX count after transpiling with NO connectivity limit (same settings)
    cx          : CX count after transpiling for the processor
    depth       : depth of the transpiled circuit, ignoring measure/barrier/reset
    two_q_depth : depth counting only CX gates
    swaps       : SWAP gates inserted by the routing stage (before basis translation)
    cx_overhead : cx - logical_cx  (extra CX caused by routing + layout)
    """
    if qc.num_qubits > processor["num_qubits"]:
        return {"processor": processor["name"], "executable": False,
                "reason": f"needs {qc.num_qubits} qubits > {processor['num_qubits']}"}
    pm = _pass_manager(processor)
    full = pm.run(qc)
    # Same init/layout/routing stages (same seed) -> count inserted SWAPs before decomposition
    routed = StagedPassManager(stages=["init", "layout", "routing"],
                               init=pm.init, layout=pm.layout, routing=pm.routing).run(qc)
    unconstrained = transpile(qc, basis_gates=BASIS_GATES, optimization_level=OPT_LEVEL,
                              seed_transpiler=SEED)
    skip = {"measure", "barrier", "reset"}
    ops = full.count_ops()
    return {
        "processor": processor["name"],
        "executable": True,
        "logical_qubits": qc.num_qubits,
        "physical_qubits": processor["num_qubits"],
        "logical_cx": unconstrained.count_ops().get("cx", 0),
        "cx": ops.get("cx", 0),
        "one_q_gates": sum(v for k, v in ops.items() if k in ("rz", "sx", "x")),
        "depth": full.depth(filter_function=lambda inst: inst.operation.name not in skip),
        "two_q_depth": full.depth(filter_function=lambda inst: inst.operation.num_qubits == 2
                                  and inst.operation.name not in skip),
        "swaps": routed.count_ops().get("swap", 0),
        "cx_overhead": ops.get("cx", 0) - unconstrained.count_ops().get("cx", 0),
    }


# --------------------------------------------------------------------------
# 4. BITSTRING CONVENTION
# --------------------------------------------------------------------------
def counts_to_x(bitstring: str) -> tuple:
    """Qiskit count keys are little-endian (qubit 0 is the RIGHT-most character).
    This returns (x0, x1, ..., x_{n-1}) with x_i = value of qubit i."""
    return tuple(int(b) for b in reversed(bitstring.replace(" ", "")))


# --------------------------------------------------------------------------
# 5. BELL-STATE VERIFICATION (Objective 1)
# --------------------------------------------------------------------------
def bell_circuit(basis: str) -> QuantumCircuit:
    """|Phi+> = (|00> + |11>)/sqrt(2) on logical qubits 0,1, measured in basis 'XX','YY' or 'ZZ'."""
    qc = QuantumCircuit(2, 2, name=f"bell_{basis}")
    qc.h(0)
    qc.cx(0, 1)
    if basis == "XX":
        qc.h([0, 1])
    elif basis == "YY":
        qc.sdg([0, 1])
        qc.h([0, 1])
    elif basis != "ZZ":
        raise ValueError("basis must be 'XX', 'YY' or 'ZZ'")
    qc.measure([0, 1], [0, 1])
    return qc


def parity_expectation(counts: dict) -> tuple:
    """<P> = (N_even - N_odd)/N for two-qubit parity, and its standard error."""
    n = sum(counts.values())
    even = sum(c for k, c in counts.items() if k.replace(" ", "").count("1") % 2 == 0)
    e = (2 * even - n) / n
    return e, float(np.sqrt(max(1 - e * e, 0.0) / n))


def bell_experiment(processor: dict, physical_pair: tuple, noisy: bool, shots: int = SHOTS,
                    noise_model=None) -> dict:
    """Run XX, YY, ZZ correlations for |Phi+> on a chosen PHYSICAL qubit pair.
    Fidelity F = (1 + <XX> - <YY> + <ZZ>)/4 (exact for |Phi+>). Ideal: XX=+1, YY=-1, ZZ=+1, F=1."""
    cm = coupling_map_from_edges(processor["edges"], processor["num_qubits"])
    sim = simulator(noisy, processor, noise_model)
    out = {"processor": processor["name"], "pair": tuple(physical_pair), "noisy": noisy,
           "shots": shots, "pair_distance": cm.distance(*physical_pair)}
    var = 0.0
    for b in ("XX", "YY", "ZZ"):
        tqc = transpile(bell_circuit(b), basis_gates=BASIS_GATES, coupling_map=cm,
                        initial_layout=list(physical_pair), optimization_level=OPT_LEVEL,
                        seed_transpiler=SEED)
        counts = sim.run(tqc, shots=shots).result().get_counts()
        e, se = parity_expectation(counts)
        out[b], out[b + "_se"] = e, se
        var += se ** 2
        if b == "ZZ":
            out["cx"] = tqc.count_ops().get("cx", 0)
    out["fidelity"] = (1 + out["XX"] - out["YY"] + out["ZZ"]) / 4
    out["fidelity_se"] = float(np.sqrt(var) / 4)
    return out


# --------------------------------------------------------------------------
# 6. PROCESSOR D: LOAD + VALIDATE processors/processor_D.json
# --------------------------------------------------------------------------
def load_processor(path: str) -> dict:
    with open(path) as f:
        p = json.load(f)
    p["edges"] = [tuple(e) for e in p["edges"]]
    return p


def validate_processor_d(p: dict, required_qubits: int) -> list:
    """Returns a list of rule violations (empty list = valid)."""
    errs = []
    n = p.get("num_qubits")
    if not isinstance(n, int) or n < required_qubits or n > D_MAX_QUBITS:
        errs.append(f"num_qubits must be an integer in [{required_qubits}, {D_MAX_QUBITS}]")
        return errs
    if set(p.get("basis_gates", [])) != set(BASIS_GATES):
        errs.append(f"basis_gates must be exactly {BASIS_GATES}")
    deg = {q: 0 for q in range(n)}
    seen = set()
    for a, b in p.get("edges", []):
        if a == b or not (0 <= a < n and 0 <= b < n):
            errs.append(f"invalid edge {(a, b)}")
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            errs.append(f"duplicate edge {key}")
        seen.add(key)
        deg[a] += 1
        deg[b] += 1
    hi = [q for q, d in deg.items() if d > D_MAX_DEGREE]
    if hi:
        errs.append(f"qubits {hi} exceed max degree {D_MAX_DEGREE}")
    adj = {q: set() for q in range(n)}
    for a, b in seen:
        adj[a].add(b)
        adj[b].add(a)
    reach, dq = {0}, deque([0])
    while dq:
        for nb in adj[dq.popleft()]:
            if nb not in reach:
                reach.add(nb)
                dq.append(nb)
    if len(reach) != n:
        errs.append("coupling graph must be connected")
    nz = p.get("noise", {})
    if nz.get("p1", P1_DEPOL) < P1_DEPOL or nz.get("p2", P2_DEPOL) < P2_DEPOL \
            or nz.get("p_readout", P_READOUT) < P_READOUT:
        errs.append("noise rates in the JSON may not be lower (better) than the official model")
    return errs
