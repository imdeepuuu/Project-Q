"""Circuit metrics, noisy execution and solution-quality metrics."""
from __future__ import annotations

import time

import numpy as np
from qiskit_aer import AerSimulator
from qiskit.quantum_info import hellinger_fidelity

from processors import OFFICIAL_NOISE
from problem import CAPACITY, VALUES, WEIGHTS


def circuit_metrics(t, swaps, p=OFFICIAL_NOISE):
    ops = t.count_ops()
    n_cx = int(ops.get("cx", 0))
    n_1q = int(sum(v for k, v in ops.items() if k in ("rz", "sx", "x")))
    n_meas = int(ops.get("measure", 0))
    # Estimated success probability under the *official* synthetic noise (product of gate fidelities)
    esp = (1 - p["p1q"]) ** n_1q * (1 - p["p2q"]) ** n_cx * (1 - p["p_readout"]) ** n_meas
    return {"depth": t.depth(), "cx_count": n_cx, "swap_count": swaps, "1q_gates": n_1q,
            "cx_depth": t.depth(lambda inst: inst.operation.name == "cx"),
            "est_success_prob": round(esp, 4)}


def run_counts(circ, noise_model=None, shots=4096, seed=42):
    sim = AerSimulator(method="density_matrix" if noise_model is not None else "automatic",
                       noise_model=noise_model, seed_simulator=seed)
    t0 = time.perf_counter()
    res = sim.run(circ, shots=shots, seed_simulator=seed).result()
    return res.get_counts(), time.perf_counter() - t0


def counts_to_probs(counts, n):
    p = np.zeros(2 ** n)
    tot = sum(counts.values())
    for k, v in counts.items():
        p[int(k.replace(" ", ""), 2)] += v / tot   # little-endian bitstring -> basis index
    return p


def solution_metrics(probs, qubo, bf, ideal_probs=None):
    """Quality of a sampled distribution w.r.t. the classical optimum.  probs index = basis state."""
    n, ni = qubo.n, qubo.n_items
    idx = np.arange(2 ** n)
    items = idx & (2 ** ni - 1)
    Xb = ((items[:, None] >> np.arange(ni)[None, :]) & 1)
    vals = Xb @ np.array(VALUES)
    wts = Xb @ np.array(WEIGHTS)
    feas = wts <= CAPACITY
    opt_mask = (vals == bf["best_value"]) & feas
    cost = qubo.cost_vector()
    out = {
        "P_optimal": float(probs[opt_mask].sum()),
        "P_feasible": float(probs[feas].sum()),
        "expected_value_feasible_only": float((probs * vals * feas).sum()),
        "approx_ratio": float((probs * vals * feas).sum() / bf["best_value"]),
        "best_feasible_value_in_samples": int(vals[(probs > 0) & feas].max()) if np.any((probs > 0) & feas) else 0,
        "expected_qubo_cost": float(np.dot(probs, cost)),
        "energy_ratio": float((cost.max() - np.dot(probs, cost)) / (cost.max() - cost.min())),
    }
    if ideal_probs is not None:
        out["hellinger_fidelity_vs_ideal"] = float(hellinger_fidelity(
            {int(i): float(p) for i, p in enumerate(ideal_probs) if p > 0},
            {int(i): float(p) for i, p in enumerate(probs) if p > 0}))
    return out


def shots_to_confidence(p_opt, conf=0.99):
    """Number of shots needed to see the optimum at least once with probability `conf`."""
    if p_opt <= 0:
        return float("inf")
    if p_opt >= 1:
        return 1
    return float(np.ceil(np.log(1 - conf) / np.log(1 - p_opt)))
