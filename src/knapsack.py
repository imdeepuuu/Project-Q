import itertools
import json

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

from qff_phase2 import counts_to_x


def load_instance(path="data/instance.json"):
    d = json.load(open(path))["PS-O3"]
    return d["values"], d["weights"], d["capacity"], d["reference_penalty_P"]


def build_qubo(v, w, W, P):
    a = np.array(list(w) + [1, 2, 4, 8])
    n = len(a)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = P * (a[i] ** 2 - 2 * W * a[i])
    for i in range(len(v)):
        Q[i, i] -= v[i]
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = 2 * P * a[i] * a[j]
    return Q, P * W * W


def qubo_energy(z, Q, c0):
    z = np.asarray(z)
    return float(z @ Q @ z + c0)


def qubo_to_hamiltonian(Q, c0):
    n = len(Q)

    def z(*idx):
        s = ["I"] * n
        for k in idx:
            s[n - 1 - k] = "Z"
        return "".join(s)

    terms, const = [], c0
    for i in range(n):
        const += Q[i, i] / 2
        terms.append((z(i), -Q[i, i] / 2))
    for i in range(n):
        for j in range(i + 1, n):
            q = Q[i, j] / 4
            const += q
            terms += [(z(i), -q), (z(j), -q), (z(i, j), q)]
    H = SparsePauliOp.from_list(terms + [("I" * n, const)]).simplify()
    return H


def split_terms(H):
    n = H.num_qubits
    lin, quad, const = {}, {}, 0.0
    for p, c in zip(H.paulis, H.coeffs):
        idx = [k for k in range(n) if p.z[k]]
        c = float(np.real(c))
        if not idx:
            const += c
        elif len(idx) == 1:
            lin[idx[0]] = c
        else:
            quad[tuple(idx)] = c
    return lin, quad, const


def pauli_scale(H):
    return max(abs(float(np.real(c))) for p, c in zip(H.paulis, H.coeffs) if p.z.any())


def qaoa_circuit(H_scaled, p=1, measure=False):
    n = H_scaled.num_qubits
    lin, quad, _ = split_terms(H_scaled)
    gs = [Parameter(f"g{k}") for k in range(p)]
    bs = [Parameter(f"b{k}") for k in range(p)]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for k in range(p):
        for (i, j), c in quad.items():
            qc.rzz(2 * c * gs[k], i, j)
        for i, c in lin.items():
            qc.rz(2 * c * gs[k], i)
        qc.rx(2 * bs[k], range(n))
    if measure:
        qc.measure_all()
    return qc, gs, bs


def bind(qc, gs, bs, theta):
    p = len(gs)
    return qc.assign_parameters(dict(zip(gs + bs, list(theta[:p]) + list(theta[p:]))))


def optimize(H_scaled, seeds=(0, 1, 2, 3, 4), p=1, maxiter=200):
    qc, gs, bs = qaoa_circuit(H_scaled, p)
    runs = []
    for s in seeds:
        rng = np.random.default_rng(s)
        x0 = rng.uniform(0, np.pi, 2 * p)
        f = lambda t: float(np.real(Statevector(bind(qc, gs, bs, t)).expectation_value(H_scaled)))
        r = minimize(f, x0, method="COBYLA", options={"maxiter": maxiter})
        runs.append({"seed": s, "x0": x0.tolist(), "theta": r.x.tolist(), "cost": float(r.fun), "nfev": int(r.nfev)})
    return runs


def classical(v, w, W):
    subsets = list(itertools.product([0, 1], repeat=len(v)))
    feas = [x for x in subsets if np.dot(w, x) <= W]
    vstar = max(np.dot(v, x) for x in feas)
    opt = [x for x in feas if np.dot(v, x) == vstar]
    return {"V_star": int(vstar), "opt_subsets": opt, "n_feasible": len(feas),
            "n_subsets": len(subsets), "P_feasible_random": len(feas) / len(subsets),
            "P_opt_random": len(opt) / len(subsets),
            "mean_value_random_feasible": float(np.mean([np.dot(v, x) for x in feas]))}


def evaluate_counts(counts, v, w, W, opt_subsets, Q, c0):
    n_items = len(v)
    shots = sum(counts.values())
    feas = opt = 0
    best_val, best_items = -1, None
    energy = 0.0
    for key, c in counts.items():
        x = counts_to_x(key)
        items = x[:n_items]
        energy += c * qubo_energy(x, Q, c0)
        if np.dot(w, items) <= W:
            feas += c
            val = int(np.dot(v, items))
            if val > best_val:
                best_val, best_items = val, items
        if tuple(items) in opt_subsets:
            opt += c
    return {"shots": shots, "P_feasible": feas / shots, "P_opt": opt / shots,
            "best_feasible_value": best_val, "best_items": best_items,
            "best_weight": int(np.dot(w, best_items)) if best_items else None,
            "mean_qubo_energy": energy / shots}
