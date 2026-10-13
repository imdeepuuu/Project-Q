"""PS-O3: 0/1 Knapsack -> QUBO -> Ising cost Hamiltonian -> QAOA.

Bit-ordering convention (identical to Qiskit): variable i <-> qubit i <-> bit i of the
integer basis index.  Bitstrings printed by Qiskit are therefore little-endian
(``"b_{n-1} ... b_1 b_0"``).
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from math import ceil, log2

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit

# --------------------------------------------------------------------------- #
# Official PS-O3 inline data
# --------------------------------------------------------------------------- #
VALUES = [10, 4, 7, 5, 8]
WEIGHTS = [5, 3, 4, 2, 6]
CAPACITY = 15


# --------------------------------------------------------------------------- #
# Classical brute-force reference
# --------------------------------------------------------------------------- #
def brute_force(values=VALUES, weights=WEIGHTS, capacity=CAPACITY):
    """Pure-python exhaustive search over all 2^n item subsets (the classical baseline)."""
    n = len(values)
    t0 = time.perf_counter()
    best_val, best_sets = -1, []
    n_feasible = 0
    for x in itertools.product((0, 1), repeat=n):
        w = sum(wi * xi for wi, xi in zip(weights, x))
        if w <= capacity:
            n_feasible += 1
            v = sum(vi * xi for vi, xi in zip(values, x))
            if v > best_val:
                best_val, best_sets = v, [x]
            elif v == best_val:
                best_sets.append(x)
    dt = time.perf_counter() - t0
    return {
        "best_value": best_val,
        "best_sets": best_sets,
        "best_weight": sum(w * xi for w, xi in zip(weights, best_sets[0])),
        "n_feasible": n_feasible,
        "n_total": 2 ** n,
        "time_s": dt,
    }


def all_bitstrings(n):
    """(2^n, n) array; row = basis index, column i = bit i (little-endian)."""
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


# --------------------------------------------------------------------------- #
# QUBO container
# --------------------------------------------------------------------------- #
@dataclass
class QUBO:
    """C(z) = offset + sum_i Q[i,i] z_i + sum_{i<j} Q[i,j] z_i z_j  (Q upper-triangular)."""

    Q: np.ndarray
    offset: float
    n_items: int
    name: str
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return self.Q.shape[0]

    def cost_vector(self):
        """Cost of every one of the 2^n basis states (index = integer value of bits)."""
        Z = all_bitstrings(self.n).astype(float)
        lin = Z @ np.diag(self.Q)
        quad = np.einsum("ki,ij,kj->k", Z, np.triu(self.Q, 1), Z)
        return self.offset + lin + quad

    def cost(self, z):
        z = np.asarray(z, dtype=float)
        return float(self.offset + np.diag(self.Q) @ z + z @ np.triu(self.Q, 1) @ z)


def knapsack_value_weight(x, values=VALUES, weights=WEIGHTS):
    x = np.asarray(x)[: len(values)]
    return int(np.dot(values, x)), int(np.dot(weights, x))


# --------------------------------------------------------------------------- #
# Formulation S: standard QUBO with binary slack variables (n + ceil(log2(W+1)) qubits)
# --------------------------------------------------------------------------- #
def qubo_slack(values=VALUES, weights=WEIGHTS, capacity=CAPACITY, A=None):
    """min  -sum v_i x_i + A (sum w_i x_i + sum_k 2^k s_k - W)^2.

    Slack bits turn  w.x <= W  into an equality; any weight overflow gives a strictly positive
    quadratic penalty A*(w.x - W)^2.  Needs n + 4 = 9 qubits for the PS-O3 data.
    """
    n = len(values)
    m = ceil(log2(capacity + 1))
    coef = np.array(list(weights) + [2 ** k for k in range(m)], dtype=float)
    N = n + m
    if A is None:
        A = float(max(values))
    Q = np.zeros((N, N))
    for i in range(N):
        Q[i, i] = A * (coef[i] ** 2 - 2 * capacity * coef[i]) - (values[i] if i < n else 0)
        for j in range(i + 1, N):
            Q[i, j] = 2 * A * coef[i] * coef[j]
    return QUBO(Q, A * capacity ** 2, n, "S (slack, %d qubits)" % N, {"A": A, "m": m})


# --------------------------------------------------------------------------- #
# Formulation U: slack-free "unbalanced penalisation" (n qubits)
# --------------------------------------------------------------------------- #
def qubo_unbalanced(values=VALUES, weights=WEIGHTS, capacity=CAPACITY, lam1=1.0, lam2=0.1):
    """min  -sum v_i x_i  - lam1 * h + lam2 * h^2 ,   h = W - w.x .

    h^2 is a quadratic penalty that grows for weight OVERFLOW (h<0); the linear term -lam1*h
    makes the penalty asymmetric (2nd-order expansion of exp(-k h)), so under-filled bags are
    not punished as in an equality constraint.  Only n = 5 qubits are required, so this
    formulation fits on the 5-qubit Processor A.
    """
    n = len(values)
    w = np.asarray(weights, float)
    W = capacity
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -values[i] + lam1 * w[i] + lam2 * (w[i] ** 2 - 2 * W * w[i])
        for j in range(i + 1, n):
            Q[i, j] = 2 * lam2 * w[i] * w[j]
    offset = -lam1 * W + lam2 * W ** 2
    return QUBO(Q, offset, n, "U (unbalanced, %d qubits)" % n, {"lam1": lam1, "lam2": lam2})


# --------------------------------------------------------------------------- #
# QUBO -> Ising   (z = (1 - Z)/2)
# --------------------------------------------------------------------------- #
def qubo_to_ising(qubo: QUBO):
    """Return (h, J, const): C = const + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j."""
    Q, n = qubo.Q, qubo.n
    h = np.zeros(n)
    J = {}
    const = qubo.offset + np.trace(Q) / 2
    for i in range(n):
        h[i] -= Q[i, i] / 2
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                J[(i, j)] = Q[i, j] / 4
                h[i] -= Q[i, j] / 4
                h[j] -= Q[i, j] / 4
                const += Q[i, j] / 4
    return h, J, const


def ising_diagonal(h, J, const, n):
    Zs = 1 - 2 * all_bitstrings(n).astype(float)  # eigenvalue of Z_i
    d = np.full(2 ** n, const) + Zs @ h
    for (i, j), v in J.items():
        d += v * Zs[:, i] * Zs[:, j]
    return d


def qubo_to_sparse_pauli_op(qubo: QUBO):
    from qiskit.quantum_info import SparsePauliOp

    h, J, const = qubo_to_ising(qubo)
    n = qubo.n
    terms = [("I" * n, const)]
    for i, v in enumerate(h):
        if abs(v) > 1e-12:
            s = ["I"] * n
            s[n - 1 - i] = "Z"
            terms.append(("".join(s), v))
    for (i, j), v in J.items():
        s = ["I"] * n
        s[n - 1 - i] = s[n - 1 - j] = "Z"
        terms.append(("".join(s), v))
    return SparsePauliOp.from_list(terms).simplify()


# --------------------------------------------------------------------------- #
# QAOA circuit
# --------------------------------------------------------------------------- #
def ising_scale(h, J):
    """Normalisation so that gamma ~ O(1):  H_scaled = H / scale."""
    return float(max(np.max(np.abs(h)), max((abs(v) for v in J.values()), default=0.0), 1e-12))


def qaoa_circuit(h, J, gammas, betas, scale=1.0, measure=True):
    """|+>^n then p layers of  exp(-i gamma H_C/scale)  exp(-i beta sum X)."""
    n = len(h)
    qc = QuantumCircuit(n, name="QAOA_p%d" % len(gammas))
    qc.h(range(n))
    for g, b in zip(gammas, betas):
        for (i, j), v in sorted(J.items()):
            qc.rzz(2 * g * v / scale, i, j)
        for i, v in enumerate(h):
            if abs(v) > 1e-12:
                qc.rz(2 * g * v / scale, i)
        qc.rx(2 * b, range(n))
    if measure:
        qc.measure_all()
    return qc


# --------------------------------------------------------------------------- #
# Fast exact (noise-free) QAOA engine in numpy + classical outer loop
# --------------------------------------------------------------------------- #
def _apply_rx_all(psi, beta, n):
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape([2] * n)
    for ax in range(n):
        a0 = np.take(psi, 0, axis=ax)
        a1 = np.take(psi, 1, axis=ax)
        psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
    return psi.reshape(-1)


def qaoa_state(diag_scaled, gammas, betas, n):
    # NOTE: reshape([2]*n) is C-ordered, so axis k <-> bit (n-1-k).  The mixer is identical on
    # every axis, so the ordering is irrelevant for the state; the flat index stays little-endian.
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * diag_scaled)
        psi = _apply_rx_all(psi, b, n)
    return psi


def qaoa_expectation(params, diag_scaled, cost_vec, p, n):
    psi = qaoa_state(diag_scaled, params[:p], params[p:], n)
    return float(np.dot(np.abs(psi) ** 2, cost_vec))


def optimize_qaoa(qubo: QUBO, p=1, n_starts=24, maxiter=400, seed=42):
    """Multi-start COBYLA on the noise-free <C>.  Returns best params + convergence trace."""
    h, J, const = qubo_to_ising(qubo)
    n = qubo.n
    scale = ising_scale(h, J)
    cost_vec = qubo.cost_vector()
    diag_s = (ising_diagonal(h, J, const, n) - const) / scale
    rng = np.random.default_rng(seed)
    best = None
    for s in range(n_starts):
        x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
        trace = []

        def fun(x):
            v = qaoa_expectation(x, diag_s, cost_vec, p, n)
            trace.append(v)
            return v

        r = minimize(fun, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
        if best is None or r.fun < best["energy"]:
            best = {"energy": float(r.fun), "params": r.x, "trace": trace, "start": s,
                    "nfev": int(r.nfev)}
    best.update({"p": p, "scale": scale, "h": h, "J": J, "const": const,
                 "gammas": best["params"][:p], "betas": best["params"][p:]})
    return best


def ideal_distribution(qubo, opt):
    n = qubo.n
    diag_s = (ising_diagonal(opt["h"], opt["J"], opt["const"], n) - opt["const"]) / opt["scale"]
    psi = qaoa_state(diag_s, opt["gammas"], opt["betas"], n)
    return np.abs(psi) ** 2
if __name__ == "__main__":
    print("Knapsack problem module loaded successfully.")
    print("Values:", VALUES)
    print("Weights:", WEIGHTS)
    print("Capacity:", CAPACITY)