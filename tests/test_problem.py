import numpy as np

from src.problem import (
    CAPACITY,
    VALUES,
    WEIGHTS,
    all_bitstrings,
    brute_force,
    qubo_slack,
    qubo_to_ising,
    qubo_unbalanced,
    ising_diagonal,
)


def test_classical_optimum_and_overflow_penalty_agree():
    reference = brute_force()
    qubo = qubo_unbalanced()
    costs = qubo.cost_vector()
    minimum = np.flatnonzero(np.isclose(costs, costs.min()))
    item_bits = all_bitstrings(qubo.n)[minimum, : qubo.n_items]
    selected = [
        tuple(bits.tolist())
        for bits in item_bits
    ]

    assert reference["best_value"] == 26
    assert reference["best_weight"] == 14
    assert (1, 1, 1, 1, 0) in reference["best_sets"]
    assert selected == [(1, 1, 1, 1, 0)]


def test_slack_qubo_has_only_feasible_optimal_item_assignments():
    reference = brute_force()
    qubo = qubo_slack()
    costs = qubo.cost_vector()
    min_cost = costs.min()
    opt_indices = np.flatnonzero(np.isclose(costs, min_cost))
    bitstrings = all_bitstrings(qubo.n)

    for index in opt_indices:
        item_bits = bitstrings[index, : qubo.n_items]
        value = int(np.dot(item_bits, VALUES))
        weight = int(np.dot(item_bits, WEIGHTS))
        assert value == reference["best_value"]
        assert weight <= CAPACITY


def test_qubo_and_ising_diagonals_match():
    qubo = qubo_unbalanced()
    h, couplings, constant = qubo_to_ising(qubo)
    np.testing.assert_allclose(
        qubo.cost_vector(),
        ising_diagonal(h, couplings, constant, qubo.n),
    )


def test_unbalanced_qubo_global_minimum():
    qubo = qubo_unbalanced()
    costs = qubo.cost_vector()
    bitstrings = all_bitstrings(qubo.n)

    best = brute_force()

    # The QUBO minimum must equal the classical optimum.
    min_cost = costs.min()
    min_states = bitstrings[
        abs(costs - min_cost) < 1e-9
    ]

    assert best["best_value"] == 26
    assert best["best_weight"] == 14

    
    for x in min_states:
        value = sum(v * bit for v, bit in zip(VALUES, x))
        weight = sum(w * bit for w, bit in zip(WEIGHTS, x))

        assert weight <= CAPACITY
        assert value == best["best_value"]
