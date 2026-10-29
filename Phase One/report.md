# PS-O3 experiment report

## Objective

Solve the 0/1 knapsack instance with QAOA, validate the optimum using exhaustive search, and
compare routed circuit cost on Processor A and Processor B.

## Method

The input is `values = [10, 4, 7, 5, 8]`, `weights = [5, 3, 4, 2, 6]`, and capacity `W = 15`.
The reference implementation evaluates all `2^5 = 32` item subsets. The five-qubit QUBO minimizes

`C(x) = -sum_i value_i*x_i - 3h + h^2`, where `h = W - sum_i weight_i*x_i`.

The residual penalty is calibrated for this instance and its ground state is checked against the
classical result before the QAOA run. `src/problem.py` also provides a standard binary-slack QUBO
for an exact generic inequality encoding. Since that formulation needs nine logical qubits here,
the A/B comparison uses the calibrated five-item circuit on both processors.

QAOA uses one layer (`p = 1`) and COBYLA multi-start optimization against an ideal statevector
expectation. Each optimized circuit is transpiled with the same basis gates, optimization level,
and transpiler seed. Noisy Aer sampling uses the same synthetic gate and readout error rates for
both topologies. Routing-seed results and shot-level solution metrics are exported as CSV.

## Results

For this run, the classical search examined all 32 subsets in `0.000086` seconds and found item
selection `(1, 1, 1, 1, 0)`: value `26`, weight `14`. The QUBO ground state matches that result.
The p=1 COBYLA multi-start optimization (12 starts, up to 250 iterations each) took `2.089`
seconds; this is a classical statevector/optimizer runtime and is not a quantum speedup result.

Both processors sampled 4096 shots using the same noise rates:

| Metric | Processor A | Processor B |
|---|---:|---:|
| Physical qubits | 5 | 9 |
| Routed SWAP gates | 4 | 1 |
| Routed CX gates | 32 | 23 |
| Circuit depth | 60 | 38 |
| Estimated gate/readout success | 0.7705 | 0.8061 |
| Sampled optimal-solution probability | 0.0791 | 0.0837 |
| Sampled feasible-solution probability | 0.7832 | 0.7864 |
| Approximation ratio | 0.6091 | 0.6166 |

Across 20 transpiler seeds, mean (standard deviation) SWAP counts were `4.15 (0.37)` for A and
`1.00 (0.00)` for B. Mean CX counts were `32.35 (0.93)` for A and `23.00 (0.00)` for B; mean
depths were `59.60 (1.93)` and `40.75 (2.86)`, respectively. Thus, for this p=1 circuit, B uses
75% fewer SWAPs and about 28% fewer CX gates on the selected transpilation seed.

The experiment exports complete per-processor metrics to
`results/tables/processor_comparison.csv`, per-seed routing counts to
`results/tables/routing_seed_sweep.csv`, and seed aggregates to
`results/tables/routing_seed_summary.csv`. The plot is
`results/figures/routing_comparison.png`; machine-readable details are in
`results/benchmark_summary.json`. Runtime values depend on the host.

## Interpretation and limitations

Processor B's dense five-qubit core is intended to natively support most of the pairwise
interactions in the QUBO cost layer. Processor A is a sparse five-qubit tree and therefore
provides the connectivity-limited baseline. The comparison measures the *compiled circuit* and
uses an identical synthetic noise model, not device calibration data. The local QAOA optimizer
and Aer simulation are classical computations; they do not establish quantum advantage.
