# PS-O3: 0/1 Knapsack with QAOA

This project formulates the five-item knapsack instance as a QUBO, checks it against exhaustive
classical search, optimizes a depth-one QAOA circuit, and compares the same logical circuit routed
onto Processor A and Processor B.

## Problem and formulation

| Item | Value | Weight |
|---:|---:|---:|
| 1 | 10 | 5 |
| 2 | 4 | 3 |
| 3 | 7 | 4 |
| 4 | 5 | 2 |
| 5 | 8 | 6 |

Capacity: `15`. The classical optimum selects items 1-4 (value `26`, weight `14`).

For the five-qubit comparison, the QUBO uses `h = W - sum(weight_i * x_i)` and minimizes
`-sum(value_i * x_i) - 3h + h^2`. This quadratic residual penalty is calibrated for this fixed
instance; exhaustive enumeration verifies that its unique ground state is feasible and agrees
with the classical optimum. For general instances, `qubo_slack` in `src/problem.py` supplies the
standard exact binary-slack QUBO (nine variables for this instance), with a penalty coefficient
large enough to exclude infeasible ground states. That larger encoding cannot run on five-qubit
Processor A, so the A/B routing comparison uses the same calibrated five-qubit QUBO on both.

## Reproduce

Use Python 3.10 or later:

```powershell
python -m pip install -r requirements.txt
python -m pytest
python -m src.benchmark
```

`src.benchmark` runs exhaustive search, the noise-free statevector QAOA parameter optimizer,
noisy Aer simulations on both processors, and a 20-seed routing sweep. The identical synthetic
gate/readout noise model is used for A and B. Results are written to:

- `results/tables/processor_comparison.csv`
- `results/tables/routing_seed_sweep.csv`
- `results/tables/routing_seed_summary.csv`
- `results/benchmark_summary.json`
- `results/figures/routing_comparison.png`

Run `main.ipynb` for the same end-to-end experiment in notebook form. The circuit is simulated,
not executed on physical quantum hardware; reported runtimes are local software timings and will
vary by machine.

## Repository layout

```text
README.md
main.ipynb
requirements.txt
data/
  raw/
  processed/
processors/
  processor_A.json
  processor_B.json
src/
  problem.py
  processors.py
  routing.py
  metrics.py
  visualization.py
  benchmark.py
tests/
results/
  tables/
  figures/
report.md
assets/
  qff_phase1_latex_images/
```
