# QFF 2026 Phase 2 — PS-O3: Resource-Constrained Item Selection (0/1 Knapsack) with QAOA

**Team:** KanyaRaasi 
**Problem statement:** PS-O3 (Optimization track), same as Phase 1

p = 1 QAOA on the locked 9-qubit knapsack QUBO (5 item bits + 4 binary slack bits, P = 5), benchmarked on Processors A, B and our Processor D (a 3×3 torus), ideal and with the official noise model.

## How to run

```
python -m venv opt-env
opt-env\Scripts\activate            # Windows
source opt-env/bin/activate         # macOS / Linux
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Or open `main.ipynb` in Jupyter and run all cells. The full notebook takes about 1 minute on a laptop CPU (the landscape, about 50 s, is the slowest step). Set `RECOMPUTE_LANDSCAPE = False` in the first code cell to reload `results/tables/landscape_D_noisy.npz` instead.

All required runs use the official settings in `qff_phase2.py` (unchanged): basis rz/sx/x/cx, optimization_level 1, SABRE, seed_transpiler 42, seed_simulator 42, 4096 shots, official noise. QAOA initializations use `numpy.random.default_rng(seed)` with seeds 0–4.

## Key results

V\* = 26 (items 1, 2, 3, 4; weight 14/15). Uniform random baseline: P_feasible = 28/32 = 0.875, P_opt = 1/32 = 0.031.

| Processor | Condition | CX | Depth | SWAPs | CX overhead | P_feasible | P_opt | Best feasible | % of V\* |
|---|---|---|---|---|---|---|---|---|---|
| A | ideal / noisy | not executable: needs 9 qubits, has 5 | | | | | | | |
| B | ideal | 121 | 117 | 17 | 49 | 0.959 | 0.021 | 26 | 100 |
| B | noisy | 121 | 117 | 17 | 49 | 0.943 | 0.025 | 26 | 100 |
| D | ideal | 89 | 76 | 9 | 17 | 0.956 | 0.023 | 26 | 100 |
| D | noisy | 89 | 76 | 9 | 17 | 0.943 | 0.023 | 26 | 100 |

Logical CX = 72 on every processor. Processor D cuts CX by 26%, SWAPs from 17 to 9 and depth from 117 to 76 compared with B.

Bell state on D: F = 0.9633 ± 0.0021 (adjacent pair 0–1, noisy) and 0.9548 ± 0.0023 (farthest pair 0–4, distance 2, noisy); both 1.0000 ideal.

Full discussion: `report.md`.

## Layout

```
README.md, main.ipynb, requirements.txt, qff_phase2.py (organizer module, unchanged), report.md
data/instance.json                 PS-O3 section of the locked instances.json
processors/processor_{A,B,D}.json  D is the frozen design
src/knapsack.py                    QUBO, Hamiltonian, QAOA circuit, optimizer, evaluation
results/tables/                    results.csv (kit format), bell.csv, quality.csv, qaoa_runs.json, qubo_9x9.json, landscape_D_noisy.npz
results/figures/                   PNG figures used in the report
```

## AI use

Claude (Anthropic) was used to help write the QUBO/QAOA code, search candidate Processor D topologies and draft the README and report. All numbers come from running `main.ipynb`; the team checked the formulation and results and is responsible for them.
