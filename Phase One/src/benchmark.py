"""Run the PS-O3 exact baseline, QAOA simulation, and Processor A/B routing comparison."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import circuit_metrics, counts_to_probs, run_counts, solution_metrics
from .problem import (
    CAPACITY,
    VALUES,
    WEIGHTS,
    brute_force,
    ideal_distribution,
    optimize_qaoa,
    qaoa_circuit,
    qubo_unbalanced,
)
from .processors import load_processor
from .routing import routing_sweep_seeds, transpile_on


ROOT = Path(__file__).resolve().parents[1]
PROCESSORS = {
    "A": ROOT / "processors" / "processor_A.json",
    "B": ROOT / "processors" / "processor_B.json",
}


def run_benchmark(
    shots=4096,
    n_starts=12,
    maxiter=250,
    sweep_seeds=range(20),
    output_dir=ROOT / "results",
):
    """Run a reproducible p=1 QAOA experiment and export metrics, routing sweep, and figure."""
    output_dir = Path(output_dir)
    sweep_seeds = tuple(sweep_seeds)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    classical = brute_force()
    qubo = qubo_unbalanced()
    qubo_min = int(np.argmin(qubo.cost_vector()))
    qubo_bits = tuple((qubo_min >> i) & 1 for i in range(qubo.n_items))
    qubo_value = sum(v * bit for v, bit in zip(VALUES, qubo_bits))
    qubo_weight = sum(w * bit for w, bit in zip(WEIGHTS, qubo_bits))
    if qubo_weight > CAPACITY or qubo_value != classical["best_value"]:
        raise RuntimeError(
            "The configured overflow penalty does not encode the classical optimum: "
            f"QUBO selected value={qubo_value}, weight={qubo_weight}; "
            f"classical optimum={classical['best_value']}."
        )

    optimize_start = time.perf_counter()
    optimized = optimize_qaoa(qubo, p=1, n_starts=n_starts, maxiter=maxiter, seed=42)
    optimization_seconds = time.perf_counter() - optimize_start
    circuit = qaoa_circuit(
        optimized["h"],
        optimized["J"],
        optimized["gammas"],
        optimized["betas"],
        scale=optimized["scale"],
        measure=True,
    )
    ideal_probs = ideal_distribution(qubo, optimized)

    benchmark_rows = []
    sweep_rows = []
    for label, path in PROCESSORS.items():
        processor = load_processor(path)
        compile_start = time.perf_counter()
        routed, swaps, layout = transpile_on(circuit, processor["cmap"])
        compile_seconds = time.perf_counter() - compile_start
        counts, simulation_seconds = run_counts(
            routed, noise_model=processor["noise"], shots=shots, seed=42
        )
        sampled_probs = counts_to_probs(counts, qubo.n)
        row = {
            "processor": label,
            "processor_name": processor["name"],
            "physical_qubits": processor["n_qubits"],
            "shots": sum(counts.values()),
            "transpile_seconds": compile_seconds,
            "noisy_simulation_seconds": simulation_seconds,
            "classical_brute_force_seconds": classical["time_s"],
            "qaoa_optimization_seconds": optimization_seconds,
            **circuit_metrics(routed, swaps),
            **solution_metrics(sampled_probs, qubo, classical, ideal_probs=ideal_probs),
        }
        benchmark_rows.append(row)

        for sweep in routing_sweep_seeds(circuit, processor["cmap"], seeds=sweep_seeds):
            sweep_rows.append({"processor": label, **sweep})

    benchmarks = pd.DataFrame(benchmark_rows)
    sweeps = pd.DataFrame(sweep_rows)
    benchmarks.to_csv(table_dir / "processor_comparison.csv", index=False)
    sweeps.to_csv(table_dir / "routing_seed_sweep.csv", index=False)

    grouped = sweeps.groupby("processor").agg(
        swaps_mean=("swaps", "mean"),
        swaps_std=("swaps", "std"),
        cx_mean=("cx", "mean"),
        cx_std=("cx", "std"),
        depth_mean=("depth", "mean"),
        depth_std=("depth", "std"),
    ).reset_index()
    grouped.to_csv(table_dir / "routing_seed_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    colors = {"A": "#2f5fe0", "B": "#2e9e4f"}
    for ax, metric, title in zip(
        axes, ("swap_count", "cx_count"), ("Routed SWAP gates", "Routed CX gates")
    ):
        ax.bar(
            benchmarks["processor"],
            benchmarks[metric],
            color=[colors[label] for label in benchmarks["processor"]],
            edgecolor="black",
        )
        ax.set_title(title)
        ax.set_ylabel("gate count")
        ax.grid(axis="y", alpha=0.25)
    figure.suptitle("Same p=1 QAOA circuit, transpiled to Processor A and B")
    figure.tight_layout()
    figure.savefig(figure_dir / "routing_comparison.png", dpi=160)
    plt.close(figure)

    summary = {
        "problem": {
            "values": VALUES,
            "weights": WEIGHTS,
            "capacity": CAPACITY,
            "qubo": qubo.name,
            "qubo_penalty": qubo.meta,
        },
        "classical": {
            "best_value": classical["best_value"],
            "best_weight": classical["best_weight"],
            "best_sets": [list(bits) for bits in classical["best_sets"]],
            "feasible_subsets": classical["n_feasible"],
            "subsets_examined": classical["n_total"],
            "runtime_seconds": classical["time_s"],
        },
        "qaoa": {
            "p": 1,
            "best_expected_qubo_cost": optimized["energy"],
            "starts": n_starts,
            "maxiter_per_start": maxiter,
            "optimization_runtime_seconds": optimization_seconds,
        },
        "processors": benchmark_rows,
        "routing_sweep": grouped.to_dict(orient="records"),
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Classical optimum: value {classical['best_value']}, weight {classical['best_weight']}")
    print(f"Classical brute force: {classical['n_total']} subsets in {classical['time_s']:.6f}s")
    print(f"QAOA p=1 optimization: {optimization_seconds:.3f}s")
    print(benchmarks[[
        "processor", "swap_count", "cx_count", "depth", "P_optimal", "P_feasible",
        "approx_ratio",
    ]].to_string(index=False))
    print(f"Results written to {output_dir}")
    return summary, benchmarks, sweeps


if __name__ == "__main__":
    run_benchmark()
