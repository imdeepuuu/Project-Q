
import sys

sys.path.insert(0, r"files (1)")

from problem import qubo_unbalanced, optimize_qaoa, qaoa_circuit
from processors import load_processor
from routing import transpile_on
from qiskit_aer import AerSimulator

q = qubo_unbalanced()
r = optimize_qaoa(q, p=1, n_starts=3, maxiter=100, seed=42)

qc = qaoa_circuit(
    r["h"], r["J"], r["gammas"], r["betas"],
    scale=r["scale"], measure=True
)

proc = load_processor(r"files (1)\processor_A.json")
t, swaps, layout = transpile_on(qc, proc["cmap"])

sim = AerSimulator(noise_model=proc["noise"])
result = sim.run(t, shots=4096).result()
counts = result.get_counts()

print("PROCESSOR A NOISY SIMULATION")
print("Shots:", sum(counts.values()))
print("Circuit depth:", t.depth())
print("CX gates:", t.count_ops().get("cx", 0))
print("SWAPs:", swaps)
print("Top 5 outcomes:", sorted(
    counts.items(), key=lambda item: item[1], reverse=True
)[:5])
