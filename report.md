# PS-O3 — Resource-Constrained Item Selection with QAOA
Qiskit Fall Fest 2026, Phase 2, Optimization Track · Team KanyaRaasi

All numbers below are produced by `main.ipynb` with the official settings (basis rz/sx/x/cx, optimization_level 1, SABRE, seeds 42, 4096 shots, official noise model).

## 1. Problem and formulation

Five items with values v = [10, 4, 7, 5, 8] and weights w = [5, 3, 4, 2, 6] must be packed into capacity W = 15 to maximize total value. Item n of the problem statement is qubit n − 1.

The constraint Σ wᵢxᵢ ≤ 15 is turned into an equality with a binary slack s = s₀ + 2s₁ + 4s₂ + 8s₃ on qubits 5–8, and enforced with the locked penalty P = 5:

min −Σ vᵢxᵢ + P(Σ wᵢxᵢ + s − 15)²

Writing all nine bits as z with coefficients a = [5, 3, 4, 2, 6, 1, 2, 4, 8] and using zᵢ² = zᵢ gives the QUBO zᵀQz + 1125 with Qᵢᵢ = P(aᵢ² − 30aᵢ) − vᵢ and Qᵢⱼ = 2P·aᵢaⱼ (i < j). The full 9×9 matrix is printed in the notebook and saved in `results/tables/qubo_9x9.json`.

Because every aᵢ > 0, **every pair of variables is coupled**: the cost graph is the complete graph K₉ with 36 edges. This is the fact that drives the whole hardware discussion.

**Classical benchmark.** Exhaustive search over the 32 item subsets gives V\* = 26 with items {1, 2, 3, 4} (weight 14), the unique optimum. 28 of 32 subsets are feasible. Brute force over all 512 QUBO bitstrings gives the minimum −26 at the same items with slack s = 1, so the QUBO's ground state is the true optimum at P = 5.

**Random baseline.** Sampling item bits uniformly gives P_feasible = 0.875 and P_opt = 1/32 = 0.031. Because most subsets fit, P_feasible is a demanding baseline, and we compare against it explicitly.

**Cost Hamiltonian.** With zᵢ = (1 − Zᵢ)/2 the QUBO becomes 9 Z terms + 36 ZZ terms + a constant (46 Pauli terms). Its diagonal matches the QUBO on all 512 bitstrings. The largest non-identity coefficient is 120, so we use H/120 inside QAOA, as the kit recommends; all γ values below are in units of this scaled Hamiltonian.

## 2. Bell-state verification (Objective 1)

|Φ⁺⟩ was prepared with H(0) and CX(0→1) and measured in the ZZ, XX and YY bases (4096 shots each) on Processor D. The farthest pair on the torus is at distance 2; we used (0, 4).

| Pair | Distance | Condition | CX | ⟨XX⟩ | ⟨YY⟩ | ⟨ZZ⟩ | F |
|---|---|---|---|---|---|---|---|
| 0–1 | 1 | ideal | 1 | +1.000 ± 0.000 | −1.000 ± 0.000 | +1.000 ± 0.000 | 1.0000 ± 0.0000 |
| 0–1 | 1 | noisy | 1 | +0.946 ± 0.005 | −0.952 ± 0.005 | +0.955 ± 0.005 | 0.9633 ± 0.0021 |
| 0–4 | 2 | ideal | 4 | +1.000 ± 0.000 | −1.000 ± 0.000 | +1.000 ± 0.000 | 1.0000 ± 0.0000 |
| 0–4 | 2 | noisy | 4 | +0.945 ± 0.005 | −0.936 ± 0.005 | +0.939 ± 0.005 | 0.9548 ± 0.0023 |

The ideal runs reach the exact values (XX = +1, YY = −1, ZZ = +1), confirming the circuits and bases. With noise, the adjacent pair loses about 3.7% fidelity. Most of that comes from readout: a 1% bit flip on each of two qubits turns about 2% of shots from even to odd parity, which lowers each correlator by about 0.04. The rest comes from depolarizing on the CX and the single-qubit gates. The distant pair needs a SWAP, so the transpiled circuit has 4 CX instead of 1; the three extra noisy CX cost another 0.0085 in fidelity, a difference of about 2.7 standard errors. Distance on the coupling graph turns directly into extra CX and extra error. That is why D was designed to keep every pair within distance 2.

## 3. Processor D design (Objective 2)

**Bottleneck on B.** On Processor B the bound p = 1 circuit needs 72 logical CX but 121 after routing: 17 SWAPs, 49 CX overhead (68% extra), depth 117 and two-qubit depth 79. B has diameter 4 and an uneven degree profile (degrees 1 to 5, one leaf qubit), while our cost graph is K₉, so many of the 36 interactions are between distant qubits.

**Design.** Processor D is a 3×3 torus: the 3×3 grid plus wrap-around edges on each row and column. It has 9 qubits and 18 edges, every qubit has degree exactly 4 (the maximum allowed), and the diameter is 2. It passes `validate_processor_d` and uses the official noise rates.

![Processors B and D](results/figures/processors_B_D.png)

**How we chose it.** We measured the actual bound QAOA circuit with `circuit_metrics()` on every valid candidate we could generate: circulant graphs C_n(j₁, j₂) on 9–12 qubits, about 130 random 4-regular graphs on 9 and 10 qubits, and 3×4, 2×5 and 2×6 tori.

| Candidate | CX | Depth | SWAPs |
|---|---|---|---|
| Processor B (reference) | 121 | 117 | 17 |
| **3×3 torus (chosen)** | **89** | **76** | **9** |
| best random 4-regular, 9 qubits | 94 | 80 | 8 |
| circulant C₉(1, 3) | 99 | 90 | 9 |
| 3×4 torus, 12 qubits | 108 | 109 | 12 |

Extra qubits did not help: with K₉ every logical qubit has to interact with every other, so the router gains more from short distances than from spare qubits.

**Expected effect vs measured.** We expected D to roughly halve the SWAPs and cut CX by about a quarter, which should reduce the noise penalty without changing the ideal result. Measured on the same angles:

| | B | D | Change |
|---|---|---|---|
| CX | 121 | 89 | −26% |
| CX overhead | 49 | 17 | −65% |
| SWAPs | 17 | 9 | −47% |
| Depth | 117 | 76 | −35% |
| Two-qubit depth | 79 | 43 | −46% |
| Mean QUBO cost, ideal | 48.9 | 52.3 | sampling difference only |
| Mean QUBO cost, noisy | 83.0 | 73.5 | −11% |
| Noise penalty (noisy − ideal) | +34.1 | +21.2 | −38% |
| P_feasible, noisy | 0.943 | 0.943 | equal |

The ideal rows prepare the same state on both processors, so their difference is shot noise. The noise-induced rise in mean QUBO cost is 38% smaller on D, consistent with 26% fewer CX and a 46% smaller two-qubit depth. P_feasible is the same on both, because feasibility is a coarse property that survives this level of noise. The gain from D shows up in the energy, not in the feasibility rate.

**Trade-off.** The torus needs wrap-around couplers between opposite edges of the grid. On a planar superconducting chip these are long connections that are hard to fabricate and add crosstalk, and giving every qubit degree 4 raises crosstalk and frequency-crowding risk compared with B. The design is also tuned to dense, all-to-all cost graphs; for a sparse problem such as a chain, a simpler lattice would do as well with fewer couplers.

## 4. QAOA and benchmarking (Objective 3)

**Setup.** p = 1, circuit = H on all qubits, cost layer exp(−iγ H/120) as 36 RZZ + 9 RZ, mixer RX(2β). Optimizer COBYLA (scipy), maxiter 200, cost evaluated with the exact statevector (no shots, no noise). Five initializations from `default_rng(seed)`, seeds 0–4, γ and β uniform in [0, π]. As the kit FAQ says, angles are optimized once in ideal simulation and the same angles are evaluated on every processor.

| Seed | Start (γ, β) | Final (γ, β) | ⟨H⟩ | Exact P_feasible | Exact P_opt |
|---|---|---|---|---|---|
| 0 | (2.001, 0.848) | (2.537, 1.518) | 232.0 | 0.873 | 0.032 |
| **1** | (1.608, 2.986) | **(0.283, 2.865)** | **50.2** | **0.955** | 0.023 |
| 2 | (0.822, 0.938) | (3.016, 3.088) | 232.3 | 0.876 | 0.032 |
| 3 | (0.269, 0.744) | (0.284, 1.354) | 130.6 | 0.892 | 0.039 |
| 4 | (2.963, 1.606) | (4.060, 1.599) | 232.9 | 0.875 | 0.031 |

Mean ± std over the five seeds: P_feasible 0.894 ± 0.031, P_opt 0.031 ± 0.005. Three of five seeds stopped at ⟨H⟩ ≈ 232, which is the energy of the uniform superposition: starting at γ > 0.8, they landed on the flat plateau of the landscape (Section 5), where COBYLA sees almost no gradient. Seed 3 found a secondary valley near β ≈ 1.35. Seed 1 found the global valley, and its angles are used for all benchmark rows.

**Results (4096 shots, best angles).**

| Processor | Condition | P_feasible | P_opt | Best feasible (items, weight) | % of V\* |
|---|---|---|---|---|---|
| A | both | not executable: needs 9 qubits, has 5 | | | |
| B | ideal | 0.959 | 0.021 | 26 ({1,2,3,4}, 14) | 100 |
| B | noisy | 0.943 | 0.025 | 26 ({1,2,3,4}, 14) | 100 |
| D | ideal | 0.956 | 0.023 | 26 ({1,2,3,4}, 14) | 100 |
| D | noisy | 0.943 | 0.023 | 26 ({1,2,3,4}, 14) | 100 |
| random | — | 0.875 | 0.031 | — | — |

![Quality vs random](results/figures/quality_vs_random.png)

**Capacity limit on A.** The required encoding needs 9 qubits and A has 5. Even the 5 item bits alone would leave no room for slack, so this instance cannot be encoded on A at all; it would need either more qubits or a slack-free encoding (Section 6).

## 5. Objective landscape on Processor D

Grid 20 × 20, β ∈ [0, π], γ ∈ [0, 1.0] in units of H/120 (so γ_max = 1/120 for the unscaled Hamiltonian), sampled mean QUBO cost from 1024 noisy shots per point on D.

![Landscape](results/figures/landscape_D_noisy.png)

We chose γ_max = 1.0 after an ideal coarse scan over γ ∈ [0, π]: for γ > 1 the cost layer gives the 46 terms large, unrelated phases, the interference averages out and the energy flattens towards the uniform-superposition value (about 232). The cost is periodic in β with period π (the mixer is RX(2β)), and at β = 0 or π/2 it equals the uniform value for every γ. Between those lines it alternates between bright bands (β ≈ 0.5 and 2.1), where probability is pushed towards high-penalty states, and dark valleys (β ≈ 1.3 and 2.8), where it is pushed towards low-penalty ones. The grid minimum, 66.6 at (γ, β) = (0.263, 2.811), sits next to our optimized point (0.283, 2.865), so the noisy landscape and the ideal optimizer agree. The narrow γ window of the good valley (about 0.15–0.45) explains why seeds that started at large γ never found it.

## 6. Analysis, limitations and next steps

**Feasibility vs quality.** QAOA beats random clearly on feasibility: 0.943 noisy on both B and D vs 0.875, which halves the infeasible fraction (12.5% → 5.7%). It does **not** beat random on P_opt (0.023 vs 0.031). The reason is the scale of the terms: the penalty coefficients reach 880 while item values are at most 10, so the cost Hamiltonian is almost entirely penalty. At p = 1, QAOA moves probability towards low-penalty states, which includes every feasible subset with its matching slack, but it hardly distinguishes value 26 from value 22 inside the feasible set. The best sampled solution is still the optimum on every executable processor (100% of V\*), because the optimum is sampled about 95 times in 4096 shots.

**Architecture.** Connectivity is the dominant resource factor for this problem because the cost graph is K₉. Moving from B to D removed 32 CX and 41 layers of depth, and reduced the noise penalty on the mean cost by 38%. Feasibility did not change, so the benefit of D appears in energy, not in P_feasible, at the official noise level.

**Limitations.**
- p = 1 only; one optimizer; angles optimized in ideal simulation and reused on noisy runs.
- 3 of 5 initializations got stuck on the plateau, so a single run is not reliable without a good starting γ.
- The slack encoding spends 4 of 9 qubits on bookkeeping and adds 26 of the 36 couplings.
- Results are from a uniform depolarizing model; real devices have non-uniform, correlated noise.

**Next improvements (additional experiments, not required runs).**
- Draw γ from [0, 0.5] (or warm-start from a coarse grid) so every initialization lands in the useful valley.
- Use a CVaR objective (average of the best α fraction of shots), which rewards concentrating mass on the optimum rather than lowering the mean penalty.
- Try smaller P or an unbalanced-penalization encoding without slack; that needs only 5 qubits, so it would even fit Processor A and remove most of the routing cost.
- Run p = 2 and compare the CX cost against the P_opt gain.
- Re-optimize angles under noise on each processor and test D with improved, cited noise rates.
