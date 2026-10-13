"""Transpilation / layout / routing helpers (opt level 1, seed 42, basis {rz,sx,x,cx})."""
from __future__ import annotations

import numpy as np
from qiskit import transpile

from processors import BASIS_GATES, TRANSPILE_CFG


def transpile_on(qc, cmap, seed=TRANSPILE_CFG["seed_transpiler"],
                 opt=TRANSPILE_CFG["optimization_level"]):
    """Route + translate `qc` for coupling map `cmap`.

    SWAP overhead is measured with a twin transpilation whose basis additionally contains 'swap'
    (same seed => identical layout and routing); the swaps are then left undecomposed and can be
    counted exactly.  The deliverable circuit is transpiled to the pure {rz,sx,x,cx} basis.
    """
    t_swap = transpile(qc, coupling_map=cmap, basis_gates=BASIS_GATES + ["swap"],
                       optimization_level=opt, seed_transpiler=seed)
    t = transpile(qc, coupling_map=cmap, basis_gates=BASIS_GATES,
                  optimization_level=opt, seed_transpiler=seed)
    swaps = int(t_swap.count_ops().get("swap", 0))
    layout = ({i: int(p) for i, p in enumerate(t.layout.initial_index_layout(filter_ancillas=True))}
              if t.layout is not None else None)  # logical -> physical
    return t, swaps, layout


def routing_sweep_seeds(qc, cmap, seeds=range(20), opt=1):
    """Routing is stochastic (SABRE): distribution of swap / cx / depth over many seeds."""
    rows = []
    for s in seeds:
        t, sw, _ = transpile_on(qc, cmap, seed=s, opt=opt)
        ops = t.count_ops()
        rows.append({"seed": s, "swaps": sw, "cx": int(ops.get("cx", 0)), "depth": t.depth()})
    return rows
