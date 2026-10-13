"""Plotting helpers: coupling graphs, benchmark bars, convergence, distributions, scaling."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

COL = {"A": "#2f5fe0", "B": "#2e9e4f", "ideal": "#555555", "random": "#bbbbbb", "opt": "#d9480f"}


def draw_coupling(edges, n, ax=None, title="", color="#2f5fe0", layout=None, pos=None):
    """Draw a coupling graph; `layout` (logical->physical) colours the qubits used by the 5-item problem."""
    ax = ax or plt.gca()
    g = nx.Graph(edges)
    g.add_nodes_from(range(n))
    pos = pos or nx.spring_layout(g, seed=7, k=1.2)
    used = set(layout.values()) if layout else set()
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#999999", width=2)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=[color if (not used or v in used) else "#dddddd" for v in g.nodes],
                           node_size=520, edgecolors="black")
    nx.draw_networkx_labels(g, pos, ax=ax, font_color="white", font_size=10, font_weight="bold")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return pos


def grouped_bars(df, metrics, group_col, ax_titles=None, colors=None, fname=None, ncols=None, suptitle=None):
    """df index = category (x), `group_col` selects bars within a category.  One panel per metric."""
    ncols = ncols or len(metrics)
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False)
    for ax, m in zip(axes.ravel(), metrics):
        piv = df.pivot(index="config", columns=group_col, values=m)
        piv.plot(kind="bar", ax=ax, color=[colors.get(c) for c in piv.columns] if colors else None,
                 edgecolor="black", rot=0, legend=False)
        ax.set_title(ax_titles.get(m, m) if ax_titles else m, fontsize=10)
        ax.set_xlabel("")
        for cnt in ax.containers:
            ax.bar_label(cnt, fmt="%.3g", fontsize=7, padding=1)
    axes.ravel()[0].legend(title=group_col, fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=160, bbox_inches="tight")
    return fig


def plot_convergence(traces: dict, fname=None, e_min=None, title="QAOA optimiser convergence (noise-free <C>)"):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for lab, tr in traces.items():
        run_min = np.minimum.accumulate(tr)
        ax.plot(run_min, label=lab, lw=2)
    if e_min is not None:
        ax.axhline(e_min, color="k", ls="--", lw=1, label="exact ground energy")
    ax.set_xlabel("COBYLA function evaluations")
    ax.set_ylabel("best <C> so far")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=160)
    return fig


def plot_distributions(dists: dict, qubo, opt_mask_items, top=10, fname=None):
    """Top-`top` item-bitstrings (marginal over slack) for each distribution; optimum highlighted."""
    ni = qubo.n_items
    fig, axes = plt.subplots(1, len(dists), figsize=(4.4 * len(dists), 3.8), sharey=True, squeeze=False)
    for ax, (lab, p) in zip(axes.ravel(), dists.items()):
        idx = np.arange(len(p)) & (2 ** ni - 1)
        marg = np.bincount(idx, weights=p, minlength=2 ** ni)
        order = np.argsort(marg)[::-1][:top]
        labels = [format(k, f"0{ni}b") for k in order]
        ax.bar(range(top), marg[order], color=[COL["opt"] if k in opt_mask_items else "#8aa" for k in order],
               edgecolor="black")
        ax.set_xticks(range(top))
        ax.set_xticklabels(labels, rotation=75, fontsize=7)
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="y", alpha=.3)
    axes.ravel()[0].set_ylabel("probability (item register x4..x0)")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=160)
    return fig


def plot_scaling(ns, t_classical, t_qaoa_sim=None, fname=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(ns, t_classical, "o-", label="classical brute force (measured, pure Python)")
    if t_qaoa_sim is not None:
        ax.semilogy(list(t_qaoa_sim.keys()), list(t_qaoa_sim.values()), "s--",
                    label="QAOA p=1 (state-vector simulation, 1 cost eval)")
    ax.set_xlabel("number of items n")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, which="both")
    ax.set_title("Brute-force scaling (2^n)")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=160)
    return fig
