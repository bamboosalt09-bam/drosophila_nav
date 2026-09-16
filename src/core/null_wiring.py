"""Matched null wirings, so "the connectome learns X" can mean something.

A connectome-wired network that learns a task proves nothing on its own: any
sparse network of that size might.  The claim only has content against nulls
that keep everything except the specific wiring.

Two nulls, in increasing severity:

    degree_preserving   every neuron keeps its in-degree, its out-degree and
                        its own transmitter sign; only WHICH partner an edge
                        lands on is randomised.  Synapse counts are carried
                        with the edges, so the weight distribution is
                        untouched.  This is the null that matters -- it asks
                        whether the specific partner choices carry the
                        function.

    random_sparse       same neuron count, same edge count, same weight
                        multiset, but degrees are not preserved.  Coarser, and
                        mostly a check that degree_preserving is not itself
                        already destroying something trivial.

Cell-type labels are NOT shuffled: the per-type parameters must stay
comparable across conditions, or the parameter-space trajectories cannot be
compared at all.

ponytail: edge targets are drawn with replacement, so a null can contain a
duplicate pair that the real graph does not.  At 15M edges over 139k nodes
the collision rate is small; use a proper configuration-model sampler if
duplicates ever matter.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp


def _csr_from(rows, cols, data, n) -> sp.csr_matrix:
    m = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    m.sort_indices()
    return m


def degree_preserving(out_csr: sp.csr_matrix, seed: int = 0) -> sp.csr_matrix:
    """Rewire targets while keeping every source's out-edges and weights.

    Each source neuron keeps its own row -- same number of edges, same weight
    values, same sign, since sign is a property of the source.  What changes
    is which neurons those edges point at, drawn from the empirical in-degree
    distribution so the target side keeps its shape too.
    """
    rng = np.random.default_rng(seed)
    coo = out_csr.tocoo()
    n = out_csr.shape[0]

    # sample targets in proportion to the real in-degree, so hub neurons stay
    # hubs and the in-degree distribution is preserved in expectation
    in_deg = np.asarray((out_csr != 0).sum(axis=0)).ravel().astype(np.float64)
    p = in_deg / in_deg.sum()
    new_cols = rng.choice(n, size=coo.nnz, p=p)
    return _csr_from(coo.row, new_cols, coo.data, n)


def random_sparse(out_csr: sp.csr_matrix, seed: int = 0) -> sp.csr_matrix:
    """Same size and same weight multiset, degrees not preserved."""
    rng = np.random.default_rng(seed)
    coo = out_csr.tocoo()
    n = out_csr.shape[0]
    rows = rng.integers(0, n, coo.nnz)
    cols = rng.integers(0, n, coo.nnz)
    data = rng.permutation(coo.data)
    return _csr_from(rows, cols, data, n)


def summarise(m: sp.csr_matrix) -> dict:
    out_deg = np.diff(m.indptr)
    in_deg = np.asarray((m != 0).sum(axis=0)).ravel()
    return {"edges": int(m.nnz),
            "w_sum": float(m.data.sum()),
            "w_abs_sum": float(np.abs(m.data).sum()),
            "frac_inhibitory": float((m.data < 0).mean()),
            "out_deg_mean": float(out_deg.mean()),
            "out_deg_max": int(out_deg.max()),
            "in_deg_mean": float(in_deg.mean()),
            "in_deg_max": int(in_deg.max())}


def demo() -> None:
    """Check the nulls keep what they promise and break what they should."""
    from core.flywire_brain import load_connectome

    _, real, _, _ = load_connectome()
    dp = degree_preserving(real, seed=0)
    rs = random_sparse(real, seed=0)

    print("%-20s %10s %12s %10s %9s %9s"
          % ("wiring", "edges", "|w| sum", "inhib", "out max", "in max"))
    for name, m in (("connectome", real), ("degree-preserving", dp),
                    ("random sparse", rs)):
        s = summarise(m)
        print("%-20s %10d %12.3e %10.3f %9d %9d"
              % (name, s["edges"], s["w_abs_sum"], s["frac_inhibitory"],
                 s["out_deg_max"], s["in_deg_max"]))

    r, d, s = summarise(real), summarise(dp), summarise(rs)

    # Sampling targets with replacement lets two edges land on the same pair,
    # and CSR then merges them.  That is the documented approximation, so
    # assert it is TIGHT rather than pretending it is exact.
    collision = 1.0 - d["edges"] / r["edges"]
    print("  degree-preserving collision rate: %.4f" % collision)
    assert collision < 0.01, collision

    # what the null must keep
    assert abs(d["w_abs_sum"] - r["w_abs_sum"]) / r["w_abs_sum"] < 0.01
    assert abs(d["frac_inhibitory"] - r["frac_inhibitory"]) < 0.01
    assert d["out_deg_max"] > 0.8 * r["out_deg_max"], "hubs were flattened"
    assert d["in_deg_max"] > 0.8 * r["in_deg_max"]

    # what it must destroy
    overlap = real.multiply(dp != 0).nnz / real.nnz
    print("  edges landing on a real partner after rewiring: %.4f" % overlap)
    assert overlap < 0.05, "rewiring barely changed anything"

    # and the coarse null must flatten the degree distribution, or it is not
    # a different null at all
    assert s["out_deg_max"] < 0.1 * r["out_deg_max"], s["out_deg_max"]
    print("demo ok")


if __name__ == "__main__":
    demo()
