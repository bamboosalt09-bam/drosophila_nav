"""Recover the column lattice from CONNECTIVITY instead of coordinates.

Coordinates gave a sheet but not a lattice (nearest-neighbour CV 0.6-0.75,
~2 neighbours where a hexagonal lattice needs ~6).  The connectome itself
should carry the retinotopy: Mi1 cells in neighbouring columns share
downstream and upstream partners, distant ones do not.

If the shared-partner graph is locally structured -- each Mi1 strongly
overlapping with a small, consistent number of others -- then the lattice is
recoverable and a direction map can be laid on it.
"""
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp


def log(m):
    sys.stdout.write(m + "\n")
    sys.stdout.flush()


ann = pd.read_csv("data/flywire/neuron_annotations_783.tsv", sep="\t",
                  low_memory=False)
e = np.load("data/flywire/edges_783.npz")
pre, post, syn = e["pre"], e["post"], e["syn"]
log("edges %d" % len(pre))

MIN_SYN = 5          # ignore the long tail of 1-2 synapse edges for this probe

for side in ("left", "right"):
    mi1 = ann[(ann.cell_type.astype(str) == "Mi1") & (ann.side == side)]
    ids = mi1.root_id.to_numpy()
    idset = pd.Index(ids)
    log("")
    log("=== %s eye: %d Mi1 ===" % (side, len(ids)))

    keep = syn >= MIN_SYN
    p, q = pre[keep], post[keep]

    # partners of each Mi1, in both directions
    m_out = idset.get_indexer(p)
    m_in = idset.get_indexer(q)
    rows, partners = [], []
    sel = m_out >= 0
    rows.append(m_out[sel]); partners.append(q[sel])
    sel = m_in >= 0
    rows.append(m_in[sel]); partners.append(p[sel])
    rows = np.concatenate(rows)
    partners = np.concatenate(partners)
    log("  Mi1-incident edges (>= %d syn): %d" % (MIN_SYN, len(rows)))

    # Mi1 x partner incidence, then Mi1 x Mi1 shared-partner counts
    pidx, puniq = pd.factorize(partners)
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, pidx)),
                      shape=(len(ids), len(puniq)))
    A.data[:] = 1.0
    S = (A @ A.T).toarray()
    np.fill_diagonal(S, 0)
    log("  partners per Mi1: median %.0f" % np.median(A.getnnz(axis=1)))

    # how concentrated is each Mi1's overlap?
    order = np.argsort(-S, axis=1)
    top = np.take_along_axis(S, order, axis=1)
    log("  shared partners with 1st/3rd/6th/12th/30th best match: "
        "%.1f / %.1f / %.1f / %.1f / %.1f"
        % tuple(np.median(top[:, k]) for k in (0, 2, 5, 11, 29)))

    # a lattice signature: a clear drop after ~6 neighbours
    ratio = np.median(top[:, 5]) / max(np.median(top[:, 11]), 1e-9)
    log("  6th/12th ratio %.2f  (a hexagonal lattice gives a clear step > 1)"
        % ratio)

    # spectral embedding of the overlap graph -> is it a 2D sheet?
    W = S.copy()
    thr = np.partition(W, -8, axis=1)[:, -8][:, None]
    W = np.where(W >= thr, W, 0.0)
    W = np.maximum(W, W.T)
    d = W.sum(1)
    good = d > 0
    L = np.diag(d[good]) - W[np.ix_(good, good)]
    dm = np.diag(1.0 / np.sqrt(d[good]))
    vals, vecs = np.linalg.eigh(dm @ L @ dm)
    log("  normalised-Laplacian eigenvalues 1..6: %s"
        % np.array2string(vals[1:7], precision=4))
    log("  (a 2D sheet gives two small, well-separated non-zero eigenvalues)")

    emb = vecs[:, 1:3]
    from scipy.spatial import cKDTree
    emb = emb / np.abs(emb).max()
    dd, _ = cKDTree(emb).query(emb, k=7)
    m = np.median(dd[:, 1])
    cnt = np.array([len(x) - 1
                    for x in cKDTree(emb).query_ball_point(emb, 1.5 * m)])
    log("  embedded neighbours within 1.5x median spacing: %.1f" % cnt.mean())
