"""Keep only what can carry input to output.  Cut by connectivity, not function.

The whole-brain principle is set aside here, deliberately and on the user's
instruction, because both dynamics models oscillate on the full graph: the
rate model swings -0.47..+0.78 for 800 ms after a bearing step, and the Shiu
et al. LIF does the same (+300 to -260 across 50 ms windows, bearing sweep not
monotonic).  Whatever causes that lives in the full recurrent graph, not in
the choice of neuron model.

The cut is mechanical:

    forward   neurons reachable from the designated INPUTS in <= k synapses
    backward  neurons that reach the designated OUTPUTS in <= k synapses
    keep      the intersection

No cell type is named, no pathway is identified, nothing is kept because of
what it is known to do.  A neuron survives only if a signal could travel from
an injected sensory neuron, through it, to a descending neuron within the hop
budget.  That is a statement about the graph, which is the same kind of
statement "inject at the lamina, read at descending neurons" already is.

What this gives up, and it is worth saying plainly: any loop longer than 2k
hops is cut, and if the real steering computation needs one, this removes it.
The claim becomes "a connectivity-defined subnetwork", never "the fly brain".

ponytail: unweighted BFS -- a single synapse counts the same as 2,405.  Add a
weight floor if the hop sets come out implausibly large.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp


def _reach(adj: sp.csr_matrix, seed: np.ndarray, hops: int) -> np.ndarray:
    """Boolean mask of everything reachable from `seed` within `hops`."""
    seen = np.zeros(adj.shape[0], dtype=bool)
    seen[seed] = True
    frontier = seen.copy()
    for _ in range(hops):
        # one BFS layer as a sparse matvec: cheaper than a Python queue here
        nxt = (adj.T @ frontier.astype(np.float32)) > 0
        frontier = nxt & ~seen
        if not frontier.any():
            break
        seen |= frontier
    return seen


def steering_subnetwork(out_csr: sp.csr_matrix, inputs: np.ndarray,
                        outputs: np.ndarray, hops: int = 3
                        ) -> Tuple[np.ndarray, dict]:
    """Indices to keep, plus what the cut did.

    `out_csr[i, j]` is the weight from i to j, so forward reach uses it as is
    and backward reach uses its transpose.
    """
    fwd = _reach(out_csr, inputs, hops)
    bwd = _reach(out_csr.T.tocsr(), outputs, hops)
    keep = fwd & bwd
    keep[inputs] = True          # the designated endpoints always survive
    keep[outputs] = True
    idx = np.flatnonzero(keep)
    sub = out_csr[idx][:, idx]
    return idx, {"hops": hops, "n_kept": len(idx),
                 "n_total": out_csr.shape[0],
                 "frac_kept": len(idx) / out_csr.shape[0],
                 "edges_kept": int(sub.nnz),
                 "edges_total": int(out_csr.nnz),
                 "frac_edges": sub.nnz / max(out_csr.nnz, 1),
                 "forward_only": int((fwd & ~bwd).sum()),
                 "backward_only": int((bwd & ~fwd).sum())}


def demo() -> None:
    """Check the cut keeps the endpoints and shrinks with fewer hops."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from core.malecns import load_malecns
    import sensors.flywire_eye as eye
    import pandas as pd

    ids, out, ann, _ = load_malecns(w_scale=1.0, symmetrise=True)
    sc = ann["super_class"].to_numpy(dtype="<U32")
    ctype = ann["cell_type"].astype(str)

    lat = eye.load_malecns_eye(ann)
    lam = pd.Index(ids).get_indexer(
        lat.root_id[lat.of_type(*eye.MALECNS_INJECT_TYPES)])
    orn = np.flatnonzero(ctype.isin(["ORN_VM2"]).to_numpy())
    inputs = np.unique(np.concatenate([lam[lam >= 0], orn]))
    outputs = np.flatnonzero(sc == "descending_neuron")
    print("inputs %d (lamina + ORN), outputs %d (descending)"
          % (len(inputs), len(outputs)))

    prev = None
    for k in (1, 2, 3, 4, 5):
        idx, info = steering_subnetwork(out, inputs, outputs, hops=k)
        print("hops %d: keep %6d / %d (%.1f%%), edges %8d (%.1f%%)"
              % (k, info["n_kept"], info["n_total"], 100 * info["frac_kept"],
                 info["edges_kept"], 100 * info["frac_edges"]))
        assert np.isin(inputs, idx).all(), "inputs must survive"
        assert np.isin(outputs, idx).all(), "outputs must survive"
        if prev is not None:
            assert info["n_kept"] >= prev, "more hops cannot keep fewer"
        prev = info["n_kept"]
    print("demo ok")


if __name__ == "__main__":
    demo()
