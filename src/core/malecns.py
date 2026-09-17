"""MaleCNS: the same pipeline, on a connectome that includes the nerve cord.

FAFB is brain-only, so it has no motor neurons and no ventral nerve cord.
That forced the steering command to be invented from 1,303 mixed descending
neurons, and it is why "does the output go to the legs or the wings" had no
answer.  MaleCNS v1.0 contains both, so the readout stops being a modelling
choice: motor neurons are in the data, with their muscles and leg segments
named.

    vnc_motor 708, cb_motor 107, vnc_efferent 94   (L 423 / R 422 / M 63)
    somaNeuromere T1 182, T2 186, T3 156           (the three leg pairs)
    types "Ti flexor MN", "Fe reductor MN", ...     (named muscles)
    vnc_intrinsic 13,161                           (the CPG, as data)
    R1-R6 3,377                                    (absent from FAFB)
    somaSide L 75,215 / R 75,119                   (FAFB was 10% lopsided)

Source: storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
CC-BY.  The neuPrint web app wants an account; the bucket does not.

This returns exactly what core.flywire_brain.load_connectome returns, with
the annotation columns renamed to the same names, so the rate model, the null
wirings and the training loop all work unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as ft
import scipy.sparse as sp

from core.flywire_brain import balance_hemispheres

DATA = Path("data/malecns")
ANN_FILE = DATA / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT_FILE = DATA / "body-neurotransmitters-male-cns-v1.0.feather"
CONN_FILE = DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# Shiu et al.'s rule, unchanged: ACh/DA/OA/5-HT excitatory, GABA/Glu inhibitory
NT_SIGN = {"acetylcholine": 1.0, "dopamine": 1.0, "octopamine": 1.0,
           "serotonin": 1.0, "gaba": -1.0, "glutamate": -1.0}

W_SYN_MV = 0.275        # Shiu et al. 2024 model.py

# The published table has 151.9M rows at >= 1 synapse.  Five is the threshold
# the whole-fly Minecraft simulation reports using, and it leaves 7.6M edges
# -- half of FAFB's 15.1M, so the simulation is FASTER, not slower.
DEFAULT_MIN_SYNAPSES = 5

_SIDE = {"L": "left", "R": "right", "M": "center"}


def _load_annotations() -> pd.DataFrame:
    """Annotation columns, renamed to the names the rest of the code uses."""
    a = ft.read_table(ANN_FILE, memory_map=True).to_pandas()
    nt = ft.read_table(NT_FILE, columns=["body", "consensus_nt"],
                       memory_map=True).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")["consensus_nt"]

    out = pd.DataFrame({
        "root_id": a["bodyId"].astype(np.int64),
        "super_class": a["superclass"],
        "cell_class": a["class"],
        "cell_type": a["type"],
        "side": a["somaSide"].map(_SIDE),
        "top_nt": a["bodyId"].map(nt),
        # kept because they are the point of using this dataset at all
        "neuromere": a["somaNeuromere"],
        "exit_nerve": a["exitNerve"],
        "group": a["group"],
        "hex1": a["assignedOlHex1"],
        "hex2": a["assignedOlHex2"],
    })
    for col in ("super_class", "cell_class", "cell_type", "side", "top_nt",
                "neuromere", "exit_nerve", "group"):
        out[col] = out[col].fillna("").astype(str)
    # 211,577 rows, but only 166,700 carry a superclass -- exactly the neuron
    # count this release reports.  The remaining 44,877 are Orphan, Glia,
    # Unimportant and Out-of-scope bodies, which are not neurons.
    return out[out["super_class"] != ""].reset_index(drop=True)


def mirror_map(ann: pd.DataFrame) -> np.ndarray:
    """Index of each neuron's left-right counterpart, or -1 where there is none.

    MaleCNS ships the correspondence FAFB lacked: `group` collects a cell with
    its bilateral partner, and 11,762 groups hold equal left and right counts.
    Within such a group the pairing is by sorted bodyId -- arbitrary for groups
    larger than one per side, exact for the 9,404 groups that are a single
    pair.

    Coverage is concentrated where the asymmetry actually hurts: vnc_motor
    94%, ascending 92%, descending 89%, vnc_intrinsic 79%, cb_intrinsic 64%.
    The optic lobe is only 5.6%, which is what the hex column assignment is
    for.
    """
    grp = ann["group"].to_numpy(dtype="<U32")
    side = ann["side"].to_numpy(dtype="<U16")
    mirror = np.full(len(ann), -1, dtype=np.int64)
    order = np.argsort(grp, kind="stable")
    g_sorted = grp[order]
    bounds = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if g_sorted[a] == "":
            continue
        members = order[a:b]
        left = np.sort(members[side[members] == "left"])
        right = np.sort(members[side[members] == "right"])
        if len(left) and len(left) == len(right):
            mirror[left] = right
            mirror[right] = left

    # The optic lobe is barely grouped (5.6%) and is where the global
    # asymmetry lives -- measured R/L 1.239 against 1.096 for the whole
    # network.  It does carry an explicit lattice, so pair it the way FAFB's
    # (p, q) assignment was used: same cell type, same hex cell, other side.
    hex1 = ann["hex1"].to_numpy()
    hex2 = ann["hex2"].to_numpy()
    ctype = ann["cell_type"].to_numpy(dtype="<U48")
    has_hex = np.isfinite(hex1) & np.isfinite(hex2) & (mirror < 0)
    if has_hex.any():
        idx = np.flatnonzero(has_hex)
        key = pd.MultiIndex.from_arrays(
            [ctype[idx], hex1[idx].astype(np.int64), hex2[idx].astype(np.int64)])
        df = pd.DataFrame({"i": idx, "side": side[idx]}, index=key)
        for _, sub in df.groupby(level=[0, 1, 2], sort=False):
            left = np.sort(sub.loc[sub["side"] == "left", "i"].to_numpy())
            right = np.sort(sub.loc[sub["side"] == "right", "i"].to_numpy())
            if len(left) and len(left) == len(right):
                mirror[left] = right
                mirror[right] = left
    return mirror


def mirror_average(out_csr: sp.csr_matrix, mirror: np.ndarray) -> sp.csr_matrix:
    """Average every edge with its mirror image, where both ends have one.

    This is not a gain that hides the imbalance -- it makes the two sides
    literally equal wherever the data says which cell pairs with which.
    Edges with an unmirrored endpoint are left untouched, so the fraction
    corrected is exactly the fraction the annotation supports.

    Known biological asymmetries exist in the fly -- the asymmetrical body is
    1679 um^3 on the right against 526 on the left -- so this must be reported
    as applied, not assumed to be free of cost.
    """
    m = out_csr.tocoo()
    has = mirror >= 0
    both = has[m.row] & has[m.col]
    n = out_csr.shape[0]

    a = sp.csr_matrix((m.data[both], (m.row[both], m.col[both])), shape=(n, n))
    # mirror is an involution, so adding the relabelled copy and halving gives
    # each edge the mean of itself and its counterpart
    am = sp.csr_matrix((m.data[both],
                        (mirror[m.row[both]], mirror[m.col[both]])),
                       shape=(n, n))
    rest = sp.csr_matrix((m.data[~both], (m.row[~both], m.col[~both])),
                         shape=(n, n))
    res = ((a + am) * 0.5 + rest).tocsr()
    res.sort_indices()
    return res.astype(np.float32)


def load_malecns(min_synapses: int = DEFAULT_MIN_SYNAPSES,
                 w_scale: float = 1.0, symmetrise: bool = False
                 ) -> Tuple[np.ndarray, sp.csr_matrix, pd.DataFrame, int]:
    """Ids, signed weight matrix, annotations -- same contract as FAFB's loader.

    Row = source neuron, so one row holds that neuron's out-edges.

    ponytail: the connectivity column is read and filtered one column at a
    time.  Reading all three int64 columns of a 151.9M-row table at once is
    3.6 GB, which this machine does not have spare.
    """
    ann = _load_annotations()

    tbl = ft.read_table(CONN_FILE, memory_map=True)
    w_all = tbl.column("weight")
    keep = pc.greater_equal(w_all, min_synapses)
    syn = pc.filter(w_all, keep).to_numpy().astype(np.int32)
    pre = pc.filter(tbl.column("body_pre"), keep).to_numpy().astype(np.int64)
    post = pc.filter(tbl.column("body_post"), keep).to_numpy().astype(np.int64)
    del tbl, w_all, keep

    # The connectivity table names bodies that are not in the annotation --
    # fragments and unproofread segments.  Taking the union gives 1.48M
    # "neurons" against the dataset's ~150k, so keep only annotated bodies at
    # BOTH ends of an edge.
    ids = np.unique(ann["root_id"].to_numpy(np.int64))
    idx = pd.Index(ids)
    i_pre, i_post = idx.get_indexer(pre), idx.get_indexer(post)
    both = (i_pre >= 0) & (i_post >= 0)
    n_dropped = int((~both).sum())
    i_pre, i_post, syn = i_pre[both], i_post[both], syn[both]

    ann_i = ann.set_index("root_id").reindex(ids)
    for col in ("super_class", "cell_class", "cell_type", "side", "top_nt",
                "neuromere", "exit_nerve", "group"):
        ann_i[col] = ann_i[col].fillna("").astype(str)

    # sign is a property of the presynaptic NEURON, one cell one transmitter
    sign_by_neuron = (ann_i["top_nt"].map(NT_SIGN)
                      .fillna(0.0).to_numpy(np.float32))
    sign = sign_by_neuron[i_pre]
    n_unsigned = int((sign == 0).sum())

    w = (sign * syn * W_SYN_MV * w_scale).astype(np.float32)
    out = sp.csr_matrix((w, (i_pre, i_post)), shape=(len(ids), len(ids)),
                        dtype=np.float32)
    out.sort_indices()
    if symmetrise:
        # Two methods composed, in the order their evidence supports.
        #
        # Pedigo et al. (eLife) study exactly this and offer two: rescaling
        # connection probabilities, or dropping weak edges.  Dropping weak
        # edges FAILS here -- measured, R/L stays 1.05-1.10 from a threshold
        # of 1 through 50, so the imbalance is spread across the whole weight
        # spectrum rather than hiding in the tail.
        #
        # So: mirror-average wherever the annotation says which cell pairs
        # with which, which makes the two sides literally equal and covers
        # 75-88% of motor, descending and nerve-cord neurons; then rescale the
        # residue per anatomical stage, because the optic lobe is only 27%
        # pairable and is where the global asymmetry lives (R/L 1.24).
        out = mirror_average(out, mirror_map(ann_i))
        out = balance_hemispheres(
            out, ann_i["side"].to_numpy(dtype="<U16"),
            group=ann_i["super_class"].to_numpy(dtype="<U32"))
    return ids, out, ann_i, n_unsigned


def demo() -> None:
    """Check the graph builds and that the motor side is really there."""
    import time

    t0 = time.perf_counter()
    ids, out, ann, n_unsigned = load_malecns()
    print("built %d neurons, %d edges in %.1f s (unsigned %d)"
          % (len(ids), out.nnz, time.perf_counter() - t0, n_unsigned))
    assert len(ids) == 166700, "expected the release's 166,700 neurons"

    sc = ann["super_class"].to_numpy(dtype="<U32")
    side = ann["side"].to_numpy(dtype="<U16")
    motor = np.flatnonzero(np.isin(sc, ["vnc_motor", "cb_motor",
                                        "vnc_efferent"]))
    desc = np.flatnonzero(sc == "descending_neuron")
    vnc = np.flatnonzero(sc == "vnc_intrinsic")
    print("  motor %d (L %d / R %d), descending %d, vnc_intrinsic %d"
          % (len(motor), (side[motor] == "left").sum(),
             (side[motor] == "right").sum(), len(desc), len(vnc)))
    assert len(motor) > 800, "the point of this dataset is the motor neurons"
    assert len(vnc) > 10000, "and the nerve cord circuitry"

    nm = ann["neuromere"].to_numpy(dtype="<U16")[motor]
    print("  motor neurons by neuromere: %s"
          % pd.Series(nm).value_counts().head(4).to_dict())
    assert {"T1", "T2", "T3"} <= set(nm), "the three leg pairs must be present"

    # signs, and that the graph is not trivially disconnected
    print("  excitatory %.1f%% / inhibitory %.1f%%"
          % (100 * (out.data > 0).mean(), 100 * (out.data < 0).mean()))
    in_deg = np.asarray((out != 0).sum(axis=0)).ravel()
    print("  motor neurons with incoming edges: %d of %d"
          % (int((in_deg[motor] > 0).sum()), len(motor)))
    assert (in_deg[motor] > 0).sum() > 0.8 * len(motor)

    # laterality, the thing that cost a long detour on FAFB
    absm = abs(out)
    inc = np.asarray(absm.sum(axis=0)).ravel()
    l = inc[side == "left"].sum()
    r = inc[side == "right"].sum()
    print("  incoming |weight| left %.4g / right %.4g  ->  R/L %.4f"
          % (l, r, r / l))
    print("demo ok")


if __name__ == "__main__":
    demo()
