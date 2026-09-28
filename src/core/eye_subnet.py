"""Lamina in, descending neurons out, WITH the optic lobe.

The optic lobe was cut on one measurement: a wall closing from 12 m to 2 m on
the LEFT made the steering readout go more positive -- turn into it -- while
growing 50-fold.  That was read as the rate model getting motion detection
wrong, and the injection moved one stage downstream to the visual projection
neurons instead.

Three defects have since been found in the picture that measurement was made
on, and each one is sufficient to produce that result:

  * the lamp was added into the scene, so a wall at 3 m was BRIGHTER than
    sky and attracted.  Measured bearing to a wall at +45 deg: -44.2 at 5 m,
    +10.4 at 3 m, +42.5 at 1.5 m -- the sign inverting at exactly the range
    the optic-lobe test used.
  * the surface point was computed as p - t*d instead of p + t*d, so the
    Lambertian normal belonged to the point reflected through the camera.
  * the drive was one column held for all 20 sub-steps, i.e. a photograph.
    A motion-detecting stage fed a still image has nothing to detect.

So the optic lobe has not actually been tested on a correct picture, and
the cost argument has weakened too: int32 CSR indices and frozen per-neuron
parameters took 20 sub-steps from 79 ms to 39 ms, and the sensing sweep went
from 33 ms to 2.5 ms.

WHAT THIS BUYS IF IT WORKS.  With the optic lobe present the circuit can in
principle compute looming itself, from the video it now receives, instead of
being handed a rangefinder.  The claim moves from "the connectome navigates
from projection-neuron activity" to "from its eyes".

WHAT WOULD MAKE IT FAIL.  Injection drops from 9,188 projection neurons to
6,199 lamina cells.  A Reichardt correlator is a delay-and-multiply, and
there is no guarantee that survives `tanh(relu(v))` with the spectral radius
rescaled.  And rescaling now normalises a different matrix, so the effective
time constant tau/(1-rho) changes -- which is what produced the original
closed-loop sign inversion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.vp_subnet import _reach, receptive_directions


def build_eye_subnet(rho_target: float = 0.50, hops: int = 4,
                     cache: bool = True):
    """Returns (net, ann, sub, info), same shape as `build_vp_subnet`."""
    import time

    import scipy.sparse.linalg as spl
    from core import subnet_file
    from core.flywire_rate import FlyWireRate
    from core.malecns import load_malecns

    # Names the injection, not an exclusion.  NO COLON: it used to read
    # "eye:lamina-in", and on NTFS a colon in a file name opens an alternate
    # data stream, so the 13 MB cache landed in a hidden stream behind a
    # 0-byte file.  It loaded fine and would have vanished on the first copy.
    drop = ("eye-lamina-in",)
    if cache:
        hit = subnet_file.load(hops, rho_target, drop)
        if hit is not None:
            sub, sids, sann, info = hit
            return (FlyWireRate(out_csr=sub, ids=sids, ann=sann),
                    sann, sub, info)

    ids, out, ann, _ = load_malecns(w_scale=1.0, symmetrise=True)
    sc = ann["super_class"].to_numpy(dtype="<U32")
    az, el, lam = receptive_directions(out, ids, ann)
    dn = np.flatnonzero(sc == "descending_neuron")

    t = time.perf_counter()
    # Forward from the LAMINA, not from the projection neurons, and nothing
    # is excluded: the optic lobe is the point.
    keep = _reach(out, lam, hops) & _reach(out.T.tocsr(), dn, hops)
    keep[lam] = True
    keep[dn] = True
    idx = np.flatnonzero(keep)
    sub0 = out[idx][:, idx].tocsr()
    rho = float(np.abs(spl.eigs(sub0.astype(np.float64), k=1,
                                return_eigenvectors=False)[0]))
    print("  cut+eigs %.0f s, %d neurons, rho %.1f"
          % (time.perf_counter() - t, len(idx), rho), flush=True)

    sub = (sub0 * (rho_target / rho)).tocsr()
    sann = ann.iloc[idx].copy()
    net = FlyWireRate(out_csr=sub, ids=ids[idx], ann=sann)

    sc_s = sann["super_class"].to_numpy(dtype="<U32")
    side_s = sann["side"].to_numpy(dtype="<U16")
    pos = np.full(out.shape[0], -1, dtype=np.int64)
    pos[idx] = np.arange(len(idx))
    lam_rows = pos[lam]
    lam_rows = lam_rows[lam_rows >= 0]
    # The index IS the root_id.  A `reset_index(drop=True)` here once made
    # 1,779 of 6,199 lamina cells point at unrelated neurons and destroyed
    # retinotopy silently, so this is checked rather than assumed.
    if len(lam_rows) < 0.9 * len(lam):
        raise ValueError("only %d of %d lamina cells survived the cut"
                         % (len(lam_rows), len(lam)))
    info = {
        "rho_raw": rho, "n_edges": int(sub.nnz),
        "lam_rows": lam_rows,
        "lam_ids": ids[idx][lam_rows],
        "vp_rows": np.flatnonzero(sc_s == "visual_projection"),
        "vp_az": az[idx][np.flatnonzero(sc_s == "visual_projection")],
        "vp_el": el[idx][np.flatnonzero(sc_s == "visual_projection")],
        "dn_l": np.flatnonzero((sc_s == "descending_neuron")
                               & (side_s == "left")),
        "dn_r": np.flatnonzero((sc_s == "descending_neuron")
                               & (side_s == "right")),
    }
    if cache:
        subnet_file.save(hops, rho_target, drop, (sub, ids[idx], sann, info))
    return net, sann, sub, info


def main(argv=None) -> int:
    import time

    t = time.perf_counter()
    net, ann, sub, info = build_eye_subnet()
    print("built %d neurons, %d edges in %.0f s"
          % (net.n, sub.nnz, time.perf_counter() - t))
    t = time.perf_counter()
    net2, ann2, sub2, info2 = build_eye_subnet()
    print("loaded in %.1f s" % (time.perf_counter() - t))
    assert net2.n == net.n and sub2.nnz == sub.nnz
    assert np.array_equal(info2["lam_rows"], info["lam_rows"])
    sc = ann["super_class"].to_numpy(dtype="<U32")
    import collections
    print("  composition:", dict(collections.Counter(sc).most_common(6)))
    print("  lamina injected %d, descending L%d/R%d"
          % (len(info["lam_rows"]), len(info["dn_l"]), len(info["dn_r"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
