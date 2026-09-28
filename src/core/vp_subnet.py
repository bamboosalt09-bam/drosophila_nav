"""Inject at the visual projection neurons, and cut the optic lobe away.

Injecting at the lamina means the signal crosses 44,654 optic-lobe cells
before it reaches the brain, and this rate model has no reason to reproduce
what those cells compute -- motion detection, ON/OFF splitting, and the rest
of it are dynamics, not just wiring.  Measured consequence: a wall closing
from 12 m to 2 m on the LEFT made the steering readout go MORE positive,
i.e. turn further into it.  The proximity information was intact and growing
50-fold; only its sign was wrong.

The projection neurons are where the optic lobe hands its result to the
brain, so that is where a camera can be attached instead.  They carry no hex
coordinates, but a receptive field is just where the input comes from, so it
is read off the wiring:

    direction(j) = sum_i w(i->j) direction(i) / sum_i w(i->j)

over lamina cells i, accumulated across hops.  Measured: 81% of projection
neurons have a direction after 2 hops and 100% after 3, spanning -137..+135
deg, and left-side cells average +55 deg while right-side average -58 -- the
anatomy comes out of the wiring without being put in.

Cutting the optic lobe is then not just an optimisation.  Those cells would
receive no input and contribute nothing, so keeping them would be paying for
44,654 rows of zeros every step.

WHAT THIS GIVES UP, and it must be stated with any result that uses it: the
model no longer computes vision.  It computes what the brain does with
already-extracted visual features.  The claim shrinks from "the connectome
navigates from its eyes" to "the connectome navigates from visual projection
neuron activity", and the optic lobe becomes an assumption rather than a
simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def receptive_directions(out_csr, ids, ann, hops: int = 3):
    """Viewing direction per neuron, propagated from the lamina by wiring.

    Returns (azimuth_deg, elevation_deg), NaN where nothing reaches it.
    Azimuth follows the project convention: + is the fly's LEFT.
    """
    import sensors.flywire_eye as eye

    lat = eye.load_malecns_eye(ann)
    sel = lat.of_type(*eye.MALECNS_INJECT_TYPES)
    rows = pd.Index(ids).get_indexer(lat.root_id[sel])
    ok = rows >= 0
    lam = rows[ok]

    n = out_csr.shape[0]
    absW = out_csr.tocsr().copy()
    absW.data = np.abs(absW.data)

    w = np.zeros(n); w[lam] = 1.0
    a = np.zeros(n); a[lam] = -lat.azimuth_deg[sel][ok]
    e = np.zeros(n); e[lam] = lat.elevation_deg[sel][ok]
    cw = np.zeros(n); ca = np.zeros(n); ce = np.zeros(n)
    for _ in range(hops):
        w, a, e = absW.T @ w, absW.T @ a, absW.T @ e
        cw += w; ca += a; ce += e
    good = cw > 0
    az = np.where(good, ca / np.maximum(cw, 1e-12), np.nan)
    el = np.where(good, ce / np.maximum(cw, 1e-12), np.nan)
    return az, el, lam


def _reach(adj, seed, hops):
    seen = np.zeros(adj.shape[0], dtype=bool)
    seen[seed] = True
    frontier = seen.copy()
    for _ in range(hops):
        nxt = (adj.T @ frontier.astype(np.float32)) > 0
        frontier = nxt & ~seen
        if not frontier.any():
            break
        seen |= frontier
    return seen


def build_vp_subnet(rho_target: float = 0.50, hops: int = 4,
                    cache: bool = True):
    """Projection neurons in, descending neurons out, optic lobe removed."""
    import time
    from pathlib import Path

    import scipy.sparse.linalg as spl
    from core.flywire_rate import FlyWireRate
    from core.malecns import load_malecns

    repo = Path(__file__).resolve().parents[2]
    drop = ("ol_intrinsic",)        # super_classes cut; part of the cache key
    # The cache key must name everything `idx` depends on, not just `hops`.
    # It was `vp_subnet_h4.npz`, so changing the exclusion list -- which is
    # how the optic lobe and the lamina come out -- silently reused a
    # subnetwork built under the old rule.
    key = "h%d_%s" % (hops, "-".join(drop))
    cf = repo / ("results/vp_subnet_%s.npz" % key)

    # A FILE FIRST.  Everything below reads the whole 166,700-neuron
    # connectome to keep 34,125 of it, and the answer never changes: 25 s
    # with the OS file cache warm, 68 s cold, against 41 s for the flight the
    # run exists to perform.  See core.subnet_file.
    from core import subnet_file
    if cache:
        hit = subnet_file.load(hops, rho_target, drop)
        if hit is not None:
            sub, sids, sann, info = hit
            return (FlyWireRate(out_csr=sub, ids=sids, ann=sann),
                    sann, sub, info)

    ids, out, ann, _ = load_malecns(w_scale=1.0, symmetrise=True)
    sc = ann["super_class"].to_numpy(dtype="<U32")
    az, el, lam = receptive_directions(out, ids, ann)

    vp = np.flatnonzero((sc == "visual_projection") & ~np.isnan(az))
    dn = np.flatnonzero(sc == "descending_neuron")

    if cache and cf.exists():
        z = np.load(cf)
        idx, rho = z["idx"], float(z["rho"])
    else:
        t = time.perf_counter()
        keep = _reach(out, vp, hops) & _reach(out.T.tocsr(), dn, hops)
        keep[vp] = True
        keep[dn] = True
        # the optic lobe is what we are replacing; without input it would be
        # 44,654 rows of zeros
        keep &= ~np.isin(sc, list(drop))
        keep[lam] = False
        idx = np.flatnonzero(keep)
        sub0 = out[idx][:, idx].tocsr()
        rho = float(np.abs(spl.eigs(sub0.astype(np.float64), k=1,
                                    return_eigenvectors=False)[0]))
        np.savez(cf, idx=idx, rho=rho)
        print("  cut+eigs %.0f s" % (time.perf_counter() - t), flush=True)

    sub = (out[idx][:, idx].tocsr() * (rho_target / rho)).tocsr()
    sann = ann.iloc[idx].copy()
    net = FlyWireRate(out_csr=sub, ids=ids[idx], ann=sann)

    sc_s = sann["super_class"].to_numpy(dtype="<U32")
    side_s = sann["side"].to_numpy(dtype="<U16")
    vp_rows = np.flatnonzero(sc_s == "visual_projection")
    info = {
        "rho_raw": rho, "n_edges": int(sub.nnz),
        "vp_rows": vp_rows,
        "vp_az": az[idx][vp_rows], "vp_el": el[idx][vp_rows],
        "dn_l": np.flatnonzero((sc_s == "descending_neuron") & (side_s == "left")),
        "dn_r": np.flatnonzero((sc_s == "descending_neuron") & (side_s == "right")),
    }
    if cache:
        subnet_file.save(hops, rho_target, drop, (sub, ids[idx], sann, info))
    return net, sann, sub, info


def demo() -> None:
    """Check the cut keeps what it must and drops what it should."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import time

    t0 = time.perf_counter()
    net, ann, sub, info = build_vp_subnet()
    sc = ann["super_class"].to_numpy(dtype="<U32")
    print("%d neurons, %d edges, rho %.0f -> 0.50  (%.0f s)"
          % (net.n, sub.nnz, info["rho_raw"], time.perf_counter() - t0))
    u, c = np.unique(sc, return_counts=True)
    print("  composition: %s" % dict(zip(u.tolist(), c.tolist())))
    print("  injection sites (visual_projection): %d, azimuth %+.0f..%+.0f"
          % (len(info["vp_rows"]), np.nanmin(info["vp_az"]),
             np.nanmax(info["vp_az"])))
    print("  readout (descending): L%d / R%d"
          % (len(info["dn_l"]), len(info["dn_r"])))
    assert "ol_intrinsic" not in u, "optic lobe should be gone"
    assert len(info["vp_rows"]) > 3000
    assert len(info["dn_l"]) > 100 and len(info["dn_r"]) > 100
    # left-side injection sites must look left
    L = info["vp_az"][:len(info["vp_az"])]
    print("  mean azimuth of injection sites: %+.1f deg" % np.nanmean(L))
    print("demo ok")


if __name__ == "__main__":
    demo()


def frontal_dn(sub, info, half_width_deg: float = 30.0, hops: int = 4):
    """Descending neurons that look FORWARD, found by wiring.

    The projection neurons carry a receptive azimuth recovered from the
    lamina; propagating it forward through the subnetwork by the same
    weighted mean gives every descending neuron one too.  Measured: 100% of
    the 1,304 descending neurons get a direction, spanning -106..+80 deg
    with a median |azimuth| of 19 deg.

    The braking channel wants the forward ones.  Steering does not -- a
    turn away from a wall needs the periphery, which is why the eye is
    290 deg wide in the first place.
    """
    import numpy as np
    A = sub.tocsr().copy()
    A.data = np.abs(A.data)
    n = sub.shape[0]
    w = np.zeros(n)
    a = np.zeros(n)
    w[info["vp_rows"]] = 1.0
    a[info["vp_rows"]] = np.nan_to_num(info["vp_az"])
    for _ in range(hops):
        w = A.T @ w + w
        a = A.T @ a + a
    az = np.where(w > 1e-12, a / np.maximum(w, 1e-12), np.nan)
    m = np.isfinite(az) & (np.abs(az) <= half_width_deg)
    return info["dn_l"][m[info["dn_l"]]], info["dn_r"][m[info["dn_r"]]]
