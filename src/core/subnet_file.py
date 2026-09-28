"""The projection-neuron subnetwork lives in a file.

Same lesson as `environment.room_file`.  `build_vp_subnet` reads the whole
MaleCNS connectome -- 166,700 neurons and every edge -- builds a sparse
matrix, symmetrises it, propagates receptive directions, and then throws away
80% of it to keep 34,125 neurons and 1.08 M edges.  The result is about
13 MB and is a pure function of `(hops, rho_target, exclusions)`, but it was
being rebuilt on every single run.

Measured on 2026-09-21: 25 s with the OS file cache warm and 68 s with it
cold, against 41 s for the flight the run actually existed to perform.  The
variance alone is larger than the experiment.

Write it once:

    .venv/Scripts/python.exe -m core.subnet_file

and every run after that loads a file.
"""
from __future__ import annotations

import pickle
from pathlib import Path

CACHE = Path(__file__).resolve().parents[2] / "results" / "subnet"


def path_for(hops: int, rho_target: float, drop: tuple) -> Path:
    # The name states everything `idx` depends on.  It used to be `h%d`
    # alone, so changing the exclusion list silently reused a subnetwork
    # built under the old rule.
    return CACHE / ("vp_h%d_rho%.2f_%s.pkl"
                    % (hops, rho_target, "-".join(drop) or "none"))


def load(hops: int, rho_target: float, drop: tuple):
    p = path_for(hops, rho_target, drop)
    if not p.exists():
        return None
    with p.open("rb") as fh:
        return pickle.load(fh)


def save(hops: int, rho_target: float, drop: tuple, payload) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = path_for(hops, rho_target, drop)
    with p.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def main(argv=None) -> int:
    import time

    import numpy as np

    from core.vp_subnet import build_vp_subnet

    t = time.perf_counter()
    net, ann, sub, info = build_vp_subnet()
    print("built %d neurons, %d edges in %.0f s"
          % (net.n, sub.nnz, time.perf_counter() - t))

    t = time.perf_counter()
    net2, ann2, sub2, info2 = build_vp_subnet()
    print("loaded in %.1f s" % (time.perf_counter() - t))

    # the file must reproduce what was built, or it is not a cache
    assert net2.n == net.n and sub2.nnz == sub.nnz
    assert np.array_equal(net2.ids, net.ids)
    assert abs(sub2 - sub).nnz == 0
    for k in ("vp_rows", "dn_l", "dn_r"):
        assert np.array_equal(info2[k], info[k]), k
    for k in ("vp_az", "vp_el"):
        assert np.allclose(info2[k], info[k], equal_nan=True), k
    assert ann2["super_class"].equals(ann["super_class"])
    print("round-trip identical: neurons, edges, ids, weights, "
          "vp/dn indices, receptive directions, annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
