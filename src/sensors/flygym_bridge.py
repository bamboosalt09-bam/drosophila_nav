"""Join flygym's 721 ommatidia to the connectome's ~875 hex columns.

Both are hexagonal lattices with a known angular extent, so the join needs no
fisheye inversion: normalise each lattice to the same field of view and take
the nearest viewing direction.

    flygym   721 ommatidia per eye, 157 deg fovy, centroids on a clean hex
             lattice (nearest-neighbour spacing CV 0.001, 5.7 neighbours)
    ours     875-887 columns per eye from MaleCNS assignedOlHex1/2, field of
             view 147-162 deg

Our lattice is the denser one, so several columns share an ommatidium.  That
is the honest direction for the mismatch: the simulator renders fewer samples
than the connectome has columns, and duplicating a reading is better than
inventing one.

ponytail: nearest-neighbour join on a flat projection, which ignores that both
"images" are really on a sphere.  At these fields of view the error is small
next to the 5 deg interommatidial angle; use a proper angular metric if the
periphery ever matters.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def ommatidia_directions(retina) -> np.ndarray:
    """(n_ommatidia, 2) azimuth and elevation in degrees, eye-local.

    The centroid of each ommatidium's pixels in the id map, rescaled so the
    lattice spans the camera's field of view.  Eye-local means centred on the
    eye's own axis, the same convention the connectome lattice uses before its
    optical-axis offset is applied.
    """
    m = retina.ommatidia_id_map
    n = retina.num_ommatidia_per_eye
    ys, xs = np.nonzero(m > 0)
    ids = m[ys, xs] - 1
    counts = np.bincount(ids, minlength=n).astype(float)
    cx = np.bincount(ids, xs, minlength=n) / counts
    cy = np.bincount(ids, ys, minlength=n) / counts

    # centre, then scale the wider axis to the stated field of view
    fov = float(getattr(retina, "fovy", 157.0))
    cx -= (cx.min() + cx.max()) / 2.0
    cy -= (cy.min() + cy.max()) / 2.0
    scale = fov / max(np.ptp(cx), np.ptp(cy))
    # image x grows toward one side and y downward; elevation is up-positive
    return np.c_[cx * scale, -cy * scale]


def join(lattice, retina) -> Tuple[np.ndarray, np.ndarray]:
    """For each connectome column, which ommatidium of its own eye it reads.

    Returns (ommatidium index per column, angular error in degrees).  The
    error is worth keeping: it says how much resolution the join throws away,
    and a large value would mean the two lattices do not actually overlap.
    """
    from scipy.spatial import cKDTree

    omm = ommatidia_directions(retina)
    tree = cKDTree(omm)

    idx = np.full(len(lattice.root_id), -1, dtype=np.int64)
    err = np.full(len(lattice.root_id), np.nan)
    for eye, sign in (("left", -1.0), ("right", 1.0)):
        sel = lattice.eye == eye
        if not sel.any():
            continue
        # undo the optical-axis offset to get back to eye-local angles
        az = sign * lattice.azimuth_deg[sel]
        from sensors.flywire_eye import EYE_CENTRE_AZIMUTH_DEG
        local = np.c_[az - EYE_CENTRE_AZIMUTH_DEG, lattice.elevation_deg[sel]]
        d, j = tree.query(local)
        idx[sel] = j
        err[sel] = d
    return idx, err


def demo() -> None:
    """Check the join covers both lattices without large angular error."""
    import sys

    sys.path.insert(0, "src")
    from flygym.vision.retina import Retina

    import sensors.flywire_eye as eye
    from core.malecns import _load_annotations

    ann = _load_annotations().set_index("root_id")
    lat = eye.load_malecns_eye(ann)
    retina = Retina()

    omm = ommatidia_directions(retina)
    print("flygym: %d ommatidia, azimuth %.0f..%.0f, elevation %.0f..%.0f deg"
          % (len(omm), omm[:, 0].min(), omm[:, 0].max(),
             omm[:, 1].min(), omm[:, 1].max()))

    idx, err = join(lat, retina)
    assert (idx >= 0).all(), "every column must find an ommatidium"
    print("connectome: %d columns joined, angular error median %.2f deg, "
          "95th %.2f, max %.2f"
          % (len(idx), np.median(err), np.percentile(err, 95), err.max()))
    # the join must be tight next to the 5 deg interommatidial angle
    assert np.median(err) < 5.0, np.median(err)

    used = len(np.unique(idx))
    print("  ommatidia actually read: %d of %d (%.0f%%); columns per "
          "ommatidium median %.1f"
          % (used, len(omm), 100 * used / len(omm),
             np.median(np.bincount(idx))))
    assert used > 0.5 * len(omm), "half the eye is going unread"
    print("demo ok")


if __name__ == "__main__":
    demo()
