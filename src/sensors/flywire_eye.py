"""The fly's own view: a rendered scene sampled by the measured eye lattice.

Every Stage 0/1 sensor read the body's heading and handed it straight to the
core.  That is fine for a reduced steering model, but for a whole-brain model
it would mean WE choose the tuning of whatever cells we inject into -- and a
heading representation that we put in by hand is not evidence that the
connectome computes one.  So the input here is a SCENE, and the tuning is
whatever the circuit makes of it.

Where the numbers come from
---------------------------
Column assignment: Matsliah et al. 2024 (Nature), distributed at
    storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz
Every Mi1 was assigned to a hexagonal lattice point and the other columnar
cells matched to it one-to-one, giving (p, q) for 45,528 neurons over both
eyes.  Nothing about that lattice is ours.

Axis convention: Zhao et al. 2022 Fig. 2, as documented in OpticLobe.jl --
for the right eye, +p is anterodorsal and +q is posterodorsal.  Measured here
(see verify() below) the true angle between the p and q axes is 120 degrees,
i.e. a real hexagonal lattice; Zhao's 90-degree version is a display
approximation.  With 120-degree axes:

    dorsal    component = (p + q) / 2
    posterior component = (q - p) * sqrt(3) / 2

What IS ours, and has to be justified
-------------------------------------
(p, q) is a lattice INDEX, not a direction.  Turning it into an angle needs a
scale, and we set one interommatidial angle for the whole eye.  The real eye
has non-uniform interommatidial angles over a curved surface, so this is an
approximation of unquantified error.

It is not a free parameter, though: the measured lattice spans 29.5 units in
BOTH azimuth and elevation, so DEG_PER_COLUMN = 5.0 gives a ~148 deg monocular
field, against the ~150 deg that is reported for Drosophila.  The scale is
therefore checked against the field of view rather than assumed.

The ommatidial acceptance angle follows Lappalainen et al. 2024 (flyvis,
datasets/rendering/eye.py), which uses 5.8 degrees.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from utils.angles import wrap

DEG_PER_COLUMN = 5.0        # interommatidial angle; checked against the FOV
ACCEPTANCE_DEG = 5.8        # flyvis, Lappalainen et al. 2024
DEFAULT_CSV = "data/flywire/column_assignment.csv.gz"

# Both eyes are published in the SAME (p, q) range -- 772 of 796 lattice
# positions are shared -- so the coordinates are EYE-LOCAL, centred on each
# eye's own optical axis.  Placing them in the body frame therefore needs the
# angle between each eye's axis and the midline, which the assignment does not
# contain.
#
# It is not free either.  Each eye is 148 deg wide (above), and Drosophila is
# reported to have a total visual field near 270 deg with a narrow frontal
# binocular overlap.  Those two facts fix the offset:
#
#     2 * 148 - 270 = 26 deg of overlap,  centre = 135 - 148/2 = 61 deg
#
# giving a right eye spanning -13 deg to +135 deg, a mirrored left eye, and a
# 26 deg frontal binocular region -- inside the 20-40 deg usually quoted.
EYE_CENTRE_AZIMUTH_DEG = 61.0

# The published convention is written for the RIGHT eye.  The left eye is its
# mirror, so its azimuth flips sign; elevation does not.
_AZIMUTH_SIGN = {"right": 1.0, "left": -1.0}


@dataclass(frozen=True)
class EyeLattice:
    """Per-neuron viewing directions, in degrees, relative to the body axis."""

    root_id: np.ndarray        # FlyWire id
    cell_type: np.ndarray
    eye: np.ndarray            # 'left' / 'right'
    azimuth_deg: np.ndarray    # + = posterior on that side
    elevation_deg: np.ndarray  # + = dorsal

    def of_type(self, *types: str) -> np.ndarray:
        """Row mask for the given columnar cell types (e.g. the L1..L5 targets)."""
        return np.isin(self.cell_type, np.asarray(types))

    def as_dict(self) -> Dict:
        return {"n_neurons": int(len(self.root_id)),
                "n_types": int(len(np.unique(self.cell_type))),
                "deg_per_column": DEG_PER_COLUMN,
                "acceptance_deg": ACCEPTANCE_DEG,
                "source": "Matsliah et al. 2024 column_assignment.csv.gz",
                "axes": "Zhao et al. 2022 Fig.2, measured as 120 deg apart"}


def load(csv_path: str | Path = DEFAULT_CSV,
         deg_per_column: float = DEG_PER_COLUMN,
         matched_only: bool = False) -> EyeLattice:
    """Build per-neuron viewing directions from the published assignment.

    `matched_only` keeps only lattice positions reconstructed in BOTH eyes.
    Without it the two eyes are injected unequally -- the right eye has 5,117
    of the injected columnar cells against the left's 4,841, purely because
    more of it was reconstructed -- and that 6.5% input difference was
    measured to grow along the path (1.00 per neuron at the injection site,
    1.08 in the optic lobe, 1.10 centrally, 1.17 at the descending neurons).
    Matching the columns makes it exactly 4,558 per side.

    This is where to fix the asymmetry, not in the connectome's weights: a
    global per-side weight rebalancing overcorrected and merely flipped the
    output's sign, because the cause is neuron COUNT at the input.
    """
    df = pd.read_csv(csv_path)
    if matched_only:
        key = list(zip(df["type"], df["p"], df["q"]))
        left = {k for k, h in zip(key, df["hemisphere"]) if h == "left"}
        right = {k for k, h in zip(key, df["hemisphere"]) if h == "right"}
        both = left & right
        df = df[[k in both for k in key]].reset_index(drop=True)
    p = df["p"].to_numpy(float)
    q = df["q"].to_numpy(float)

    # true hexagonal geometry: the p and q axes are 120 degrees apart
    dorsal = (p + q) / 2.0
    posterior = (q - p) * (np.sqrt(3.0) / 2.0)

    sign = df["hemisphere"].map(_AZIMUTH_SIGN).to_numpy(float)
    if np.isnan(sign).any():
        raise ValueError("unexpected hemisphere label in the assignment file")

    # body frame: 0 = straight ahead, + = to the fly's right
    azimuth = sign * (EYE_CENTRE_AZIMUTH_DEG + posterior * deg_per_column)

    return EyeLattice(
        root_id=df["root_id"].to_numpy(np.int64),
        cell_type=df["type"].astype(str).to_numpy(),
        eye=df["hemisphere"].astype(str).to_numpy(),
        azimuth_deg=azimuth,
        elevation_deg=dorsal * deg_per_column,
    )


def bar_scene(bar_azimuth_deg: float, bar_width_deg: float = 15.0,
              contrast: float = 1.0, background: float = 0.0):
    """A single bright vertical bar on a dark cylinder.

    The standard heading cue in the ring-attractor literature: constant in
    elevation, so it carries azimuth only, which is all a heading task needs.
    Returns a callable mapping (azimuth, elevation) in degrees to luminance.
    """
    def luminance(azimuth_deg, elevation_deg):
        d = np.rad2deg(np.abs(wrap(np.deg2rad(
            np.asarray(azimuth_deg) - bar_azimuth_deg))))
        return np.where(d <= bar_width_deg / 2.0,
                        background + contrast, background)
    return luminance


def sample(lattice: EyeLattice, scene, heading_deg: float,
           acceptance_deg: float = ACCEPTANCE_DEG,
           n_sub: int = 5) -> np.ndarray:
    """Luminance seen by each neuron when the body points at `heading_deg`.

    Each ommatidium integrates over its acceptance angle rather than sampling a
    point, so a bar narrower than the acceptance angle still registers.  The
    integration is a plain box average over `n_sub` offsets -- a Gaussian
    acceptance function would be more faithful and is the obvious upgrade.
    ponytail: box acceptance, swap for a Gaussian if the bar width matters.
    """
    offs = np.linspace(-0.5, 0.5, n_sub) * acceptance_deg
    # the scene is world-fixed, so the body's heading shifts what each
    # ommatidium looks at
    az = lattice.azimuth_deg + heading_deg
    total = np.zeros(len(az))
    for da in offs:
        total += scene(az + da, lattice.elevation_deg)
    return total / n_sub


def verify(csv_path: str | Path = DEFAULT_CSV) -> None:
    """Runnable check: the lattice really is hexagonal and the scale is sane."""
    from scipy.spatial import cKDTree

    df = pd.read_csv(csv_path)
    m = df[(df["type"] == "Mi1") & (df["hemisphere"] == "right")]
    p, q = m["p"].to_numpy(float), m["q"].to_numpy(float)

    # 120-degree axes must give a clean hexagonal lattice; 90 must not
    for deg, expect_hex in ((120.0, True), (90.0, False)):
        a = np.deg2rad(deg)
        xy = np.c_[p + q * np.cos(a), q * np.sin(a)]
        d, _ = cKDTree(xy).query(xy, k=7)
        nn = np.median(d[:, 1])
        cv = d[:, 1].std() / d[:, 1].mean()
        deg6 = np.mean([len(i) - 1 for i in
                        cKDTree(xy).query_ball_point(xy, 1.5 * nn)])
        if expect_hex:
            assert cv < 0.01, (deg, cv)
            assert 5.0 < deg6 < 6.5, (deg, deg6)
        else:
            assert deg6 > 7.0, (deg, deg6)

    lat = load(csv_path)
    right = lat.eye == "right"
    left = ~right

    # monocular field of view, against the ~150 deg reported for Drosophila
    for name, sel in (("right", right), ("left", left)):
        fov_az = np.ptp(lat.azimuth_deg[sel])
        fov_el = np.ptp(lat.elevation_deg[sel])
        assert 130.0 < fov_az < 165.0, (name, fov_az)
        assert 130.0 < fov_el < 165.0, (name, fov_el)

    # the two eyes must be mirror images in azimuth and aligned in elevation
    assert abs(lat.azimuth_deg[right].mean()
               + lat.azimuth_deg[left].mean()) < 5.0
    assert abs(lat.elevation_deg[right].mean()
               - lat.elevation_deg[left].mean()) < 1.0

    # each eye points outward, not forward, and they meet in a narrow frontal
    # binocular region rather than overlapping everywhere
    assert lat.azimuth_deg[right].mean() > 50.0
    assert lat.azimuth_deg[left].mean() < -50.0
    binocular = (np.ptp(lat.azimuth_deg[right]) * 2.0
                 - (lat.azimuth_deg[right].max()
                    - lat.azimuth_deg[left].min()))
    assert 15.0 < binocular < 45.0, binocular

    # a bar straight ahead falls in the binocular region, so BOTH eyes see it
    scene = bar_scene(0.0, bar_width_deg=20.0)
    front = sample(lat, scene, heading_deg=0.0)
    assert front[right].sum() > 0 and front[left].sum() > 0

    # turning the body to put the bar on one side makes that eye dominate
    side = sample(lat, bar_scene(0.0, 20.0), heading_deg=90.0)
    assert not np.allclose(front, side)
    assert side[left].sum() > 3.0 * side[right].sum(), (
        side[left].sum(), side[right].sum())

    print("flywire_eye: %d neurons, %d types" % (len(lat.root_id),
                                                 len(np.unique(lat.cell_type))))
    print("  right eye FOV: %.0f deg azimuth x %.0f deg elevation"
          % (np.ptp(lat.azimuth_deg[right]), np.ptp(lat.elevation_deg[right])))
    print("  L1-L5 with directions: %d" % lat.of_type("L1", "L2", "L3", "L4",
                                                      "L5").sum())
    print("  R7/R8 with directions: %d" % lat.of_type("R7", "R8").sum())
    print("  bar ahead lights %d neurons (L %d / R %d); after a 90 deg turn, "
          "%d (L %d / R %d)"
          % (int((front > 0).sum()), int((front[left] > 0).sum()),
             int((front[right] > 0).sum()), int((side > 0).sum()),
             int((side[left] > 0).sum()), int((side[right] > 0).sum())))
    print("all checks passed")


if __name__ == "__main__":
    verify()


# --- MaleCNS -----------------------------------------------------------
# The same construction on a different dataset.  MaleCNS carries the lattice
# in the annotation itself as assignedOlHex1 / assignedOlHex2, where FAFB
# needed Matsliah et al.'s separate column-assignment file.  Measured here the
# axes are again 120 degrees apart (nearest-neighbour CV 0.000-0.048 with 5.7
# neighbours per column, against 7.6 at 90 degrees), so the same dorsal and
# posterior components apply.  The indices are 1-based rather than centred,
# so they are centred here.
#
# Injection targets differ: L1, L2, L3 and L5 carry hex, while L4, R7 and R8
# do not.
MALECNS_INJECT_TYPES = ("L1", "L2", "L3", "L5")


def load_malecns_eye(ann, deg_per_column: float = DEG_PER_COLUMN,
                     matched_only: bool = False) -> EyeLattice:
    """Viewing directions from MaleCNS's built-in hex assignment."""
    h = ann[ann["hex1"].notna() & ann["hex2"].notna()].copy()
    h = h[h["side"].isin(["left", "right"])]
    if matched_only:
        key = list(zip(h["cell_type"], h["hex1"], h["hex2"]))
        left = {k for k, s in zip(key, h["side"]) if s == "left"}
        right = {k for k, s in zip(key, h["side"]) if s == "right"}
        both = left & right
        h = h[[k in both for k in key]]

    p = h["hex1"].to_numpy(float)
    q = h["hex2"].to_numpy(float)
    # 1-based indices; centre them so the eye's axis sits at 0
    p = p - (p.min() + p.max()) / 2.0
    q = q - (q.min() + q.max()) / 2.0

    dorsal = (p + q) / 2.0
    posterior = (q - p) * (np.sqrt(3.0) / 2.0)
    sign = h["side"].map(_AZIMUTH_SIGN).to_numpy(float)

    return EyeLattice(
        root_id=h.index.to_numpy(np.int64),
        cell_type=h["cell_type"].to_numpy(dtype="<U48"),
        eye=h["side"].to_numpy(dtype="<U8"),
        azimuth_deg=sign * (EYE_CENTRE_AZIMUTH_DEG
                            + posterior * deg_per_column),
        elevation_deg=dorsal * deg_per_column,
    )


def verify_malecns() -> None:
    """Same field-of-view check that validated the FAFB scale."""
    import sys
    sys.path.insert(0, "src")
    from core.malecns import _load_annotations

    ann = _load_annotations().set_index("root_id")
    lat = load_malecns_eye(ann)
    right = lat.eye == "right"
    left = ~right
    print("malecns eye: %d neurons, %d types"
          % (len(lat.root_id), len(np.unique(lat.cell_type))))
    for name, sel in (("right", right), ("left", left)):
        fov_az = np.ptp(lat.azimuth_deg[sel])
        fov_el = np.ptp(lat.elevation_deg[sel])
        print("  %-5s eye FOV: %.0f deg azimuth x %.0f deg elevation"
              % (name, fov_az, fov_el))
        assert 120.0 < fov_az < 190.0, (name, fov_az)
        assert 120.0 < fov_el < 190.0, (name, fov_el)

    assert lat.azimuth_deg[right].mean() > 40.0
    assert lat.azimuth_deg[left].mean() < -40.0

    scene = bar_scene(0.0, bar_width_deg=20.0)
    front = sample(lat, scene, heading_deg=0.0)
    side_on = sample(lat, scene, heading_deg=90.0)
    assert front[right].sum() > 0 and front[left].sum() > 0
    assert side_on[left].sum() > 3.0 * side_on[right].sum()
    print("  bar ahead lights %d (L %d / R %d); after a 90 deg turn %d (L %d / R %d)"
          % ((front > 0).sum(), (front[left] > 0).sum(),
             (front[right] > 0).sum(), (side_on > 0).sum(),
             (side_on[left] > 0).sum(), (side_on[right] > 0).sum()))
    n_inj = lat.of_type(*MALECNS_INJECT_TYPES).sum()
    print("  injection targets (%s): %d"
          % ("/".join(MALECNS_INJECT_TYPES), n_inj))
    assert n_inj > 5000
    print("malecns eye ok")
