"""Three cameras into the connectome's two sensory addresses.

    camera L, R  ->  both eyes      lamina L1/L2/L3/L5, ~6,199 cells
    camera 3     ->  pheromone      ORN_DA1 / VA1d / VA1v / DL3, 569 cells

Two separate streams, as the design says.  The eyes get the SCENE -- obstacles
as silhouettes, which is what makes them visible to the circuit at all.  The
third camera gets only the target's centroid, reduced to a bearing, and
delivers it where an attractive odour would arrive.

The pheromone address needs no functional designation: the glomerulus name IS
the address, exactly as the lamina is for vision.  A bearing becomes a
left-right imbalance across the bilateral ORN populations, which is how a fly's
paired antennae carry direction in the first place.

Everything upstream already existed -- `load_malecns_eye` for the lattice,
`eye.sample` for scene-to-neuron with the heading shift, `AttractionCueSensor`
for the centroid.  This file is only the wiring.

ponytail: ORN drive is a linear left/right split of the bearing.  A real
bilateral comparison is concentration-based and saturating; swap in a
saturating law if the cue's dynamic range ever matters more than its sign.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

import sensors.flywire_eye as eye
from environment.target_world import TargetWorld, world_scene
from sensors.attraction_cue import AttractionCueSensor
from sensors.fisheye import (BlobTracker, FisheyeCamera,
                             contrast_centroid)

# Stage-2 working point, established by the heading probe: signal crosses the
# optic lobe without pinning the injected cells.
DRIVE_LIT = 0.20
# Which olfactory channel the attraction cue is delivered to.  Chosen by
# measurement, not by literature valence: all 53 ORN glomeruli were swept and
# only 5 give a steering signal that increases with target bearing (approach).
# VM2 (Or43b) is the cleanest by a wide margin -- offset/range 0.017, i.e. the
# signal sits almost exactly on zero and swings symmetrically either way --
# and it is an attractant channel in the literature too, so the choice is not
# picked to flatter the result.  The other canonical attractants (DM1, VA2,
# DM2, DM4) all steer the other way in this model; that is recorded, not
# hidden.  Full sweep: results/glomerulus_polarity.csv
PHEROMONE_TYPES = ("ORN_VM2",)


@dataclass
class InputTrace:
    """What each stream actually delivered, for checking the loop is alive."""

    eye_mean: float
    eye_dark_fraction: float
    orn_left: float
    orn_right: float
    cue_valid: bool


class ConnectomeInput:
    """Build the connectome's drive vector from the world and a pose."""

    def __init__(self, net, ann, orn_gain: float = 0.20,
                 drive_gain: float = DRIVE_LIT,
                 cue_sensor: Optional[AttractionCueSensor] = None,
                 cue_types=PHEROMONE_TYPES):
        # `cue_types` is the ADDRESS the attraction cue is delivered to.  It
        # is an argument rather than the module constant because a subnetwork
        # cut for the central complex may not contain the ORNs at all, and
        # then the cue has to arrive at a cell type that survived the cut.
        # Default unchanged, so every existing caller is unaffected.
        self.net, self.ann = net, ann
        self.drive_gain = drive_gain
        self.orn_gain = orn_gain
        self.cue_sensor = cue_sensor or AttractionCueSensor()
        # When set, the cue's magnitude is pinned and only its DIRECTION
        # varies.  Apparent size changes 5x between 20 m and 4 m (measured
        # 0.038 -> 0.190), so this separates "the network cannot cope with a
        # changing input magnitude" from "the network cannot steer".
        self.fixed_strength = None
        # The goal direction, in WORLD coordinates, held across steps.
        #
        # An attraction cue is not a visual reflex and must not behave like
        # one.  Until now it did: the cue was injected only while the target
        # was in frame and dropped to zero the instant it went behind a wall,
        # which made it the same KIND of signal as vision -- instantaneous,
        # body-relative, gone when the input is gone -- differing only in
        # amplitude.  Raising its gain then just starts an amplitude contest
        # it cannot win against 6,199 lamina cells.
        #
        # A pheromone-like goal signal is a different kind of thing: it is
        # allocentric and it persists.  The fly holds the direction it was
        # last heading for, re-expresses it in body coordinates as it turns,
        # and keeps flying that way while the source is out of contact.
        # Vision then perturbs that heading moment to moment rather than
        # replacing it.  That IS the hierarchy -- goal underneath, reflex on
        # top -- and it is what the central complex maintains.
        #
        # `FlyPolicy._last_goal` and master doc section 22 already specified
        # this; the connectome arm simply never had it.
        self._goal_world = None

        # --- eyes ---------------------------------------------------------
        lat = eye.load_malecns_eye(ann)
        idx = pd.Index(net.ids).get_indexer(lat.root_id)
        inject = lat.of_type(*eye.MALECNS_INJECT_TYPES) & (idx >= 0)
        self.lat = lat
        self.eye_sel = inject
        self.eye_rows = idx[inject]
        # `ann` must be indexed by root_id: the eye lattice is keyed off that
        # index, and a reindexed frame silently matches a handful of unrelated
        # neurons instead of failing.  It did -- 1,779 of 6,199 -- and every
        # viewing direction then sat on the wrong cell.
        n_typed = int(lat.of_type(*eye.MALECNS_INJECT_TYPES).sum())
        if n_typed and inject.sum() < 0.9 * n_typed:
            raise ValueError(
                "only %d of %d lamina cells matched the network; `ann` is "
                "probably not indexed by root_id"
                % (int(inject.sum()), n_typed))
        # CONVENTION CLASH, and it mirrored the world.  `flywire_eye` documents
        # its lattice as "0 = straight ahead, + = to the fly's RIGHT" (right
        # eye mean +59 deg, left -59).  Everything on this side of the project
        # -- heading, yaw rate, the attraction cue's bearing -- is CCW-positive,
        # i.e. + = LEFT.  Feeding a CCW scene to a right-positive lattice put
        # every object on the wrong side of the fly: 48 of 53 glomeruli then
        # steered away from the target, which is a systematic sign error, not
        # 53 independent circuit facts.  Convert once, here, rather than
        # changing a convention other code and its tests already rely on.
        self.eye_az = -lat.azimuth_deg[inject]
        self.eye_el = lat.elevation_deg[inject]
        # acceptance-angle offsets, precomputed.  `eye.sample` would do this
        # over the whole 23,720-column lattice; only 6,199 are injected, so
        # sampling the rest is 74% wasted work (measured: 6.97 ms -> 1.6 ms).
        self._offs = np.linspace(-0.5, 0.5, 5) * eye.ACCEPTANCE_DEG
        # L1, L2, L3 and L5 share an ommatidium, so 6,199 injected cells look
        # along only 1,769 distinct directions: every ray was being cast 3.5
        # times.  Cast the unique set and scatter the result back.
        # ONE fisheye image feeds both streams.  Before this the lamina cast
        # 6,199 x 5 rays and the cue projected 69 surface samples per target
        # separately -- 49.6 ms and 34.2 ms in a 161-obstacle room, with the
        # target rendered twice.  The image is 9.7 ms and the cue reads out of
        # it, so six beacons cost what one costs and the tracker is never told
        # where anything is.
        self.cam = FisheyeCamera()
        self._px = self.cam.bind(self.eye_az, self.eye_el)
        self.tracker = BlobTracker()
        key = np.stack([np.round(self.eye_az, 4),
                        np.round(self.eye_el, 4)], axis=1)
        _, uidx, self._uinv = np.unique(key, axis=0, return_index=True,
                                        return_inverse=True)
        self._uaz = self.eye_az[uidx]
        self._uel = self.eye_el[uidx]
        # one reusable drive buffer, so a 166,700-element tensor is not
        # allocated 100 times per episode
        self._buf = torch.zeros(self.net.n, 1)

        # --- pheromone ORNs, by side -------------------------------------
        ctype = ann["cell_type"].to_numpy(dtype="<U48")
        side = ann["side"].to_numpy(dtype="<U16")
        is_orn = np.isin(ctype, tuple(cue_types))
        self.orn_l = np.flatnonzero(is_orn & (side == "left"))
        self.orn_r = np.flatnonzero(is_orn & (side == "right"))
        # The annotation gives 106 left against 285 right (plus 178 with no
        # side at all).  A real fly's antennae are matched, so that is
        # reconstruction completeness, not biology.  Equal per-neuron drive
        # would hand the right side 2.7x the total input and manufacture a
        # standing turn bias -- the exact artefact the wiring balance exists
        # to remove.  So drive is normalised per side: equal bearing in, equal
        # total drive out.
        self.orn_scale_l = 1.0 / max(len(self.orn_l), 1)
        self.orn_scale_r = 1.0 / max(len(self.orn_r), 1)

    def reset(self) -> None:
        """Forget the goal and the lock.  Episodes must not leak."""
        self._goal_world = None
        self.tracker.reset()

    def counts(self) -> dict:
        return {"lamina": int(self.eye_sel.sum()),
                "orn_left": len(self.orn_l), "orn_right": len(self.orn_r)}

    def _lamina_luminance(self, world, position, heading_rad,
                          contrast: bool = False, img=None) -> np.ndarray:
        """Luminance per injected lamina cell, read out of the fisheye image.

        `contrast` subtracts the mean, which is what the drive path wants and
        what L1/L2 large monopolar cells actually do; the raw form is kept
        for diagnostics that want absolute luminance.  `img` lets a caller
        that already rendered the frame avoid rendering it twice.
        """
        if img is None:
            img = self.cam.render(world, position, heading_rad)
        lum = img.ravel()[self._px]
        return lum - lum.mean() if contrast else lum


    def observation(self, world: TargetWorld, position: np.ndarray,
                    heading_rad: float, n_bins: int = 32) -> tuple:
        """The same sensory information, pooled for a network that has no eye.

        Identical scene, identical lamina viewing directions, identical cue --
        only the encoding differs, because a GRU cannot be handed 6,199
        columns and still be the small interpretable baseline the design asks
        for.  Pooling to 32 azimuth bins is near the fly's own resolution
        anyway: 360 deg over a 5.8 deg acceptance angle is ~62 independent
        samples, and half of that is still above what the task needs.

        ponytail: mean-pool by azimuth, elevation discarded.  Obstacles here
        are vertical, so elevation carries little; restore it the day the
        scene has structure above and below the horizon.
        """
        lum = self._lamina_luminance(world, position, heading_rad)
        # eye-local azimuth, wrapped to [-180, 180)
        rel = (self.eye_az + 180.0) % 360.0 - 180.0
        b = np.clip(((rel + 180.0) / 360.0 * n_bins).astype(int), 0, n_bins - 1)
        tot = np.bincount(b, lum, minlength=n_bins)
        cnt = np.bincount(b, minlength=n_bins)
        profile = np.where(cnt > 0, tot / np.maximum(cnt, 1), 1.0)
        cue = self.cue_sensor.sense(position, heading_rad, world.target,
                                    world.obstacles)
        return profile, cue

    def drive(self, world: TargetWorld, position: np.ndarray,
              heading_rad: float) -> tuple:
        """(drive vector over all neurons, InputTrace).

        ONE render feeds both streams: the lamina indexes into the image and
        the cue is found in it as a bright blob.  Neither is handed a target
        position, and the cost no longer grows with the number of targets.
        """
        img = self.cam.render(world, position, heading_rad)
        # contrast, not absolute luminance: a uniform scene carries no
        # direction, and L1/L2 high-pass for exactly that reason
        lum = self._lamina_luminance(world, position, heading_rad,
                                     contrast=True, img=img)

        # A REUSED buffer is only safe while the caller consumes it before the
        # next call.  `act_batch` does not, so allocate per call -- measured at
        # 0.01 ms, far below anything else here.
        d = torch.zeros(self.net.n, 1)
        d[self.eye_rows, 0] = torch.from_numpy(
            (lum * self.drive_gain).astype(np.float32))

        # pheromone: the whole-image contrast centroid, as a left-right
        # imbalance.  NOT a detection: thresholding for bright blobs first
        # discards every wall at the first step, and then nothing downstream
        # can avoid one.  Weighting every pixel by (I - sky) makes a wall a
        # negative weight, open sky neutral and a beacon strongly positive,
        # so approach and avoidance come out of the same weighted mean.
        bearing, blocked, weight = contrast_centroid(self.cam, img)
        self._goal_world = heading_rad + np.radians(bearing)
        rel = bearing
        frac = np.clip(rel / (self.cam.az_span / 2), -1.0, 1.0)
        s_ = weight if self.fixed_strength is None else self.fixed_strength
        base = self.orn_gain * s_
        l, r = base * (1 + frac), base * (1 - frac)
        d[self.orn_l, 0] = float(l) * self.orn_scale_l
        d[self.orn_r, 0] = float(r) * self.orn_scale_r

        return d, InputTrace(eye_mean=float(lum.mean()),
                             eye_dark_fraction=float(blocked),
                             orn_left=float(l), orn_right=float(r),
                             cue_valid=bool(weight > 1e-6))


def demo() -> None:
    """Check both streams reach the network, and carry what they should."""
    import sys
    import time
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "src"))

    from core.flywire_rate import FlyWireRate
    from core.malecns import load_malecns
    from environment.target_world import Obstacle

    t0 = time.perf_counter()
    ids, out, ann, _ = load_malecns(w_scale=0.005, symmetrise=True)
    net = FlyWireRate(out_csr=out, ids=ids, ann=ann)
    inp = ConnectomeInput(net, ann)
    print("loaded in %.0f s; %s" % (time.perf_counter() - t0, inp.counts()))
    assert inp.counts()["lamina"] > 1000
    assert inp.counts()["orn_left"] > 0 and inp.counts()["orn_right"] > 0

    world = TargetWorld(target=np.array([15.0, 0.0, 2.0]),
                        obstacles=[Obstacle(x=6.0, y=0.0, radius=1.0,
                                            z=0.0, height=6.0)])
    p = np.array([0.0, 0.0, 2.0])

    t0 = time.perf_counter()
    d, tr = inp.drive(world, p, 0.0)
    dt = time.perf_counter() - t0
    print("drive built in %.1f ms; eye mean %.3f, dark %.3f, ORN L %.4f R %.4f"
          % (1e3 * dt, tr.eye_mean, tr.eye_dark_fraction, tr.orn_left,
             tr.orn_right))
    assert float(d.abs().sum()) > 0, "nothing reached the network"
    # the pillar dead ahead must darken some of the eye
    assert tr.eye_dark_fraction > 0, "a pillar in front should be visible"

    # walking into the pillar must darken MORE of the eye -- looming, but now
    # measured at the neurons rather than at the scene function
    _, near = inp.drive(world, np.array([4.0, 0.0, 2.0]), 0.0)
    print("  pillar at 6 m: dark %.3f  ->  at 2 m: dark %.3f"
          % (tr.eye_dark_fraction, near.eye_dark_fraction))
    assert near.eye_dark_fraction > tr.eye_dark_fraction

    # a target to the left must drive the LEFT ORNs harder, and mirror
    wl = TargetWorld(target=np.array([10.0, 5.0, 2.0]), obstacles=[])
    wr = TargetWorld(target=np.array([10.0, -5.0, 2.0]), obstacles=[])
    _, tl = inp.drive(wl, p, 0.0)
    _, trr = inp.drive(wr, p, 0.0)
    print("  target left : ORN L %.4f R %.4f" % (tl.orn_left, tl.orn_right))
    print("  target right: ORN L %.4f R %.4f" % (trr.orn_left, trr.orn_right))
    assert tl.orn_left > tl.orn_right and trr.orn_right > trr.orn_left
    assert abs(tl.orn_left - trr.orn_right) < 1e-9, "the two must mirror"

    # the two streams must be independent: hiding the target kills the ORN
    # drive but leaves the eyes seeing the scene
    hidden = TargetWorld(target=np.array([10.0, 0.0, 2.0]),
                         obstacles=[Obstacle(x=5.0, y=0.0, radius=2.0,
                                             z=0.0, height=6.0)])
    _, th = inp.drive(hidden, p, 0.0)
    print("  target occluded: cue_valid=%s, ORN L %.4f, eye dark %.3f"
          % (th.cue_valid, th.orn_left, th.eye_dark_fraction))
    assert not th.cue_valid and th.orn_left == 0.0
    assert th.eye_dark_fraction > 0, "the eyes still see the blocker"
    print("demo ok")


if __name__ == "__main__":
    demo()
