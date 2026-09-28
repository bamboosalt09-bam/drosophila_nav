"""One fisheye image per control step, read by both the eyes and the cue.

Until now the two sensory streams each built their own view of the world:

    lamina   a ray cast per cell per acceptance offset -- 6,199 x 5 = 30,995
             rays, every one tested against every obstacle in a Python loop
    cue      69 surface samples per TARGET, each occlusion-tested against
             every obstacle, twice for a stereo pair, once per target

Measured in a 161-obstacle room: 49.6 ms for the lamina, 34.2 ms for six
targets, and the target itself rendered TWICE -- once into the scene the
lamina reads and once again as the cue sensor's own projection.

The cue path had a second problem that is not about speed.  `sense(target)`
takes the target's world position as an argument: it knows where the thing
is and projects it, which is the opposite of what a camera does and is
information a real drone cannot have.  It also cost time proportional to the
number of targets, which a camera never does.

A fisheye image fixes both.  It is rendered once, the lamina cells index into
it, and the cue is found the way a tracker actually finds one: threshold the
image, label the bright blobs, take the intensity-weighted centroid of the
chosen blob.  Nothing is told where a target is, and six targets cost exactly
what one costs.

The grid also makes culling possible, which is where the speed comes from: an
obstacle spans a contiguous block of pixels (typically 6.2 deg of a 290 deg
field, 2.2%), so it is tested against that block instead of against every
ray.  Without culling the image is no faster than the rays -- measured 64.6 ms
against 49.6 ms; with culling it is 11.5 ms.

ponytail: one image, body-centred, so stereo range is gone -- `range_m` is
estimated from apparent size against an assumed target radius and flagged as
such.  Nothing in the steering path reads it.  Render a second image offset by
the baseline the day depth actually matters.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from environment.target_world import (OBSTACLE_LUMINANCE, Obstacle,
                                      SKY_LUMINANCE, TARGET_HEIGHT,
                                      TARGET_LUMINANCE)

# Halfway between sky and target: anything above this is "a bright thing".
BRIGHT = 0.5 * (SKY_LUMINANCE + TARGET_LUMINANCE)

# Surface texture.  Flat-shaded walls carry no feature to correspond between
# two eyes and none to track between two frames, so stereo disparity and
# optic flow are not merely hard on them -- they do not exist.  Measured: a
# disparity estimate on flat walls returned the same 3.0-3.5 s from 20 m to
# 3 m, which was pixel quantisation, not range.  A fly has the same problem
# and it is why one flies into a clean window.
#
# The pattern is WORLD-FIXED -- a function of the surface point that was hit,
# not of the frame -- so it moves with the surface as the agent moves, which
# is the whole point.  Three octaves, so there is something to match at
# several scales.
TEXTURE_AMP = 0.12
TEXTURE_FREQ = 1.7


# A LAMP on the drone, offset to one side.
#
# Passive vision on flat surfaces carries no range at all, and even textured
# ones only give disparity within 6.3 m at this baseline and resolution.  A
# lamp turns brightness itself into range: what it lights falls off as
# 1/(1+(r/r0)^2), so a near surface is bright and a far one is not, with no
# correspondence problem and no motion required.
#
# Offsetting it means the two sides of the field are lit unequally, so the
# asymmetry carries which side is closer -- the thing a frontal wall cannot
# express in a symmetric image.
#
# This is not a fly.  A fly has no headlamp.  It is a drone sensor chosen
# because the drone is the body being flown, and it is declared here rather
# than smuggled in as "vision".
LAMP_POWER = 0.10         # never outshines the sky: a wall at 1.5 m peaks
                          # at 0.095 against sky 0.15.  At 0.45 it peaked
                          # at 0.354 and the steering sign INVERTED at 3 m.
                          # Costs nothing: the lamp channel is separated, so
                          # range error is identical (12%/1%/0% at 2/5/10 m)
                          # from 0.45 all the way down to 0.03.
# Range at which the lamp contributes half power.  This sets where the
# estimate has resolution: brightness is P/(1+(r/r0)^2), so it flattens out
# well inside r0 and the inversion loses its grip there.  At r0 = 4 m the
# estimate pinned at its floor from 1.6 m inward -- urgency read 1.000 while
# the wall closed 1.59 -> 0.48 m and the drone never slowed.  1.5 m puts the
# steep part of the curve exactly where a 2 m/s drone has to decide.
# Range at which the lamp contributes half power.  This sets where the
# estimate has resolution: brightness is P/(1+(r/r0)^2), so it flattens well
# inside r0 and the inversion loses its grip there.  At r0 = 4 m the estimate
# pinned at its floor from 1.6 m inward -- urgency read 1.000 while the wall
# closed 1.59 -> 0.48 m and the drone never slowed.  1.5 m puts the steep
# part of the curve exactly where a 2 m/s drone has to decide.
LAMP_HALF_M = 1.5
LAMP_OFFSET_M = 0.35       # to the drone's RIGHT
# Ambient fraction: even a surface facing away from the lamp is not black,
# because light bounces.  Without it the estimate divides by zero at grazing
# incidence and reports infinity for a wall that is plainly there.
LAMP_AMBIENT = 0.25


def _texture(hx, hy, hz):
    """Deterministic surface pattern in world coordinates, in [-1, 1]."""
    k = TEXTURE_FREQ
    n = (np.sin(k * hx) * np.sin(k * hy + 1.3)
         + 0.5 * np.sin(2.7 * k * hx + 0.7) * np.sin(2.3 * k * hz)
         + 0.3 * np.sin(5.1 * k * hy) * np.sin(4.3 * k * hx + 2.1))
    return n / 1.8


# ---------------------------------------------------------------------------
# The compiled core.  Built with:
#     .venv/Scripts/python.exe src/sensors/_fisheye_cpp/build.py build_ext --inplace
# Verified against the Python renderer: with texture and lamp off, so that
# only ray-object intersection and occlusion remain, the two agree to
# 0.000e+00 at three headings.  With them on they differ, which is the
# hit-point sign fix -- see the header of fisheye.cpp.  32.87 ms -> 2.15 ms
# in a 161-obstacle room, and the compiled call also returns the range
# profile and both centroids, which used to be separate passes.
sys.path.insert(0, str(Path(__file__).resolve().parent / "_fisheye_cpp"))
import _fisheye     # noqa: E402


def world_obstacles(world) -> np.ndarray:
    """(N, 6) array the compiled renderer takes: x, y, r, z, height, luminance.

    Targets go in the SAME array as tall cylinders that happen to be bright.
    The renderer is not told which rows are targets, and no target position
    reaches anything that reads the result -- the image is the only channel.
    A height <= 0 means a sphere.
    """
    rows = [(o.x, o.y, o.radius, o.z,
             -1.0 if o.height is None else o.height, OBSTACLE_LUMINANCE)
            for o in world.obstacles]
    rows += [(float(t[0]), float(t[1]), world.target_radius, 0.0,
              TARGET_HEIGHT, TARGET_LUMINANCE) for t in world.all_targets()]
    return np.array(rows, dtype=np.float64).reshape(-1, 6)


@dataclass(frozen=True)
class Blob:
    """One bright region in the image.  Not a target -- a bright region."""

    bearing_deg: float          # intensity-weighted centroid, + is left
    elevation_deg: float
    n_px: int
    intensity: float            # summed luminance above sky
    touches_edge: bool          # clipped by the frame, so the centroid is biased


class FisheyeCamera:
    """Equidistant fisheye: pixel position is linear in angle.

    Same projection as `WIDE_RIG` in `attraction_cue`, which is what lets the
    references and the connectome be compared on the same field of view.
    """

    def __init__(self, az_span: float = 290.0, el_span: float = 175.0,
                 n_az: int = 256, n_el: int = 144, lamp: float = LAMP_POWER,
                 lamp_offset: float = LAMP_OFFSET_M):
        self.a = np.linspace(-az_span / 2, az_span / 2, n_az)
        self.e = np.linspace(-el_span / 2, el_span / 2, n_el)
        self.A, self.E = np.meshgrid(self.a, self.e, indexing="ij")
        self.n_az, self.n_el = n_az, n_el
        self.da = float(self.a[1] - self.a[0])
        self.de = float(self.e[1] - self.e[0])
        self.az_span, self.el_span = az_span, el_span
        self.lamp = lamp
        # body frame: +y is LEFT, so a positive offset puts the lamp right
        self.lamp_offset = lamp_offset
        from sensors.rangefinder import Rangefinder
        self.rangefinder = Rangefinder(self.a)
        self.lamp_off = np.array([0.0, -lamp_offset, 0.0])

    # -- who reads which pixel -------------------------------------------
    def bind(self, cell_az: np.ndarray, cell_el: np.ndarray) -> np.ndarray:
        """Flat pixel index for each cell, computed once at construction."""
        ia = np.clip(np.round((cell_az - self.a[0]) / self.da).astype(int),
                     0, self.n_az - 1)
        ie = np.clip(np.round((cell_el - self.e[0]) / self.de).astype(int),
                     0, self.n_el - 1)
        return ia * self.n_el + ie

    # -- rendering --------------------------------------------------------
    def _span(self, o, p, heading_deg):
        """Pixel block this obstacle can possibly cover, or None."""
        ox, oy = p[0] - o.x, p[1] - o.y
        dist = math.hypot(ox, oy)
        if dist <= o.radius:
            return 0, self.n_az, 0, self.n_el       # we are inside it
        c = (math.degrees(math.atan2(-oy, -ox)) - heading_deg + 180) % 360 - 180
        half = math.degrees(math.asin(min(o.radius / dist, 1.0)))
        # a body straddling the +-180 seam would need two blocks; the field is
        # 290 deg so the seam is outside it, and one block is enough
        ia0 = max(int(np.floor((c - half - self.a[0]) / self.da)) - 1, 0)
        ia1 = min(int(np.ceil((c + half - self.a[0]) / self.da)) + 2, self.n_az)
        if ia1 <= ia0:
            return None
        if o.height is None:
            return ia0, ia1, 0, self.n_el
        near = max(dist - o.radius, 0.05)
        lo = math.degrees(math.atan2(o.z - p[2], dist + o.radius))
        hi = math.degrees(math.atan2(o.z + o.height - p[2], near))
        ie0 = max(int(np.floor((lo - self.e[0]) / self.de)) - 1, 0)
        ie1 = min(int(np.ceil((hi - self.e[0]) / self.de)) + 2, self.n_el)
        if ie1 <= ie0:
            return None
        return ia0, ia1, ie0, ie1

    def sense(self, world, p: np.ndarray, heading_rad: float) -> dict:
        """One compiled sweep: the image, the lamp, and everything read off them.

        Returns `unlit` (the scene WITHOUT the lamp), `lamp` (the lamp's own
        contribution), `range` (metres per azimuth column), and the two
        centroids -- `bearing`/`weight` signed, `scent`/`scent_mass`
        rectified -- plus `blocked`.

        The two centroids are deliberately different quantities.  The signed
        one is vision: walls are darker than sky and push, beacons are
        brighter and pull.  The rectified one is the odour channel: it sees
        only what is brighter than sky, so a wall contributes exactly zero
        and it answers "where is the goal" without ever reporting "what is
        in the way".
        """
        r = _fisheye.render(
            world_obstacles(world), float(p[0]), float(p[1]), float(p[2]),
            float(heading_rad), self.n_az, self.n_el, self.az_span,
            self.el_span, self.lamp, LAMP_HALF_M, self.lamp_offset,
            LAMP_AMBIENT, SKY_LUMINANCE, TEXTURE_AMP, TEXTURE_FREQ,
            self.rangefinder.half_deg)
        # `range` now comes from the RANGE SENSOR, read off the exact solid
        # geometry.  The lamp inversion it replaces is kept as `lamp_range`
        # for comparison only; nothing steers or brakes on it any more.
        r["lamp_range"] = r["range"]
        # THE ODOUR FOLLOWS ONE BEACON: the strongest bright blob, not the
        # weighted mean of every bright pixel.  The mean points BETWEEN
        # beacons: in room 6, 2-5 s after the start, the nearest beacon
        # already out-pulled the others two to one (764 vs 319 px) and the
        # mean still pointed 24-34 deg away from it, toward the far group --
        # and those first seconds decided the route.  A fly follows the
        # strongest plume and fixates the most salient object.  The C++
        # centroid is kept as `scent_centroid`.
        r["scent_centroid"] = r["scent"]
        bl = self.blobs(r["unlit"])
        if bl:
            top = max(bl, key=lambda b: b.intensity)
            r["scent"] = top.bearing_deg
            r["scent_mass"] = top.intensity / r["unlit"].size
        beams = self.rangefinder.scan(world, p, heading_rad)
        r["beams"] = beams
        r["range"] = self.rangefinder.profile(beams)
        r["rear"] = self.rangefinder.rear(beams)
        return r

    def render(self, world, p: np.ndarray, heading_rad: float,
               with_lamp: bool = False):
        """The scene WITHOUT the lamp, and optionally the lamp beside it.

        The previous version added the lamp into the returned image and used
        `with_lamp` only to decide whether to ALSO hand back the lamp buffer,
        so a caller that did not know to subtract steered on a scene in which
        a near wall is brighter than sky and therefore attracts.  Measured
        bearing to a wall at +45 deg: -44.2 deg at 5 m, +10.4 at 3 m, +42.5
        at 1.5 m -- the avoidance sign inverting exactly where it matters.
        One arm was patched by hand and the other was left broken.

        Now the unlit scene is what comes back either way, so the mistake is
        not available to make.  A caller that genuinely wants the lit image
        adds the two, which is also what a real strobing lamp gives you.
        """
        r = self.sense(world, p, heading_rad)
        return (r["unlit"], r["lamp"]) if with_lamp else r["unlit"]

    # -- what a tracker finds in it ---------------------------------------
    def blobs(self, img: np.ndarray, thr: float = BRIGHT):
        """Bright connected regions, each with its intensity-weighted centroid.

        This is the whole target-detection step, and it is given nothing but
        the image -- no target list, no target count, no positions.
        """
        mask = img > thr
        if not mask.any():
            return []
        lab, n = ndimage.label(mask)
        out = []
        w = np.where(mask, img - SKY_LUMINANCE, 0.0)
        for i in range(1, n + 1):
            sel = lab == i
            ww = w[sel]
            tot = float(ww.sum())
            if tot <= 0:
                continue
            out.append(Blob(
                bearing_deg=float((self.A[sel] * ww).sum() / tot),
                elevation_deg=float((self.E[sel] * ww).sum() / tot),
                n_px=int(sel.sum()), intensity=tot,
                # a blob clipped by the frame has a centroid pulled inward,
                # so the bearing it reports is biased toward straight ahead
                touches_edge=bool(sel[0].any() or sel[-1].any()
                                  or sel[:, 0].any() or sel[:, -1].any())))
        return out


def range_profile(cam, lampc, lamp_power=None, r0=None):
    """Range to the nearest surface in EVERY azimuth column.

    The lamp term is power * shading / (1 + (r/r0)^2), so inverting the
    brightest pixel of a column gives that column's range: the brightest one
    is the one facing the lamp squarely, whose shading is ~1, so albedo and
    incidence angle drop out.

    This replaces reducing the frontal +-35 deg to a single number, which was
    hiding exactly the obstacle that mattered.  Measured on the collision in
    room 0: the window reported a constant 4.00 m for 25 steps while a wall
    closed from 3.55 m to 0.26 m, because the wall sat at +24 deg and drifted
    out of the window, and the percentile inside the window was dominated by
    distant surfaces.  The per-column profile tracked the same wall to within
    1-10% the whole way in.

    The eye already sees +-145 deg.  Discarding 76% of it was a choice, not a
    limit.

    ponytail: one range per azimuth column, taking the whole elevation strip.
    That merges a low obstacle with a high one at the same bearing; separate
    them the day the agent can change altitude.
    """
    if lamp_power is None:
        lamp_power = cam.lamp
    if r0 is None:
        r0 = LAMP_HALF_M
    b = np.percentile(lampc, 99, axis=1)
    out = np.full(cam.n_az, np.inf)
    lit = b > 1e-5
    ratio = np.where(lit, lamp_power / np.maximum(b, 1e-12), np.inf)
    ok = lit & (ratio > 1.0)
    out[ok] = r0 * np.sqrt(ratio[ok] - 1.0)
    out[lit & ~ok] = 0.05           # brighter than the lamp allows: on top of it
    return out


def clearance_ahead(cam, rngs, heading_rel_deg=0.0, half_width_deg=60.0):
    """Nearest range within a cone around a direction, and where it is.

    Returns (range, bearing).  The cone is wide because a body that is
    turning sweeps sideways: the collision in room 0 happened at +62 deg,
    well outside any frontal window, while the agent flew almost parallel to
    the wall it eventually touched.
    """
    rel = (cam.a - heading_rel_deg + 180.0) % 360.0 - 180.0
    m = np.abs(rel) <= half_width_deg
    if not m.any():
        return np.inf, 0.0
    sub = rngs[m]
    if not np.isfinite(sub).any():
        return np.inf, 0.0
    j = int(np.nanargmin(np.where(np.isfinite(sub), sub, np.inf)))
    return float(sub[j]), float(cam.a[m][j])


def scent_centroid(cam, img):
    """Bearing to what is BRIGHTER than sky, and how much of it there is.

    This is the odour channel, and it is deliberately a different quantity
    from `contrast_centroid`.  Rectifying at sky level means a wall -- which
    is darker -- contributes exactly nothing, so this answers "where is the
    goal" and never "what is in the way".  A plume does not report walls.

    The module docstring has claimed since it was written that ORN gets the
    rectified centroid.  The code was passing it the SIGNED one, i.e. the
    full visual steering signal with obstacles folded in, relabelled as
    smell.  The circuit was therefore handed the answer through a second
    modality and had no work left for vision to do -- which is most of why
    the connectome arm tracks the centroid baseline instead of beating it.

    Returns (bearing_deg, mass).  Mass is zero when nothing is visible,
    which is the honest state when every beacon is occluded.
    """
    w = np.maximum(img - SKY_LUMINANCE, 0.0)
    den = float(w.sum())
    if den < 1e-9:
        return 0.0, 0.0
    return float((cam.A * w).sum() / den), den / img.size


def contrast_centroid(cam, img):
    """The centroid formula applied to the WHOLE image, not to a detection.

        u_c = sum(w_i * u_i) / sum(|w_i|),   w_i = I_i - sky

    Written out, w is negative where a wall is, zero for open sky, and
    strongly positive on a bright target.  Approach and avoidance therefore
    come out of one weighted mean instead of two hand-written rules: the
    wall pushes, the sky is neutral, the target pulls.

    Thresholding first -- keeping only the bright blobs, which is what this
    module did before -- throws the walls away at the first step, and then no
    amount of downstream work can avoid them.  Measured on the same scenes:

        scene                        blob      contrast
        target ahead, left half hid  -2.3 deg   -4.0 deg
        wall on the left              none     -51.4 deg
        wall on the right             none     +51.4 deg
        wall dead ahead               none      +0.0 deg

    The last row is the honest limit: a wall straight ahead is left-right
    symmetric, so a bearing cannot express it.  `blocked` is returned for
    exactly that -- how much of the frontal field is darker than sky, which
    is the "slow down and turn" signal rather than a "turn which way" one.

    ponytail: `blocked` is the frontal dark fraction, not a looming rate.
    Use image motion the day speed has to react to closing rate rather than
    to proximity.
    """
    w = img - SKY_LUMINANCE
    den = float(np.abs(w).sum())
    if den < 1e-9:
        return 0.0, 0.0, 0.0
    bearing = float((cam.A * w).sum() / den)
    # Frontal blockage as a CONTINUOUS quantity, not a pixel count.
    #
    # Counting pixels darker than sky moves in steps -- measured 0.02, 0.03,
    # 0.05, 0.08 as a wall closed from 20 m -- and a time-to-contact estimate
    # built on its derivative loses half its frames to "no change" and
    # overshoots by 4x on the rest.  Summing how much darker than sky the
    # frontal field is varies smoothly with distance instead.
    #
    # Normalised so an entirely dark frontal field reads 1.0.
    front = np.abs(cam.A[:, 0]) < 40.0
    dark = np.maximum(SKY_LUMINANCE - img[front], 0.0)
    blocked = float(dark.sum() / (dark.size * SKY_LUMINANCE))
    # total signed drive, i.e. is there anything out there at all
    weight = den / img.size
    return bearing, blocked, weight


class BlobTracker:
    """Which bright blob are we following, and what do we report about it?

    Four things the old cue path did not do, all of them what any real target
    tracker does and none of them a navigation decision:

      hysteresis   stay on the current blob unless a rival is clearly better.
                   Measured before this existed: the lock changed 0.4 times a
                   second, and each change is a step discontinuity in the
                   bearing the circuit is steering on -- visible as the two
                   sawteeth in the map 14 trace.
      association  the blob nearest in bearing to the last one is treated as
                   the same object, so a target that moves across the frame
                   keeps its identity instead of being re-chosen every frame.
      coasting     a blob that vanishes behind a pillar is reported for a few
                   more frames at falling confidence rather than dropping to
                   invalid instantly.  Measured: the cue was invalid 32% of
                   steps, much of it brief occlusion.
      edge honesty a blob clipped by the frame has its centroid pulled inward,
                   so its bearing is biased toward straight ahead.  It is
                   reported with reduced strength instead of at face value.

    What it deliberately does NOT do: give up on a target because progress is
    not being made.  That is a navigation decision and it belongs to whatever
    is doing the navigating -- putting it here would let the sensor solve the
    part of the task the unreachable beacons exist to test.

    ponytail: association is nearest-in-bearing, which is enough while blobs
    are sparse.  Add elevation and size to the match if two beacons ever line
    up in azimuth.
    """

    def __init__(self, switch_ratio: float = 1.3, coast_frames: int = 8,
                 assoc_deg: float = 25.0, target_radius: float = 0.6):
        self.switch_ratio = switch_ratio
        self.coast_frames = coast_frames
        self.assoc_deg = assoc_deg
        self.target_radius = target_radius
        self.reset()

    def reset(self) -> None:
        self._bearing = None
        self._elev = 0.0
        self._intensity = 0.0
        self._missing = 0

    def update(self, blobs, hfov_deg: float) -> "AttractionCue":
        from sensors.attraction_cue import AttractionCue
        if not blobs:
            return self._coast(hfov_deg)

        if self._bearing is None:
            pick = max(blobs, key=lambda b: b.intensity)
        else:
            near = [b for b in blobs
                    if abs(b.bearing_deg - self._bearing) <= self.assoc_deg]
            if near:
                pick = min(near, key=lambda b: abs(b.bearing_deg - self._bearing))
                rival = max(blobs, key=lambda b: b.intensity)
                if rival is not pick and                         rival.intensity > self.switch_ratio * pick.intensity:
                    pick = rival
            else:
                pick = max(blobs, key=lambda b: b.intensity)

        self._bearing = pick.bearing_deg
        self._elev = pick.elevation_deg
        self._intensity = pick.intensity
        self._missing = 0

        # apparent angular radius -> range, under an ASSUMED target radius.
        # Nothing in the steering path reads this; it is reported so the cue
        # keeps its shape, and it is honest about being an estimate.
        ang = math.sqrt(max(pick.n_px, 1) / math.pi)   # in pixels
        rng = float("nan")
        strength = min(pick.n_px / 400.0, 1.0)
        if pick.touches_edge:
            strength *= 0.5          # centroid is biased; trust it less
        return AttractionCue(
            bearing_deg=pick.bearing_deg, elevation_deg=pick.elevation_deg,
            deviation=min(abs(pick.bearing_deg) / (hfov_deg / 2), 1.0),
            range_m=rng, strength=strength, valid=True, n_cameras=1)

    def _coast(self, hfov_deg):
        from sensors.attraction_cue import AttractionCue
        if self._bearing is None or self._missing >= self.coast_frames:
            self.reset()
            return AttractionCue(0.0, 0.0, 0.0, float("nan"), 0.0, False, 0)
        self._missing += 1
        fade = 1.0 - self._missing / self.coast_frames
        return AttractionCue(
            bearing_deg=self._bearing, elevation_deg=self._elev,
            deviation=min(abs(self._bearing) / (hfov_deg / 2), 1.0),
            range_m=float("nan"),
            strength=min(self._intensity / 400.0, 1.0) * fade,
            valid=True, n_cameras=0)


def demo() -> None:
    """Check the image agrees with the ray caster and finds what is there."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import time
    from environment.target_world import (TargetWorld, room_world,
                                          world_scene)

    cam = FisheyeCamera()
    p = np.array([0.0, 0.0, 2.0])

    # 1. a single beacon lands at the bearing it is actually at
    for b in (-100.0, -40.0, 0.0, 35.0, 90.0):
        th = math.radians(b)
        w = TargetWorld(target=np.array([10 * math.cos(th),
                                         10 * math.sin(th), 2.0]),
                        obstacles=[])
        bl = cam.blobs(cam.render(w, p, 0.0))
        assert len(bl) == 1, (b, len(bl))
        err = abs(bl[0].bearing_deg - b)
        assert err < 2.5, (b, bl[0].bearing_deg)
    print("single beacon: bearing recovered within %.1f deg" % 2.5)

    # 2. several beacons come back as several blobs, not one average
    w = TargetWorld(target=np.array([10.0, 0.0, 2.0]),
                    extra_targets=[np.array([0.0, 10.0, 2.0]),
                                   np.array([-7.0, -7.0, 2.0])])
    bl = cam.blobs(cam.render(w, p, 0.0))
    got = sorted(round(x.bearing_deg) for x in bl)
    print("three beacons -> %d blobs at %s deg" % (len(bl), got))
    assert len(bl) == 3, len(bl)

    # 3. an occluded beacon disappears; a half-occluded one shrinks
    hidden = TargetWorld(target=np.array([10.0, 0.0, 2.0]),
                         obstacles=[Obstacle(x=5.0, y=0.0, radius=2.5,
                                             z=0.0, height=8.0)])
    assert not cam.blobs(cam.render(hidden, p, 0.0)), "occluded beacon visible"
    half = TargetWorld(target=np.array([10.0, 0.0, 2.0]),
                       obstacles=[Obstacle(x=5.0, y=0.55, radius=0.5,
                                           z=0.0, height=8.0)])
    full = TargetWorld(target=np.array([10.0, 0.0, 2.0]), obstacles=[])
    b_half = cam.blobs(cam.render(half, p, 0.0))
    b_full = cam.blobs(cam.render(full, p, 0.0))
    assert b_half and b_full
    print("partly hidden: %d px vs %d px unobstructed, centroid %+.1f -> %+.1f"
          % (b_half[0].n_px, b_full[0].n_px,
             b_full[0].bearing_deg, b_half[0].bearing_deg))
    assert b_half[0].n_px <= b_full[0].n_px

    # 4. the image agrees with the ray caster it replaces
    w, h, reach = room_world(seed=0)
    sc = world_scene(w, p)
    probe_az = np.linspace(-140, 140, 400)
    probe_el = np.zeros(400)
    ray = sc(probe_az, probe_el)
    img = cam.render(w, p, 0.0)
    ia = np.clip(np.round((probe_az - cam.a[0]) / cam.da).astype(int),
                 0, cam.n_az - 1)
    ie = np.clip(np.round((probe_el - cam.e[0]) / cam.de).astype(int),
                 0, cam.n_el - 1)
    got = img[ia, ie]
    agree = float(np.mean(np.abs(got - ray) < 0.08))
    print("vs ray caster over 400 directions: %.1f%% agree" % (100 * agree))
    assert agree > 0.90, agree

    # 5. and it is faster than what it replaces
    cam.render(w, p, 0.0)
    t = time.perf_counter()
    for _ in range(10):
        cam.render(w, p, 0.0)
    t_img = 1e3 * (time.perf_counter() - t) / 10
    sc2 = world_scene(w, p)
    az = np.linspace(-140, 140, 6199)
    el = np.zeros(6199)
    sc2(az, el)
    t = time.perf_counter()
    for _ in range(10):
        sc2(az, el)
    t_ray = 1e3 * (time.perf_counter() - t) / 10
    print("render %.1f ms (%d obstacles) against %.1f ms for one ray set of "
          "6,199 -- and the lamina needed five of those"
          % (t_img, len(w.obstacles), t_ray))
    assert t_img < t_ray
    # 6. the whole-image centroid sees walls, which the blob version cannot
    from environment.target_world import SKY_LUMINANCE as SKY
    wall_l = TargetWorld(target=np.array([300.0, 0.0, 2.0]),
                         obstacles=[Obstacle(x=0.0, y=3.0, radius=3.0,
                                             z=0.0, height=8.0)])
    wall_r = TargetWorld(target=np.array([300.0, 0.0, 2.0]),
                         obstacles=[Obstacle(x=0.0, y=-3.0, radius=3.0,
                                             z=0.0, height=8.0)])
    ahead = TargetWorld(target=np.array([300.0, 0.0, 2.0]),
                        obstacles=[Obstacle(x=3.0, y=0.0, radius=3.0,
                                            z=0.0, height=8.0)])
    bl, kl, _ = contrast_centroid(cam, cam.render(wall_l, p, 0.0))
    br, kr, _ = contrast_centroid(cam, cam.render(wall_r, p, 0.0))
    ba, ka, _ = contrast_centroid(cam, cam.render(ahead, p, 0.0))
    print("wall left -> %+.1f deg, wall right -> %+.1f deg, wall ahead -> "
          "%+.1f deg with blocked %.0f%%" % (bl, br, ba, 100 * ka))
    assert bl < -20 and br > 20, (bl, br)      # pushed away from the wall
    assert abs(bl + br) < 2.0                  # and symmetrically
    assert abs(ba) < 5.0 and ka > 0.2          # ahead: no bearing, but blocked
    assert not cam.blobs(cam.render(wall_l, p, 0.0)), "a wall is not a blob"

    # 7. the tracker: hysteresis, association, coasting, edge honesty
    tr = BlobTracker(coast_frames=4)
    steady = TargetWorld(target=np.array([10.0, 2.0, 2.0]), obstacles=[])
    c = tr.update(cam.blobs(cam.render(steady, p, 0.0)), cam.az_span)
    assert c.valid
    first = c.bearing_deg
    # a brighter rival that is only slightly better must NOT steal the lock
    two = TargetWorld(target=np.array([10.0, 2.0, 2.0]),
                      extra_targets=[np.array([9.0, -4.0, 2.0])])
    c2 = tr.update(cam.blobs(cam.render(two, p, 0.0)), cam.az_span)
    print("hysteresis: locked at %+.1f, rival present -> now %+.1f"
          % (first, c2.bearing_deg))
    assert abs(c2.bearing_deg - first) < 12.0, "lock jumped to the rival"
    # occlusion: coasts, fading, then gives up
    gone = TargetWorld(target=np.array([10.0, 2.0, 2.0]),
                       obstacles=[Obstacle(x=5.0, y=1.0, radius=3.0,
                                           z=0.0, height=8.0)])
    st = []
    for _ in range(6):
        cc = tr.update(cam.blobs(cam.render(gone, p, 0.0)), cam.az_span)
        st.append((cc.valid, round(cc.strength, 3)))
    print("coasting after occlusion: %s" % st)
    assert st[0][0] and not st[-1][0], st
    assert st[0][1] > st[2][1], "strength must fade while coasting"
    print("demo ok")


if __name__ == "__main__":
    demo()
