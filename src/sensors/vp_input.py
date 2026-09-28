"""Fisheye image straight into the visual projection neurons.

Same idea as `ConnectomeInput`, one stage later: each projection neuron reads
the pixel its receptive field points at, instead of each lamina cell reading
the pixel its ommatidium points at.  The optic lobe between them is not
simulated, because this rate model has no reason to reproduce what it
computes and measurement says it got the sign wrong when it tried.

The ORN cue is the strongest bright blob (winner-take-all), see fisheye.sense.

ponytail: every projection neuron gets the same quantity -- local contrast.
Their real specialisations (LPLC2 looming, LC11 small objects, ...) are not
modelled, because encoding them means designing the features rather than
measuring what the circuit does with them.  Add per-type preprocessing only
if the undifferentiated version is shown to be the limit.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from sensors.fisheye import FisheyeCamera

# Time to contact at which a manoeuvre is fully urgent.  A fly starts its
# evasive turn about 100 ms before contact; a 2 m/s drone needs longer
# because its turn takes longer.
# 2.0 s: one turn (0.64 s at 90 deg/s through 90 deg) with a factor of three
# of margin.  A sweep over 1.0-5.0 preferred 1.0-2.0, but that was measured
# before the absolute standoff existed and before the per-azimuth range
# profile, so it is not evidence for this value any more -- the physical
# argument is.  Re-measure it when something depends on it.
TAU_CRIT = 2.0
# Sub-steps the circuit takes per control cycle.  Must match the runner's
# SETTLE: the drive is now a SEQUENCE over these, not one held column.
#
# Holding one column for all 20 was feeding the connectome a photograph.
# The fly's visual system is a motion system end to end -- Reichardt
# correlators in the medulla, LPLC2/LC/HS carrying motion rather than
# luminance -- and a still image run to steady state has no time axis for
# any of it to act on.  Interpolating the last frame into the current one
# across the sub-steps replays the 100 ms of motion that just happened at
# 5 ms resolution.  Costs one frame of delay, which every real camera
# pipeline has anyway, and no extra render.
#
# This is deliberately NOT a designed feature: no looming detector, no
# optic flow.  It is the time axis not being thrown away.  The
# differentiation is left to the circuit's own dynamics.
N_SUB = 20

# RANGE ENTERS AS THREAT, on the fly's own looming detectors.
#
# Distance has to come from a range sensor -- the fly's own looming pathway
# does not produce it here: driven by vision alone, the escape descending
# neurons DNp01/02/04/06/11 tracked 1/tau at -0.08..+0.04.  But fed at the
# looming detectors, that pathway is the strongest thing the circuit does.
# One-sided drive of 1.0/cell into LC4 raised the SAME-side DNp04 by +0.56,
# DNp01 (the giant fibre) +0.34, DNp02 +0.21, DNp11 +0.18, with the other
# side near zero; LPLC2 gave DNp01 +0.24, DNp04 +0.18.  The pooled steering
# readout moved by 0.002.  Excitation is the biologically right sign for a
# looming detector, so nothing is inverted.
#
# This REPLACES injecting proximity as "darkness" into all 9,188 projection
# neurons.  That did turn the drone away, but only because its sign was
# chosen to; this uses the pathway flies actually use for threats, and its
# output is read separately from steering, by the runner.
#
# Strength = (THREAT_TTC / time-to-contact)^2, capped.  Time to contact
# along a beam is range / (speed * cos(bearing)): what LC4/LPLC2 encode is
# APPROACH, not nearness.  It was (1.5 / range)^2 in any direction, so a wall
# running alongside -- never getting closer -- read as a threat, and the
# more beams the drone had the more of those walls it saw: in room 6 the
# fly swung at 85-135 deg/s for a wall 2 m off at 45 deg, and flights with
# 0-1 beams (front only) were the cleanest.  Now a wall at 90 deg is 0, one
# at 45 deg and 2 m is 0.28 (was 0.56), and head-on at 1.5 m and 2 m/s is 1,
# the old anchor.
THREAT_TTC = 0.75            # s
THREAT_MAX = 4.0
THREAT_GAIN = 0.5            # drive per cell at strength 1
THREAT_TYPES = ("LC4", "LPLC2")
ESCAPE_TYPES = ("DNp01", "DNp02", "DNp04", "DNp06", "DNp11")
# Steering is read from the identified steering descending neurons, not
# from all 1,304 pooled.  Pooled, a one-sided LC4 threat moved the readout by
# only 0.0025 -- but the steering gain is ~-2.8e4, so that became a 4,000
# deg/s intended turn TOWARD the threat: with side beams the drone crawled
# (81% stopped, 2.1 m in 60 s).  The same LC4 drive raised DNa01/DNa02 on
# the OPPOSITE side, i.e. away.
STEER_TYPES = ("DNa01", "DNa02")


def looming(rng, bearing_deg, speed):
    """Threat per reading from its time to contact; 0 if not closing."""
    closing = speed * np.cos(np.radians(bearing_deg))
    ttc = np.where((closing > 1e-3) & np.isfinite(rng),
                   rng / np.maximum(closing, 1e-3), np.inf)
    return np.minimum((THREAT_TTC / ttc) ** 2, THREAT_MAX)


MIRROR_BAND = 10.0           # deg


def mirror_weights(az, band=MIRROR_BAND):
    """Per-cell input weight so a mirror image gets the same TOTAL drive.

    Within each azimuth band, the cells looking at +az and those at -az
    often differ in number (LC4+LPLC2 are 165 left vs 146 right), so the
    same stimulus on the other side arrived stronger or weaker purely by
    count.  Each cell's share is scaled by (mean count of the pair) / (its
    side's count) -- the rule the odour and gyro inputs already follow per
    side.  Cells without a direction, or whose mirror band is empty, keep 1.

    ponytail: azimuth only; elevation bands are not balanced.
    """
    az = np.asarray(az, dtype=float)
    w = np.ones(len(az))
    ok = np.isfinite(az)
    idx = np.where(ok, np.floor(np.abs(az) / band), -1).astype(int)
    pos, neg = ok & (az >= 0), ok & (az < 0)
    for b in np.unique(idx[ok]):
        p, n = pos & (idx == b), neg & (idx == b)
        if p.any() and n.any():
            m = 0.5 * (p.sum() + n.sum())
            w[p], w[n] = m / p.sum(), m / n.sum()
    return w


def threat_level(beams, centres_deg, speed):
    """(left, right): the strongest looming reading on each side."""
    if not len(beams):
        return 0.0, 0.0
    c = np.asarray(centres_deg, dtype=float)
    s = looming(beams, c, speed)
    return (float(s[c >= 0].max()) if (c >= 0).any() else 0.0,
            float(s[c <= 0].max()) if (c <= 0).any() else 0.0)

# GYRO -> JOHNSTON'S ORGAN.  The drone's measured yaw rate goes to the JO
# afferents in the subnetwork (19 cells: JO-EV3 14, JO-FV 3, JO-EV1, JO-ED2_a).
# Drosophila's main rotation sensor is the halteres, but antennal JO works as
# a gyroscope in hawkmoths (Sane et al. 2007), and JO carries real weight to
# the descending neurons here: driven on one side only it moved the steering
# readout by +202..+436 deg/s (left) and -54..-116 (right), where 7 olfactory
# channels managed 0.2-10.
#
# SIGN is set by what rotation sensing is FOR -- a corrective reflex that
# opposes the rotation -- not by trial: turning left drives the RIGHT JO
# harder, and one-sided right JO drive was measured to turn the drive right.
# Push-pull like the goal odour, normalised per side because the counts are
# 14:5.  With no rotation both sides get GYRO_BASE, which the static
# calibration absorbs into `zero`.  GYRO_BASE is a declared setting, chosen
# for a correction of about a third of the 90 deg/s limit at full rate.
GYRO_FULL = math.radians(90.0)
GYRO_BASE = 1.0

# SPONTANEOUS ALTERNATION was here (v8-v11): a push through PFL3 after a
# full 360 deg turn within 20 s, later only within a 3 m radius.  Removed:
# the corridor loop it was aimed at was a standing RIGHT turn from the
# circuit's own left/right asymmetry (docs/HANDOFF.md, 2026-09-28), and
# angle/radius thresholds
# could not tell that loop from a room-scale sweep anyway.  git has it.

# TRIED AND REMOVED: aversion through the CO2-sensing ORNs (ORN_V, the V
# glomerulus; innate avoidance in Drosophila).  Proximity was sent by side
# into the 23 of them in the subnetwork.  They saturate by gain 30 (rate
# 0.652) and even saturated they moved the steering readout by 1-4 deg/s,
# slightly TOWARD the wall, against 110-148 deg/s of avoidance from the
# range drive above.  23 cells do not carry enough weight to the descending
# neurons -- the same starvation of weak paths seen in the optic lobe.


class VPInput:
    """Build the drive vector for a projection-neuron-injected subnetwork."""

    def __init__(self, net, ann, info, drive_gain: float = 0.20,
                 orn_gain: float = 200.0, cue_types=("ORN_DA1", "ORN_VM2",
                                                     "ORN_VA1v", "ORN_DC3",
                                                     "ORN_VA6"),
                 fixed_strength=None):
        self.net, self.ann = net, ann
        self.drive_gain, self.orn_gain = drive_gain, orn_gain
        # None means the odour's strength is its MASS, i.e. how much of the
        # field is brighter than sky.  A constant here made "every beacon is
        # occluded" -- where the rectified centroid returns (0.0, 0.0) --
        # indistinguishable from "the goal is dead ahead", because ORN still
        # drove at full strength claiming bearing zero.  A room with
        # unreachable beacons is exactly that condition.
        self.fixed_strength = fixed_strength
        self.cam = FisheyeCamera()

        self.vp_rows = info["vp_rows"]
        az, el = info["vp_az"], info["vp_el"]
        self._px = self.cam.bind(np.nan_to_num(az), np.nan_to_num(el))
        self.vp_az = az
        # which azimuth column of the range profile each neuron looks down
        # Some projection neurons get no receptive direction from the
        # wiring (NaN).  Casting NaN to int silently pointed them at column
        # 0, i.e. at -145 deg; they now read nothing instead.
        self._az_ok = np.isfinite(az) & np.isfinite(el)
        self._vp_w = mirror_weights(az)
        # the MIRROR TWIN reads each cell's pixel at -az: the mirror image
        # of the scene, exactly, because the camera's azimuth grid is
        # symmetric about 0
        self._px_m = self.cam.bind(-np.nan_to_num(az), np.nan_to_num(el))
        # MIRROR TWIN.  When True, drive() also builds `d_mirror`, the drive
        # this circuit would receive in the mirror-image world, and the
        # runner steers on (turn - mirror turn) / 2.  The circuit is not
        # mirror-symmetric even with equal input per side: a corridor with a
        # wall 1.5 m on each side turned it -34 deg/s and in room 6 that made
        # a clockwise orbit at every beam count, 0 beams included.  A real
        # fly is bilaterally symmetric; the residue is the reconstruction's.
        self.twin = False
        self.d_mirror = None

        ct = ann["cell_type"].astype(str).to_numpy(dtype="<U48")
        side = ann["side"].to_numpy(dtype="<U16")
        is_orn = np.array([any(c.startswith(t) for t in cue_types) for c in ct])
        self.orn_l = np.flatnonzero(is_orn & (side == "left"))
        self.orn_r = np.flatnonzero(is_orn & (side == "right"))
        self.orn_scale_l = 1.0 / max(len(self.orn_l), 1)
        self.orn_scale_r = 1.0 / max(len(self.orn_r), 1)
        # Threat goes to each LC4/LPLC2 cell in ITS OWN receptive direction.
        # It used to be one number per side, given alike to all 311 of them
        # -- two cell types and every viewing direction lumped into one
        # signal.  A cell whose direction no beam covers now gets nothing.
        az_full = np.full(net.n, np.nan)
        az_full[self.vp_rows] = self.vp_az
        cand = np.flatnonzero(np.isin(ct, THREAT_TYPES))
        cand = cand[np.isfinite(az_full[cand])]
        self.thr_rows = cand
        self.thr_az = az_full[cand]
        self._thr_w = mirror_weights(self.thr_az)
        is_st = np.isin(ct, STEER_TYPES)
        self.steer_l = np.flatnonzero(is_st & (side == "left"))
        self.steer_r = np.flatnonzero(is_st & (side == "right"))
        is_esc = np.isin(ct, ESCAPE_TYPES)
        self.esc_l = np.flatnonzero(is_esc & (side == "left"))
        self.esc_r = np.flatnonzero(is_esc & (side == "right"))
        is_jo = np.char.startswith(ct, "JO")
        self.jo_l = np.flatnonzero(is_jo & (side == "left"))
        self.jo_r = np.flatnonzero(is_jo & (side == "right"))
        self.jo_scale_l = 1.0 / max(len(self.jo_l), 1)
        self.jo_scale_r = 1.0 / max(len(self.jo_r), 1)
        self.gyro_base = GYRO_BASE
        self.yaw_rate = 0.0       # rad/s, set by the runner from the plant
        self.speed = 2.0          # current forward speed, for looming
        self.n_sub = N_SUB
        self._prev_val = None         # last frame's per-neuron drive

    def reset(self) -> None:
        self._prev_val = None
        self._prev_val_m = None

    def drive(self, world, position, heading_rad):
        # One compiled sweep returns the unlit scene, the per-azimuth range
        # and both centroids.
        r = self.cam.sense(world, position, heading_rad)
        beams_m = r["beams"]
        scene = r["unlit"].ravel()  # the lamp is NEVER in the steering image
        cen = self.cam.rangefinder.centres
        a_l, a_r = threat_level(beams_m, cen, self.speed)
        # THE ODOUR CHANNEL IS NOT THE VISION CHANNEL.  ORN gets the bearing
        # of the strongest bright blob only -- "where is the goal"; "what is
        # in the way" is left to the visual injection.  Feeding ORN the
        # signed centroid once handed the circuit a finished steering command
        # through a second modality.
        scent, scent_mass = r["scent"], r["scent_mass"]
        frac = float(np.clip(scent / (self.cam.az_span / 2), -1.0, 1.0))
        s = (scent_mass if self.fixed_strength is None
             else self.fixed_strength)
        # gyro: opposing push-pull on JO; + yaw is a LEFT turn
        g = float(np.clip(-self.yaw_rate / GYRO_FULL, -1.0, 1.0))
        d = self._build(scene[self._px], self.thr_az, frac, g, s, beams_m,
                        cen, "_prev_val")
        if self.twin:
            # the mirror world: every azimuth negated, left and right swapped
            self.d_mirror = self._build(scene[self._px_m], -self.thr_az,
                                        -frac, -g, s, beams_m, cen,
                                        "_prev_val_m")
        return d, {"bearing": r["bearing"], "scent": scent,
                   "scent_mass": scent_mass,
                   "blocked": r["blocked"], "weight": r["weight"],
                   "rear": r["rear"], "threat_in": (a_l, a_r),
                   "beams": beams_m}

    def _build(self, lum, thr_az, frac, g, s, beams_m, cen, prev_attr):
        """The drive for one view of the world (as seen, or its mirror)."""
        # contrast, as the lamina transmits it: a uniform scene has no
        # direction in it and must not drive anything
        lum = lum - lum.mean()
        # Replay prev -> cur across the sub-steps, so the drive is a SEQUENCE
        # rather than one column held for the whole cycle.  Holding it was
        # feeding the connectome a photograph: at tau 20 ms and rho 0.50 the
        # effective time constant is 40 ms, so 100 ms settles to 92% and
        # every frame started from a state washed clean.  Deliberately NOT a
        # designed feature -- no looming detector, no optic flow; the
        # differentiation is left to the circuit's own dynamics.  Costs one
        # frame of delay, which every real camera pipeline has.
        val = lum * self.drive_gain * self._az_ok * self._vp_w
        prev = getattr(self, prev_attr, None)
        prev = prev if prev is not None else val
        setattr(self, prev_attr, val)
        ramp = np.linspace(1.0 / self.n_sub, 1.0, self.n_sub)
        seq = prev[:, None] * (1.0 - ramp) + val[:, None] * ramp

        d = torch.zeros(self.net.n, self.n_sub)
        d[self.vp_rows] = torch.from_numpy(seq.astype(np.float32))
        # Each cell takes the NEAREST beam, so every cell gets an input and
        # the same wall drives the circuit equally hard whatever the beam
        # count.  Reading only beams whose cone covered the cell's direction
        # left 176 of 311 cells with nothing at 12 beams, so the same wall
        # arrived about twice as strong at 24 and room-6 flights split 3.4 s
        # in.  Fewer beams now means coarser angular resolution, not weaker
        # threat.
        if len(cen):
            gap = np.abs((thr_az[:, None] - cen[None, :] + 180.0)
                         % 360.0 - 180.0)
            nb = np.argmin(gap, axis=1)
            s_cell = looming(beams_m[nb], cen[nb], self.speed)
        else:
            s_cell = np.zeros(len(self.thr_rows))
        d[self.thr_rows] += torch.from_numpy(
            (THREAT_GAIN * self._thr_w * s_cell).astype(np.float32))[:, None]
        base = self.orn_gain * s
        # odour does not flicker within a control cycle
        d[self.orn_l, :] = base * (1 + frac) * self.orn_scale_l
        d[self.orn_r, :] = base * (1 - frac) * self.orn_scale_r
        d[self.jo_l, :] = self.gyro_base * (1 + g) * self.jo_scale_l
        d[self.jo_r, :] = self.gyro_base * (1 - g) * self.jo_scale_r
        return d


def demo() -> None:
    """mirror_weights: equal total per mirror band, whatever the counts."""
    az = np.array([5.0, 6.0, 7.0, -5.0, 25.0, -25.0, -26.0, 40.0, np.nan])
    w = mirror_weights(az)
    assert abs(w[:3].sum() - w[3]) < 1e-12            # 3 cells vs 1 cell
    assert abs(w[4] - w[5:7].sum()) < 1e-12           # 1 vs 2
    assert w[7] == 1.0 and w[8] == 1.0                # no mirror / no direction
    print("vp_input demo ok")


if __name__ == "__main__":
    demo()
