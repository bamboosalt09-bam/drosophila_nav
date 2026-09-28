"""Fisheye image straight into the visual projection neurons.

Same idea as `ConnectomeInput`, one stage later: each projection neuron reads
the pixel its receptive field points at, instead of each lamina cell reading
the pixel its ommatidium points at.  The optic lobe between them is not
simulated, because this rate model has no reason to reproduce what it
computes and measurement says it got the sign wrong when it tried.

The attraction cue (the strongest bright blob, see fisheye.sense) goes to
PFL3 as a goal bearing, push-pull -- see CUE_GAIN.

ponytail: every projection neuron gets the same quantity -- local contrast.
Their real specialisations (LPLC2 looming, LC11 small objects, ...) are not
modelled, because encoding them means designing the features rather than
measuring what the circuit does with them.  Add per-type preprocessing only
if the undifferentiated version is shown to be the limit.
"""
from __future__ import annotations

import math

import numpy as np
import os
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")  # see flywire_rate.py
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
    side's count) -- the rule the goal cue follows per
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

# THE OPTOMOTOR REFLEX: self-rotation, seen by the camera, into HS/H2.
#
# A fly turning one way sees the world stream the other way, and its
# lobula plate's horizontal wide-field cells (HSE, HSN, HSS, H2) turn it
# back -- which is what keeps a fly's small built-in turning bias from
# closing into a circle.  Without it the circuit's residual right bias
# (~-9 deg/s on symmetric scenes) turned room 6 into a clockwise orbit
# whenever nothing was in view.  The cells are in the subnet (1 per side
# each) and reach DNa01/02 strongly and symmetrically: 0.1/cell on one side
# turns the drone +70.5 (left cells) / -69.8 (right) deg/s, linear to 0.5.
#
# The rotation is MEASURED FROM THE IMAGE, as the fly does -- no extra
# sensor: the per-azimuth brightness profile of this frame against the
# last one, shifted to the best match (`yaw_flow`).  The side excited is
# the one whose cells turn the drone AGAINST the rotation, and the drive
# per degree/s is set so the reflex cancels OPTO_K of the rotation; both are
# measured by the runner (`wire_opto`), not assumed.
OPTO_K = 0.5                   # fraction of the measured rotation turned back
OPTO_TYPES = ("HSE", "HSN", "HSS", "H2")
OPTO_MAX_SHIFT = 12            # columns (~13.6 deg) searched per frame
CONTROL_DT = N_SUB * 0.005     # s between frames


def yaw_flow(prev, cur, da_deg, dt=CONTROL_DT, max_shift=OPTO_MAX_SHIFT):
    """Rotation rate (deg/s, + = LEFT) from two brightness-per-azimuth
    profiles: the shift that best lines them up, to a fraction of a column.

    Turning left moves everything to lower azimuth, i.e. lower column index,
    so cur[i] ~ prev[i + s] with s > 0.
    """
    a = cur - cur.mean()
    b = prev - prev.mean()
    n = len(a)
    sc = []
    for s in range(-max_shift, max_shift + 1):
        lo, hi = max(0, -s), min(n, n - s)
        sc.append(float(np.dot(a[lo:hi], b[lo + s:hi + s])) / max(hi - lo, 1))
    sc = np.array(sc)
    k = int(np.argmax(sc))
    frac = 0.0
    if 0 < k < len(sc) - 1:
        den = sc[k - 1] - 2 * sc[k] + sc[k + 1]
        if den < 0:
            frac = 0.5 * (sc[k - 1] - sc[k + 1]) / den
    return (k - max_shift + frac) * da_deg / dt


# NO GYRO INPUT (removed 2026-09-28, user: "biological route, or none").
# The IMU yaw rate went to Johnston's organ, push-pull.  Measured against the
# pooled 1,304-DN readout it was strong; against DNa01/02 it moved the turn
# by 0.2 deg/s at 90 deg/s of rotation -- dead since v2.  The fly's real
# rotation sensor is the haltere: MaleCNS has 205 haltere afferents
# (entryNerve DMetaN, subclass "haltere"), no direct synapse onto DNa01/02,
# 2,489 synapses via 98 relay cells (PS059, AN02A002, PS013, GNG100, ...).
# Only 2 of the 205 are in this brain-only subnet, so using them means
# rebuilding it -- recorded in docs/HANDOFF.md as the way back in.

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


# THE VIRTUAL ATTRACTION CUE GOES TO PFL3, not to the ORNs.
#
# ORN (5 glomerular types) barely reaches the steering neurons DNa01/02:
# odour alone, beacon 60 deg left, turned the drone at most +3.8 deg/s at
# any strength, and 0.0 at the strength a beacon has at 10-25 m.  It had
# been measured strong only against the old pooled 1,304-DN readout; after
# steering moved to DNa01/02 nobody re-measured it, so from v2 to v14 every
# pull toward a goal came from VISION of the bright beacon alone, which
# walls outvote (room 6: a beacon at +105 deg for 6 s, the drone flying
# away).  PFL3 is the central complex's goal -> steering output
# (Westeinde et al. 2024, this project's Stage 0 core, PFL3 -> DNa02):
# one-sided at 0.02/cell it turns the drone ~72 deg/s.
#
# Push-pull by bearing, split per side by cell count (7 left, 6 right).
# Which side turns which way is MEASURED by the runner (`wire_cue`), not
# assumed; the default below is the last measurement (right PFL3 turned
# the drone left, +2,741 deg/s at 1/cell).
# Strength is a detection confidence that saturates with the blob's
# mass, m / (m + CUE_HALF): the raw mass falls as 1/d^2 and at 10 m was
# 0.003 -- too little to matter anywhere.  CUE_HALF is a beacon's mass at
# 25 m, where the strength is one half.
CUE_GAIN = 0.01               # drive per cell at strength 1, bearing 0
CUE_HALF = 0.0007


class VPInput:
    """Build the drive vector for a projection-neuron-injected subnetwork."""

    def __init__(self, net, ann, info, drive_gain: float = 0.20,
                 fixed_strength=None):
        self.net, self.ann = net, ann
        self.drive_gain = drive_gain
        # None: strength from the blob's mass (0 when no beacon is in view).
        # A constant made "every beacon is occluded" indistinguishable from
        # "the goal is dead ahead"; a room with unreachable beacons is
        # exactly that condition.
        self.fixed_strength = fixed_strength
        self.cue_gain = CUE_GAIN
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
        # A MIRROR TWIN lived here (v14-v16): a second copy of the circuit
        # fed the mirror image, steering on (turn - mirror turn) / 2.  It
        # removed the circuit's left/right bias exactly, and with it every
        # signal a symmetric scene carries -- a wall dead ahead gave 0 deg/s
        # all the way in.  Removed once the drone layer took over modifying
        # the path around walls.  git has it.

        ct = ann["cell_type"].astype(str).to_numpy(dtype="<U48")
        side = ann["side"].to_numpy(dtype="<U16")
        is_pfl3 = ct == "PFL3"
        self.pfl3_l = np.flatnonzero(is_pfl3 & (side == "left"))
        self.pfl3_r = np.flatnonzero(is_pfl3 & (side == "right"))
        # rows whose drive turns the drone LEFT / RIGHT (set by wire_cue)
        self.cue_left, self.cue_right = self.pfl3_r, self.pfl3_l
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
        is_opto = np.isin(ct, OPTO_TYPES)
        self.opto_l = np.flatnonzero(is_opto & (side == "left"))
        self.opto_r = np.flatnonzero(is_opto & (side == "right"))
        # rows excited by a LEFT / RIGHT self-rotation, and drive per deg/s:
        # set by the runner's wire_opto; 0 = reflex off
        self.opto_on_left, self.opto_on_right = self.opto_r, self.opto_l
        self.opto_per_deg = 0.0
        self._prev_prof = None
        self.flow = 0.0               # last measured rotation, deg/s
        is_esc = np.isin(ct, ESCAPE_TYPES)
        self.esc_l = np.flatnonzero(is_esc & (side == "left"))
        self.esc_r = np.flatnonzero(is_esc & (side == "right"))
        self.speed = 2.0          # current forward speed, for looming
        self.n_sub = N_SUB
        self._prev_val = None         # last frame's per-neuron drive

    def reset(self) -> None:
        self._prev_val = None
        self._prev_prof = None
        self.flow = 0.0

    def drive(self, world, position, heading_rad):
        # One compiled sweep returns the unlit scene, the per-azimuth range
        # and both centroids.
        r = self.cam.sense(world, position, heading_rad)
        beams_m = r["beams"]
        scene = r["unlit"].ravel()  # the lamp is NEVER in the steering image
        cen = self.cam.rangefinder.centres
        a_l, a_r = threat_level(beams_m, cen, self.speed)
        # THE CUE IS NOT THE VISION CHANNEL.  It carries the bearing of the
        # strongest bright blob only -- "where is the goal"; "what is in
        # the way" is left to the visual injection.  Feeding it the signed
        # centroid once handed the circuit a finished steering command
        # through a second route.
        scent, scent_mass = r["scent"], r["scent_mass"]
        frac = float(np.clip(scent / (self.cam.az_span / 2), -1.0, 1.0))
        s = (scent_mass / (scent_mass + CUE_HALF)
             if self.fixed_strength is None else self.fixed_strength)
        d = self._build(scene[self._px], frac, s, beams_m, cen)
        # optomotor: this frame's rotation, turned back through HS/H2
        prof = r["unlit"].reshape(self.cam.n_az, self.cam.n_el).mean(axis=1)
        self.flow = (yaw_flow(self._prev_prof, prof, self.cam.da)
                     if self._prev_prof is not None else 0.0)
        self._prev_prof = prof
        rows = self.opto_on_left if self.flow > 0 else self.opto_on_right
        d[rows, :] += self.opto_per_deg * abs(self.flow)
        return d, {"bearing": r["bearing"], "scent": scent,
                   "scent_mass": scent_mass,
                   "blocked": r["blocked"], "weight": r["weight"],
                   "rear": r["rear"], "threat_in": (a_l, a_r),
                   "beams": beams_m, "flow": self.flow}

    def _build(self, lum, frac, s, beams_m, cen):
        """The drive for one control cycle, one column per sub-step."""
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
        prev = self._prev_val if self._prev_val is not None else val
        self._prev_val = val
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
            gap = np.abs((self.thr_az[:, None] - cen[None, :] + 180.0)
                         % 360.0 - 180.0)
            nb = np.argmin(gap, axis=1)
            s_cell = looming(beams_m[nb], cen[nb], self.speed)
        else:
            s_cell = np.zeros(len(self.thr_rows))
        d[self.thr_rows] += torch.from_numpy(
            (THREAT_GAIN * self._thr_w * s_cell).astype(np.float32))[:, None]
        # goal bearing, push-pull into PFL3; equal total per side
        n = 0.5 * (len(self.cue_left) + len(self.cue_right))
        c = self.cue_gain * s * n
        d[self.cue_left, :] += c * (1 + frac) / max(len(self.cue_left), 1)
        d[self.cue_right, :] += c * (1 - frac) / max(len(self.cue_right), 1)
        return d


def demo() -> None:
    """mirror_weights: equal total per mirror band, whatever the counts."""
    az = np.array([5.0, 6.0, 7.0, -5.0, 25.0, -25.0, -26.0, 40.0, np.nan])
    w = mirror_weights(az)
    assert abs(w[:3].sum() - w[3]) < 1e-12            # 3 cells vs 1 cell
    assert abs(w[4] - w[5:7].sum()) < 1e-12           # 1 vs 2
    assert w[7] == 1.0 and w[8] == 1.0                # no mirror / no direction
    # yaw_flow: a profile shifted 5.4 columns reads as that rotation
    rng = np.random.default_rng(0)
    x = np.convolve(rng.normal(size=300), np.ones(5) / 5, "same")
    xs = np.interp(np.arange(256) + 5.4, np.arange(300), x)
    est = yaw_flow(x[:256], xs, 1.0, dt=0.1)
    assert abs(est - 54.0) < 3.0, est                 # +5.4 col = left turn
    assert abs(yaw_flow(xs, x[:256], 1.0, dt=0.1) + 54.0) < 3.0
    print("vp_input demo ok")


if __name__ == "__main__":
    demo()
