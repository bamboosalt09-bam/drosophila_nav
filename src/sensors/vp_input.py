"""Fisheye image straight into the visual projection neurons.

Same idea as `ConnectomeInput`, one stage later: each projection neuron reads
the pixel its receptive field points at, instead of each lamina cell reading
the pixel its ommatidium points at.  The optic lobe between them is not
simulated, because this rate model has no reason to reproduce what it
computes and measurement says it got the sign wrong when it tried.

The ORN cue is unchanged: the rectified whole-image centroid, which ignores
anything at or below sky level and is pulled only by what is brighter.

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

from sensors.fisheye import FisheyeCamera, clearance_ahead

# Time to contact at which a manoeuvre is fully urgent.  A fly starts its
# evasive turn about 100 ms before contact; a 2 m/s drone needs longer
# because its turn takes longer.
# 2.0 s: one turn (0.64 s at 90 deg/s through 90 deg) with a factor of three
# of margin.  A sweep over 1.0-5.0 preferred 1.0-2.0, but that was measured
# before the absolute standoff existed and before the per-azimuth range
# profile, so it is not evidence for this value any more -- the physical
# argument is.  Re-measure it when something depends on it.
TAU_CRIT = 2.0
# Stereo lived here -- BASELINE_M, MIN_DISPARITY_DEG, a `stereo` flag and a
# `tau_stereo` term -- with a paragraph of justification each.  None of it
# ever ran: the correlation window spans the frontal +-45 deg, more than
# half of which is sky at zero disparity, so the best match was always a
# shift of zero and the estimate returned inf at every distance.  Deleted
# rather than left looking live.  The lamp does this job.
#
# LAMP_R0 lived here too, duplicating `fisheye.LAMP_HALF_M`.  Two copies of
# one physical constant that nothing kept in step; the compiled renderer
# takes the one in `fisheye`.
# How fast the range estimate may change, in metres per second of estimate.
# A beacon slipping behind a pillar, or a wall appearing at a doorway, steps
# the brightness in one frame; without a limit the speed channel would slam.
# 12 m/s lets a real approach at 2 m/s through untouched while turning a
# step into a ramp of a few frames.
RANGE_SLEW = 12.0
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

# SPONTANEOUS ALTERNATION: turned one way long enough, turn the other way.
# The only memory is the recent net rotation -- a leaky integral of the yaw
# rate the gyro already measures -- so it needs no map and no extra sensor.
# A full loop (360 deg) accumulated in one direction triggers a push the
# opposite way for ALT_STEPS control cycles, through the same JO push-pull
# the gyro uses (measured: strong, correctly lateralised).  The circuit
# still weighs it against the goal and the threat.  Aimed at the failure in
# room 6: five beacons taken, the sixth visible behind a partition, and the
# last ~40 s spent circling in one corridor.
# The memory is the net rotation over the last ALT_WINDOW seconds -- a
# plain sum, not a leaky one.  A 10 s leaky integral saturates at
# (turn rate x 10 s), so any loop slower than 36 deg/s never reached 360 deg:
# in room 6 it fired 0 times in 6 flights while the drone circled a
# corridor for the last ~40 s of each.  A window catches a loop of any size
# that closes within it.
ALT_WINDOW = 20.0             # s
ALT_TRIGGER = 2.0 * math.pi   # rad of net turning that counts as a loop
# ...and only if that last full turn happened in a SMALL AREA.  Turning
# alone also counts a sweep around the whole room, which is navigation, not
# a loop: in room 6 the push fired 3 times on such sweeps with 6 and 24
# beams and broke routes that had been finding 6 of 6.  The corridor loops
# it is meant for were 5-6 m across.
ALT_RADIUS = 3.0              # m, around the centre of the last full turn
ALT_STEPS = 30                # control cycles of opposite push (3 s)
# The push goes into PFL3, the central-complex output that commands turns
# through DNa02 -- where a turn bias born of memory belongs.  It first went
# through the JO push-pull, measured when steering was read from all 1,304
# descending neurons; after steering moved to DNa01/02 nobody re-measured
# it, and JO turned out to reach DNa01/02 at 0.1-1 deg/s.  The push had
# silently gone dead (it fired 2-3 times per flight in room 6 and changed
# nothing).  PFL3 one-sided at 1/cell moves the same readout by ~3,000
# deg/s.  Which side turns which way, and how hard to drive it for
# ALT_TARGET_DEG, are MEASURED by the runner, not assumed.
ALT_TARGET_DEG = 90.0

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
        self._prev_val = None

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
        is_st = np.isin(ct, STEER_TYPES)
        self.steer_l = np.flatnonzero(is_st & (side == "left"))
        self.steer_r = np.flatnonzero(is_st & (side == "right"))
        is_esc = np.isin(ct, ESCAPE_TYPES)
        self.esc_l = np.flatnonzero(is_esc & (side == "left"))
        self.esc_r = np.flatnonzero(is_esc & (side == "right"))
        is_pfl3 = ct == "PFL3"
        self.pfl3_l = np.flatnonzero(is_pfl3 & (side == "left"))
        self.pfl3_r = np.flatnonzero(is_pfl3 & (side == "right"))
        self.alt_drive = 0.0          # set by calibration; 0 = no push
        self.alt_rows_left = self.pfl3_r    # rows that push LEFT (measured)
        self.alt_rows_right = self.pfl3_l
        is_jo = np.char.startswith(ct, "JO")
        self.jo_l = np.flatnonzero(is_jo & (side == "left"))
        self.jo_r = np.flatnonzero(is_jo & (side == "right"))
        self.jo_scale_l = 1.0 / max(len(self.jo_l), 1)
        self.jo_scale_r = 1.0 / max(len(self.jo_r), 1)
        self.gyro_base = GYRO_BASE
        self.yaw_rate = 0.0       # rad/s, set by the runner from the plant
        self._yaw_hist = []       # yaw * dt over the last ALT_WINDOW s
        self._pos_hist = []       # where the drone was at each of those
        self.alt_dir = 0.0        # -1/+1 while alternating
        self.alt_left = 0
        self.alt_count = 0
        self.dt = 0.1                 # control period, for the tau estimate
        self.speed = 2.0          # current forward speed, for tau
        self._goal_world = None
        self._hist = []               # recent linear-size samples
        self._tau = np.inf
        self._range = np.inf          # slew-limited lamp range
        self._profile = None          # range per azimuth, last frame
        self._clear_bearing = 0.0
        self.n_sub = N_SUB
        self._prev_val = None         # last frame's per-neuron drive

    def reset(self) -> None:
        self._pos_hist = []
        self._yaw_hist, self.alt_dir, self.alt_left, self.alt_count = \
            [], 0.0, 0, 0
        self._goal_world = None
        self._hist = []
        self._tau = np.inf
        self._range = np.inf
        self._prev_val = None

    def drive(self, world, position, heading_rad):
        # One compiled sweep returns the unlit scene, the lamp channel, the
        # per-azimuth range and both centroids.  In Python these were a
        # render plus three more passes over 36,864 pixels; together they
        # were 33 ms of the 43 ms control cycle.
        r = self.cam.sense(world, position, heading_rad)
        rear_m = r["rear"]
        beams_m = r["beams"]
        scene = r["unlit"]          # the lamp is NEVER in the steering image
        lum = scene.ravel()[self._px]
        # contrast, as the lamina transmits it: a uniform scene has no
        # direction in it and must not drive anything
        lum = lum - lum.mean()
        bearing, blocked, weight = r["bearing"], r["blocked"], r["weight"]

        # Replay prev -> cur across the sub-steps, so the drive is a SEQUENCE
        # rather than one column held for the whole cycle.  Holding it was
        # feeding the connectome a photograph: at tau 20 ms and rho 0.50 the
        # effective time constant is 40 ms, so 100 ms settles to 92% and
        # every frame started from a state washed clean.  A fly's visual
        # system is a motion system end to end; a still image run to steady
        # state gives it nothing to act on.
        #
        # Deliberately NOT a designed feature -- no looming detector, no
        # optic flow.  It is the time axis not being thrown away; the
        # differentiation is left to the circuit's own dynamics.  Costs one
        # frame of delay, which every real camera pipeline has, and no extra
        # render.  Verified: the standing sweep resets between headings so
        # the sequence is constant there and calibration is unchanged, while
        # the same pose reached by closing versus opening now reads
        # differently, which it could not before.
        val = lum * self.drive_gain * self._az_ok

        prev = self._prev_val if self._prev_val is not None else val
        self._prev_val = val
        ramp = np.linspace(1.0 / self.n_sub, 1.0, self.n_sub)
        seq = prev[:, None] * (1.0 - ramp) + val[:, None] * ramp

        d = torch.zeros(self.net.n, self.n_sub)
        d[self.vp_rows] = torch.from_numpy(seq.astype(np.float32))
        cen = self.cam.rangefinder.centres
        a_l, a_r = threat_level(beams_m, cen, self.speed)
        # Each cell takes the NEAREST beam, so every cell gets an input and
        # the same wall drives the circuit equally hard whatever the beam
        # count.  Reading only beams whose cone covered the cell's direction
        # left 176 of 311 cells with nothing at 12 beams (30 deg spacing,
        # 15 deg cones) and none at 24, so the same wall arrived about twice
        # as strong at 24: threat read 0.25-0.29 against 0.12-0.16 at the
        # same moment in room 6, and the two flights split 3.4 s in.  Fewer
        # beams now means coarser angular resolution, not weaker threat.
        if len(cen):
            gap = np.abs((self.thr_az[:, None] - cen[None, :] + 180.0)
                         % 360.0 - 180.0)
            nb = np.argmin(gap, axis=1)
            s_cell = looming(beams_m[nb], cen[nb], self.speed)
        else:
            s_cell = np.zeros(len(self.thr_rows))
        d[self.thr_rows] += torch.from_numpy(
            (THREAT_GAIN * s_cell).astype(np.float32))[:, None]

        # TIME TO CONTACT, two independent estimates.
        #
        # The connectome is tuned to a fly: 0.3 m/s past a 5 cm obstacle is
        # 344 deg/s of image motion, this drone at 2 m/s past a 2 m obstacle
        # is 57 deg/s.  Raw distance means nothing across that gap, but
        # tau = distance / closing speed does: 2 m at 2 m/s and 30 cm at
        # 0.3 m/s are both one second away and equally urgent.
        #
        # (a) EXPANSION.  Works at any range, needs motion.  The area-like
        #     `blocked` grows as 1/d^2, so its logarithmic derivative is
        #     twice what it should be -- measured tau came out at half the
        #     true value, exactly.  sqrt(blocked) is a LINEAR size, grows as
        #     1/d, and needs no correction.  Differencing over several frames
        #     rather than one removes the 23% of frames where the change was
        #     below pixel noise.
        # (b) STEREO.  Works standing still, near field only.  0.25 m of
        #     baseline resolves ~3 px at 2 m and nothing past 5 m.
        #
        # They are complementary, so the nearer (more urgent) one wins.
        lin = math.sqrt(max(blocked, 0.0))
        self._hist.append(lin)
        if len(self._hist) > 4:
            self._hist.pop(0)
        tau_flow = np.inf
        if len(self._hist) >= 3 and lin > 1e-3:
            span = (len(self._hist) - 1) * self.dt
            rate = (self._hist[-1] - self._hist[0]) / span
            if rate > 1e-5:
                tau_flow = lin / rate

        # LAMP RANGE.  What the lamp adds above the unlit level falls as
        # 1/(1+(r/r0)^2), so the brightest frontal pixels give range directly
        # -- no correspondence, no motion, and free because the render has
        # already computed it.  Measured contribution: 0.151 at 2 m, 0.029 at
        # 5 m, 0.002 at 12 m.
        #
        # Slew-limited, because brightness steps when a beacon slips behind a
        # pillar or a doorway opens, and an unfiltered step would slam the
        # speed channel.  The limit is loose enough that a real 2 m/s
        # approach passes through untouched.
        rng = self._clearance(r["range"])
        if np.isfinite(rng):
            if np.isfinite(self._range):
                lim = RANGE_SLEW * self.dt
                rng = float(np.clip(rng, self._range - lim, self._range + lim))
            self._range = rng
        elif np.isfinite(self._range):
            self._range = min(self._range + RANGE_SLEW * self.dt, np.inf)
            if self._range > 30.0:
                self._range = np.inf
        tau_lamp = (self._range / max(self.speed, 1e-3)
                    if np.isfinite(self._range) else np.inf)

        # The lamp inverts range to within 1-2% wherever it reaches, so it
        # wins outright there; expansion rate is the fallback beyond it and
        # is only trustworthy in the far field, where it saturates least.
        # (A real build would use a rangefinder here -- ultrasonic or ToF --
        # and the lamp stands in for one.)
        self._tau = tau_lamp if np.isfinite(tau_lamp) else tau_flow
        # NOT clipped at 1: above 1 it says "closing faster than the safe
        # time allows, by this factor", which is exactly what the speed law
        # needs to divide by.  Clipping it there capped the braking at the
        # moment braking mattered most.
        urgency = (float(TAU_CRIT / max(self._tau, 1e-3))
                   if np.isfinite(self._tau) else 0.0)
        # THE ODOUR CHANNEL IS NOT THE VISION CHANNEL.
        #
        # `bearing` above is the signed centroid: walls push, beacons pull,
        # obstacles folded in.  Feeding that to ORN handed the circuit a
        # finished steering command through a second modality, so vision had
        # nothing left to contribute and the readout could only ever track
        # the baseline it was being fed.
        #
        # ORN now gets the rectified centroid, which is what this module's
        # docstring has always said it gets: bright things only, walls
        # contributing exactly zero.  It answers "where is the goal", and
        # "what is in the way" is left to the visual injection, where it
        # belongs.  Neither needs a target's world position.
        scent, scent_mass = r["scent"], r["scent_mass"]
        self._goal_world = heading_rad + np.radians(scent)
        frac = float(np.clip(scent / (self.cam.az_span / 2), -1.0, 1.0))
        s = (scent_mass if self.fixed_strength is None
             else self.fixed_strength)
        base = self.orn_gain * s
        od_l, od_r = base * (1 + frac), base * (1 - frac)
        # odour does not flicker within a control cycle
        d[self.orn_l, :] = od_l * self.orn_scale_l
        d[self.orn_r, :] = od_r * self.orn_scale_r
        # gyro: opposing push-pull on JO; + yaw is a LEFT turn
        g = -self.yaw_rate / GYRO_FULL
        self._yaw_hist.append(self.yaw_rate * self.dt)
        self._pos_hist.append((float(position[0]), float(position[1])))
        if len(self._yaw_hist) > int(round(ALT_WINDOW / self.dt)):
            self._yaw_hist.pop(0)
            self._pos_hist.pop(0)
        if self.alt_left == 0:
            # walk back to where the last full turn began
            acc, start = 0.0, None
            for i in range(len(self._yaw_hist) - 1, -1, -1):
                acc += self._yaw_hist[i]
                if abs(acc) > ALT_TRIGGER:
                    start = i
                    break
            if start is not None:
                pts = np.asarray(self._pos_hist[start:])
                spread = np.linalg.norm(pts - pts.mean(axis=0), axis=1).max()
                if spread < ALT_RADIUS:
                    # + acc is a LEFT loop; the push goes the other way
                    self.alt_dir = -math.copysign(1.0, acc)
                    self.alt_left = ALT_STEPS
                    self.alt_count += 1
                    self._yaw_hist, self._pos_hist = [], []
        g = float(np.clip(g, -1.0, 1.0))
        d[self.jo_l, :] = self.gyro_base * (1 + g) * self.jo_scale_l
        d[self.jo_r, :] = self.gyro_base * (1 - g) * self.jo_scale_r
        if self.alt_left > 0:
            rows = (self.alt_rows_left if self.alt_dir > 0
                    else self.alt_rows_right)
            d[rows, :] += self.alt_drive
            self.alt_left -= 1
        return d, {"bearing": bearing, "scent": scent,
                   "scent_mass": scent_mass,
                   "blocked": blocked, "weight": weight,
                   "tau": self._tau, "urgency": urgency,
                   "tau_flow": tau_flow, "tau_lamp": tau_lamp,
                   "range": self._range, "profile": self._profile,
                   "clear_bearing": self._clear_bearing,
                   "rear": rear_m, "threat_in": (a_l, a_r),
                   "beams": beams_m, "alternating": self.alt_left > 0}

    def _clearance(self, profile):
        """Nearest range anywhere in a wide forward cone.

        Was a single percentile over the frontal +-35 deg, which reported a
        flat 4.00 m for 25 steps while a wall closed from 3.55 m to 0.26 m at
        +24..+62 deg: the wall drifted out of the window and the percentile
        inside it was dominated by distant surfaces.  The cone is wide
        because a turning body sweeps sideways -- the collision in room 0
        happened at +62 deg.
        """
        self._profile = profile
        rng, brg = clearance_ahead(self.cam, profile, 0.0, 60.0)
        self._clear_bearing = brg
        return rng
