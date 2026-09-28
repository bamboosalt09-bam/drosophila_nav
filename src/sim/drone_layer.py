"""The drone layer: follow the path the fly intends, within what a drone can do.

The fly proposes, the drone disposes.  Whatever proposes the path -- the
connectome, a planner, a centroid follower -- hands over two things: the turn
it INTENDS (unclipped; a fly can turn at 1,000 deg/s) and a threat level on
each side.  The drone turns that into something it can fly:

  * the intended turn is held as a CURVATURE, k = u / V.  A turn too tight
    for 90 deg/s is not clipped to 90 deg/s and swung wide -- the drone
    slows down until the same curve fits, so it traces the fly's path
    instead of a fatter one;
  * the path is bent away from the side with the greater threat, and speed
    drops with the nearest threat;
  * a hard envelope underneath: blocked closer than D_STOP means back off if
    the rear is clear, else stop.

Every arm goes through this same function, so arms differ only in the path
they propose -- which is the comparison the project is about.
"""
from __future__ import annotations

import math

import numpy as np

from sim.episode import V_CRUISE

R_MAX = math.radians(90.0)    # yaw-rate limit of the airframe
# m: nothing allowed nearer than this.  Body radius 0.25 m (the collision
# test's clearance) plus 0.2 m.  It was 0.8, chosen when the drone could
# still be doing 2 m/s at a wall; the speed law now floors speed at 0.1 m/s
# by 1.5 m ahead, so stopping takes millimetres, and 0.8 m over a +-60 deg
# cone made door jambs read as "blocked" and turned doorways into dead ends.
D_STOP = 0.45
V_BACK = -0.4                 # m/s: backing off, only with a clear rear
V_TURN = 0.3                  # m/s: turn rate while stopped is k * this
# SPEED COMES FROM A PREDICTED COLLISION, NOT FROM PROXIMITY.
# Fly as fast as the airframe allows unless the path being flown is predicted
# to hit something.  Only sensed points within the body's swept width of the
# INTENDED curve count, so a wall alongside does not slow the drone at all.
# The previous law braked on (1.5/r)^2 of the nearest echo in a +-60 deg
# cone: side walls at 50 deg counted as "ahead", anything within 3 m cost
# 25% of speed whether on the path or not, and the more beams the drone had
# the slower it flew -- 44-61 m in 120 s with 12-24 beams in room 6.
BODY_R = 0.25                 # m, the collision test's clearance
SWEPT = BODY_R + 0.15         # m, half-width of the corridor the path sweeps
A_BRAKE = 2.0                 # m/s^2, half the airframe's 4, for latency
HORIZON = 4.0                 # m, the sensor range
BEAM_HALF = 7.5               # deg, half the ultrasonic cone
# Echoes are remembered for this many control cycles (2 s), in world
# coordinates from the drone's own pose estimate, and join the prediction.
# With one front beam the planner stopped at a wall, turned in place about
# 144 deg until the front was clear -- which put the same wall at +82 deg,
# outside the beam -- then was cleared to swing left at full speed, "unknown
# = free", and hit the wall it had been looking at 2 s earlier.
MEMORY_STEPS = 20


def beam_points(beams, centres_deg, half_deg=BEAM_HALF):
    """Sensed echoes as body-frame points (+x ahead, +y left), cone edges
    included, so a reading anywhere in the cone is taken seriously."""
    ok = np.isfinite(beams)
    if not ok.any():
        return np.zeros((0, 2))
    r = np.tile(beams[ok], 3)
    c = np.asarray(centres_deg, dtype=float)[ok]
    ang = np.radians(np.concatenate([c - half_deg, c, c + half_deg]))
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)


def hit_distance(pts, k, width=SWEPT, horizon=HORIZON):
    """Path length along curvature `k` to the first point within `width`.

    The body starts at the origin heading +x; + k turns LEFT.  inf if the
    swept corridor is clear out to `horizon`.
    """
    if not len(pts):
        return np.inf
    x, y = pts[:, 0], pts[:, 1]
    if abs(k) < 1e-6:
        m = (x > 0) & (np.abs(y) < width) & (x <= horizon)
        return float(x[m].min()) if m.any() else np.inf
    rc = 1.0 / k                                   # centre at (0, rc)
    off = np.abs(np.hypot(x, y - rc) - abs(rc))    # distance to the arc
    phi = np.arctan2(y - rc, x)
    phi0 = np.arctan2(-rc, 0.0)
    th = np.mod((phi - phi0) * np.sign(k), 2 * np.pi)
    s = th / abs(k)
    m = (off < width) & (s <= min(horizon, np.pi / abs(k)))
    return float(s[m].min()) if m.any() else np.inf
# A committed turn is released once the front is this far past D_STOP, so it
# does not chatter on the threshold.
SACCADE_CLEAR = 0.3


def follow(u_intent, pts, rear, mem=None, pose=None):
    """-> (yaw-rate command, speed command).

    The drone does NOT bend the path.  It used to, by the threat imbalance,
    on top of the fly's own avoidance through LC4 -> DNa01/02: avoidance
    counted twice, and the drone was choosing direction.  Direction belongs
    to whatever proposes the path.  `pts` are the sensed echoes (see
    `beam_points`); speed
    is the fastest that can still stop before the first predicted contact
    on the intended curve.  + yaw is a LEFT turn.

    BLOCKED AHEAD -> A COMMITTED TURN (needs `mem`, a dict kept per flight).
    Measured in room 5 with 3 beams: stopped 838 of 1200 steps; the
    intended turn flipped sign 267 times while stopped, and the longest stop
    lasted 58 s with a net heading change of 0 deg -- a corner faced
    head-on gives balanced threat and a weak, dithering intent.  So the
    direction is taken ONCE, from the curvature at the moment of stopping
    (the fly's intent plus the threat bend), and held at full rate until the
    front clears.  Flies change direction the same way, with ballistic body
    saccades.  The fly still picks the direction; the drone finishes it.
    """
    if mem is not None and pose is not None:
        pts = _remember(pts, mem, pose)
    k = u_intent / V_CRUISE
    d_path = hit_distance(pts, k)
    if mem is not None:
        # committed turn: triggered by the intended path, released once the
        # way straight ahead is clear
        if d_path < D_STOP or ("dir" in mem and hit_distance(pts, 0.0)
                               < D_STOP + SACCADE_CLEAR):
            mem.setdefault("dir", 1.0 if k >= 0 else -1.0)
            return mem["dir"] * R_MAX, (V_BACK if rear > D_STOP else 0.0)
        mem.pop("dir", None)
    v = min(V_CRUISE,
            math.sqrt(2.0 * A_BRAKE * max(d_path - D_STOP, 0.0)))
    if abs(k) > 1e-9:
        v = min(v, R_MAX / abs(k))
    if d_path < D_STOP:
        v = V_BACK if rear > D_STOP else 0.0
    # At a crawl the curve is meaningless -- turn at the rate the fly
    # intends.  It was k * V_TURN, i.e. the intent times 0.3/2.0: while
    # stopped the drone turned at 22% of what the fly asked for.
    if abs(v) < V_TURN:
        r = float(np.clip(k * V_CRUISE, -R_MAX, R_MAX))
    else:
        r = float(np.clip(k * v, -R_MAX, R_MAX))
    return r, v


def _remember(pts, mem, pose):
    """Current echoes plus those of the last MEMORY_STEPS cycles, body frame."""
    x0, y0, hd = pose
    c, s = math.cos(hd), math.sin(hd)
    world = np.stack([x0 + c * pts[:, 0] - s * pts[:, 1],
                      y0 + s * pts[:, 0] + c * pts[:, 1]], axis=1)
    hist = mem.setdefault("seen", [])
    hist.append(world)
    if len(hist) > MEMORY_STEPS:
        hist.pop(0)
    allw = np.concatenate(hist)
    dx, dy = allw[:, 0] - x0, allw[:, 1] - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


def demo() -> None:
    none = np.zeros((0, 2))
    # the swept corridor: a wall alongside is not on the path
    assert hit_distance(np.array([[2.0, 0.1]]), 0.0) == 2.0
    assert np.isinf(hit_distance(np.array([[2.0, 1.0]]), 0.0))
    # a point on a left arc of radius 2, a quarter turn along
    assert abs(hit_distance(np.array([[2.0, 2.0]]), 0.5) - np.pi) < 1e-9
    # full speed past a side wall, slowed by one on the path
    assert follow(0.0, np.array([[1.5, 1.0]]), np.inf)[1] == V_CRUISE
    v1 = follow(0.0, np.array([[1.0, 0.0]]), np.inf)[1]
    assert abs(v1 - math.sqrt(2 * A_BRAKE * (1.0 - D_STOP))) < 1e-9  # stops in time
    # a turn too tight for the airframe: same curve, lower speed
    r, v = follow(math.radians(360.0), none, np.inf)
    assert abs(r - R_MAX) < 1e-9 and v < V_CRUISE
    assert abs(r / v - math.radians(360.0) / V_CRUISE) < 1e-9, "curve changed"
    # blocked ahead: back off only if the rear is clear
    wall = np.array([[0.3, 0.0]])
    assert follow(0.0, wall, np.inf)[1] == V_BACK
    assert follow(0.0, wall, 0.3)[1] == 0.0
    # blocked: the direction is chosen once and held through a flipping intent
    mem = {}
    r1, _ = follow(+0.1, wall, 0.0, mem)
    r2, v2 = follow(-2.0, np.array([[0.6, 0.0]]), 0.0, mem)
    assert r1 == R_MAX and r2 == R_MAX and v2 == 0.0   # held through the flip
    r3, _ = follow(-2.0, none, 0.0, mem)     # clear: released
    assert r3 < 0 and "dir" not in mem
    # a wall seen once is still there when it has left the beam
    mem = {}
    follow(0.0, np.array([[1.0, 0.0]]), np.inf, mem, pose=(0.0, 0.0, 0.0))
    assert follow(0.0, none, np.inf, mem, pose=(0.0, 0.0, 0.0))[1] < V_CRUISE
    print("drone_layer demo ok")


if __name__ == "__main__":
    demo()
