"""The drone layer: follow the path the fly intends, within what a drone can do.

The user's design: "the fly proposes the expected path, and the drone
MODIFIES that path within realistic conditions and follows it" -- because a
fly's sharp turns and its wall-risking manoeuvres are not a drone's.
Whatever proposes the path -- the connectome, a planner, a centroid
follower -- hands over the turn it INTENDS (unclipped; a fly can turn at
1,000 deg/s).  The drone then picks, every cycle, the flyable curve closest
to that intent (see `follow`):

  * a turn too tight for the airframe becomes the tightest curve it can
    fly at a useful speed, not a crawl;
  * a curve that runs into something within SWERVE_D is bent away early,
    by as little as clears it -- a swerve, not a stop;
  * only when no curve moves at all does it turn on the spot.  It never
    reverses: that stopped-and-backing drone was the drone layer refusing
    to modify the path, which is its job.

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
V_TURN = 0.3                  # m/s: slower than this is not moving
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


# The curves the drone may choose from: curvature k (1/m, + = left) from
# straight to the tightest it can fly while still moving (R_MAX / V_TURN,
# a 0.19 m radius).
K_GRID = np.linspace(-R_MAX / V_TURN, R_MAX / V_TURN, 81)
# A curve whose way is not clear for SWERVE_D metres costs up to SWERVE_W
# (in the units of curvature, 1/m).  The penalty must outgrow the curvature
# a swerve needs: at 2.0 a wall 1.5 m ahead cost straight-on 1.0 while the
# clearing curve (k 1.0) cost 1.3, and the closer the wall the worse that
# trade got, so the drone flew on until it had to stop.  At 4.0 the swerve
# starts ~2.5 m out with k ~0.3.  3 m is 1.5 s at cruise: flies turn away from a looming surface
# well before contact, and so should the drone.
SWERVE_D = 3.0
SWERVE_W = 4.0
# Changing the chosen curve from one cycle to the next also costs, or a
# wall dead ahead -- equally good to pass left or right -- flips the choice
# every cycle.
HOLD_W = 0.3


def speed_on(pts, k):
    """Fastest speed on curve k that can still stop before the first contact
    and stays within the yaw-rate limit."""
    d = hit_distance(pts, k)
    v = min(V_CRUISE, math.sqrt(2.0 * A_BRAKE * max(d - D_STOP, 0.0)))
    return min(v, R_MAX / abs(k)) if abs(k) > 1e-9 else v


def follow(u_intent, pts, mem=None, pose=None):
    """-> (yaw-rate command, speed command); + yaw is a LEFT turn.

    Among the curves the drone can fly while moving, take the one that
    minimises
        |k - k_fly|                          (stay on the fly's path)
      + SWERVE_W * blocked share of SWERVE_D  (do not fly at walls)
      + HOLD_W * |k - k_last|                 (do not dither)
    and fly it at the fastest speed that can still stop in time.  `pts`
    are the sensed echoes (see `beam_points`), plus the last 2 s of them
    when `mem` and `pose` are given.

    Nothing movable: turn on the spot, one way, held until a curve opens --
    the committed saccade that ended 58 s dithering stops in corners (room
    5, 3 beams: the intent flipped sign 267 times while stopped).
    """
    if mem is not None and pose is not None:
        pts = _remember(pts, mem, pose)
    k_fly = u_intent / V_CRUISE
    k_last = mem.get("k", k_fly) if mem is not None else k_fly
    best = None
    for k in np.append(K_GRID, k_fly):
        v = speed_on(pts, k)
        if v < V_TURN:
            continue
        blocked = max(0.0, 1.0 - hit_distance(pts, k) / SWERVE_D)
        cost = abs(k - k_fly) + SWERVE_W * blocked + HOLD_W * abs(k - k_last)
        if best is None or cost < best[0]:
            best = (cost, float(k), v)
    if best is not None:
        _, k, v = best
        if mem is not None:
            mem["k"] = k
            mem.pop("dir", None)
        return float(np.clip(k * v, -R_MAX, R_MAX)), v
    side = 1.0 if k_fly >= 0 else -1.0
    if mem is not None:
        side = mem.setdefault("dir", side)
    return side * R_MAX, 0.0


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
    # open space: exactly the fly's path, full speed; a side wall changes
    # nothing
    assert follow(0.3, none) == (0.3, V_CRUISE)
    assert follow(0.0, np.array([[1.5, 1.0]])) == (0.0, V_CRUISE)
    # a wall across the path 1.5 m ahead: swerve, keep moving, no reverse
    wall = np.array([[1.5, y] for y in np.arange(-0.8, 0.81, 0.1)])
    r, v = follow(0.0, wall)
    assert r != 0.0 and v >= V_TURN
    # ...and the swerve clears it: the chosen curve does not hit the wall
    assert hit_distance(wall, r / v) > 1.5
    # a turn far too tight (1,000 deg/s): the tightest movable curve instead
    r, v = follow(math.radians(1000.0), none)
    assert abs(r - R_MAX) < 1e-6 and v >= V_TURN
    # boxed in: turn on the spot, never backwards, one way through a flip
    box = np.array([[0.3 * math.cos(a), 0.3 * math.sin(a)]
                    for a in np.linspace(0, 2 * math.pi, 36)])
    mem = {}
    r1, v1 = follow(+0.1, box, mem)
    r2, v2 = follow(-2.0, box, mem)
    assert v1 == v2 == 0.0 and r1 == r2 == R_MAX
    assert follow(-2.0, none, mem)[0] < 0 and "dir" not in mem   # released
    # a wall seen once is still there when it has left the beam
    mem = {}
    follow(0.0, np.array([[1.0, 0.0]]), mem, pose=(0.0, 0.0, 0.0))
    assert follow(0.0, none, mem, pose=(0.0, 0.0, 0.0))[0] != 0.0
    print("drone_layer demo ok")


if __name__ == "__main__":
    demo()
