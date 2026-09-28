"""An omnidirectional ring of range sensors, because a drone cannot touch the
wall to find it.

WHY A SENSOR AND NOT VISION.  A fly gets distance from motion: T4/T5 in the
optic lobe compute it, LPLC2 turns it into looming.  That pathway is in the
connectome -- T4 1,410 cells, T5 6,703, LPLC 415, all of it -- but in this
rate model it would not compute distance.  Measured on held-out rooms, the
descending common mode against 1/tau:

    no optic lobe                                   -0.15
    optic lobe, types split, gains normalised       -0.13 .. +0.12
    trained 250 steps, 30 ms gradient window        +0.03
    trained 250 steps, 200 ms window, braking only  +0.03
    this sensor's range injected                    +0.77

And the stakes differ.  A fly that misjudges a wall lands on it.  A drone
that misjudges a wall is finished.  So distance comes from a range sensor,
the way a real build would get it, and every result says so.

WHY 360 DEG.  Front-only coverage let the drone stop 0.8 m from a wall and
then sit there: the stop was correct, but with the goal on the far side of
the wall the steering kept aiming back into it, and with no view behind,
reversing out would have been blind.  The centroid arm spent 82% of a
600-step flight at zero speed that way.  A ring with a rear sector makes
backing off a checked manoeuvre instead of a blind one.

THE SPEC is a ring of ultrasonic sensors of the HC-SR04 class: 24 beams,
one every 15 deg, each a 15 deg cone, 0.02-4.0 m, returning the NEAREST echo
in the cone.  Gaussian noise of 2 cm, seeded, so runs repeat.  Each beam is
cast analytically against the obstacles at flight altitude -- 24 beams x 5
rays -- rather than read off the camera, because the camera does not see
behind.  In front the camera's own solid-depth range agrees with it and
served as the cross-check.

Beacons are not echoed: the collision test ignores them, so they are markers
rather than objects, and a sensor that saw them would park the drone at the
0.8 m standoff short of every beacon.
"""
from __future__ import annotations

import math

import numpy as np

REAR_HALF_DEG = 37.5        # the rear sector: beams at 150, 165, 180 and mirror

# Beam layouts for the sensor-count sweep, same 15 deg sensor in each.
LAYOUTS = {0: (), 1: (0,), 3: (-45, 0, 45),
           6: tuple(range(-120, 181, 60)), 12: tuple(range(-150, 181, 30)),
           24: tuple(range(-180, 180, 15))}


class Rangefinder:
    def __init__(self, cam_az_deg: np.ndarray,
                 centres_deg=tuple(range(-180, 180, 15)),
                 half_deg: float = 7.5, rays: int = 5, max_m: float = 4.0,
                 min_m: float = 0.02, noise_m: float = 0.02, seed: int = 0):
        self.centres = np.asarray(centres_deg, dtype=float)
        self.half_deg, self.max_m = half_deg, max_m
        self.min_m, self.noise_m = min_m, noise_m
        self._offs = np.linspace(-half_deg, half_deg, rays)
        self._rs = np.random.RandomState(seed)
        a = np.asarray(cam_az_deg, dtype=float)
        # Only beams whose WHOLE cone lies inside the camera's field map onto
        # its columns.  The +-150 beams span 142.5-157.5 deg; mapping them to
        # the edge columns (142.5-145) let a wall behind the drone at 155 deg
        # appear as a frontal range.
        self._beam_of_col = np.full(len(a), -1)
        if len(self.centres):
            inside = np.abs(self.centres) + half_deg <= np.abs(a).max() + 1e-9
            d = np.abs(a[:, None] - self.centres[None, :])
            d[:, ~inside] = np.inf
            near = np.argmin(d, axis=1)
            self._beam_of_col = np.where(
                d[np.arange(len(a)), near] <= half_deg, near, -1)
        self._rear = (np.abs(self.centres) >= 180.0 - REAR_HALF_DEG
                      + half_deg - 1e-9)

    def scan(self, world, p, heading_rad: float) -> np.ndarray:
        """One reading per beam, metres; inf where there is no echo."""
        out = np.full(len(self.centres), np.inf)
        if not len(self.centres):
            return out
        circ = []
        for o in world.obstacles:
            if o.height is None:                         # sphere
                dz = p[2] - o.z
                if abs(dz) < o.radius:
                    circ.append((o.x, o.y, math.sqrt(o.radius ** 2 - dz * dz)))
            elif o.z <= p[2] <= o.z + o.height:          # cylinder
                circ.append((o.x, o.y, o.radius))
        if not circ:
            return out
        c = np.asarray(circ)
        ang = heading_rad + np.radians(self.centres[:, None]
                                       + self._offs[None, :])      # (B, K)
        dx, dy = np.cos(ang).ravel()[:, None], np.sin(ang).ravel()[:, None]
        ox, oy = p[0] - c[:, 0][None, :], p[1] - c[:, 1][None, :]  # (1, N)
        b = 2.0 * (dx * ox + dy * oy)
        cc = ox * ox + oy * oy - (c[:, 2] ** 2)[None, :]
        disc = b * b - 4.0 * cc
        t = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / 2, np.inf)
        t = np.where(t > 0, t, np.inf)
        t = np.where(cc < 0, 0.0, t)                     # inside an obstacle
        beam = t.min(axis=1).reshape(len(self.centres), -1).min(axis=1)
        for i, m in enumerate(beam):
            if not np.isfinite(m):
                continue
            m = max(m + self.noise_m * self._rs.randn(), self.min_m)
            if m <= self.max_m:
                out[i] = m
        return out

    def profile(self, beams: np.ndarray) -> np.ndarray:
        """Readings spread over the camera columns, the old `range` shape.

        Every consumer -- the clearance gate, the speed law, the injection
        into the circuit -- takes the sensor without being rewritten.
        """
        prof = np.full(len(self._beam_of_col), np.inf)
        ok = self._beam_of_col >= 0
        prof[ok] = beams[self._beam_of_col[ok]]
        return prof

    def rear(self, beams: np.ndarray) -> float:
        """Nearest echo in the rear sector, i.e. what reversing would hit.

        No rear beam means the rear is UNKNOWN, which reads as blocked:
        backing up blind is exactly what the ring exists to prevent.
        """
        if not self._rear.any():
            return 0.0
        return float(np.min(beams[self._rear]))


def demo() -> None:
    from environment.target_world import Obstacle, TargetWorld
    a = np.linspace(-145, 145, 256)
    rf = Rangefinder(a, noise_m=0.0)
    far = np.array([300.0, 0.0, 2.0])
    p = np.array([0.0, 0.0, 2.0])
    w = TargetWorld(target=far, obstacles=[
        Obstacle(x=-3.5, y=0.0, radius=1.5, z=0.0, height=8.0)])  # behind
    beams = rf.scan(w, p, 0.0)
    assert abs(rf.rear(beams) - 2.0) < 1e-6, rf.rear(beams)
    assert np.isinf(rf.profile(beams)).all(), "rear wall leaked forward"
    w = TargetWorld(target=far, obstacles=[
        Obstacle(x=3.5, y=0.0, radius=1.5, z=0.0, height=8.0)])   # ahead
    beams = rf.scan(w, p, 0.0)
    assert abs(beams[rf.centres == 0.0][0] - 2.0) < 1e-6
    assert np.isinf(rf.rear(beams))
    print("rangefinder: %d beams, 360 deg, max %.1f m, rear sector %d beams "
          "-- demo ok" % (len(rf.centres), rf.max_m, int(rf._rear.sum())))


if __name__ == "__main__":
    demo()
