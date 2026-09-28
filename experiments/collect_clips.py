"""Training clips: what the eye saw, and what was actually true.

Two targets, both measured off the world's geometry, neither one any
controller's output:

  bearing   where the nearest beacon really is, relative to the heading.
            NOT `contrast_centroid`'s estimate of it.  The two agree while
            the beacon is in view and diverge exactly when it is occluded,
            which is the case the circuit has to get right.  Fitting the
            readout to the centroid is what `calibrate` used to do, and once
            the rangefinder channel made the circuit respond to obstacles
            that fit read the disagreement as a sign error and inverted the
            gain: avoidance came out as attraction and the arm found 0 of 5.

  inv_tau   v / (true distance to the nearest frontal surface), by analytic
            ray-cylinder intersection, not by inverting lamp brightness.
            The lamp estimate is the thing we would like to stop needing.

AVOIDANCE IS NOT IN EITHER TARGET, deliberately.  Steering toward a goal and
braking for proximity are sensory quantities a fly's circuitry is known to
carry; turning away from an obstacle is a control decision, and writing one
into the loss would make the trained circuit a copy of whatever wrote it.
So avoidance is TESTED, not trained: if it appears in flight it came out of
the circuit combining the two channels, and if it does not, that is the
result that says supervised sensory training is not enough.

What gets stored is `lum` -- the 6,199 per-ommatidium luminances -- and not
the (68045, 20) drive tensor it expands into.  The expansion is a ramp
between consecutive frames and costs nothing to redo, while storing it would
be 5.4 MB a frame instead of 25 kB.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from environment.room_file import ensure as room
from environment.target_world import Agent, DRONE_YAW
from sensors.eye_input import EyeInput
from sim.episode import DT, V_CRUISE

R_MAX = math.radians(90.0)
FRAMES = 6                 # per clip; long enough for the ramp to carry motion
OUT = REPO / "results" / "clips"


def true_range(world, p, heading, half_deg=30.0, n_rays=9):
    """Distance to the nearest frontal surface, analytically.

    Ray-cylinder in closed form over every obstacle at once.  This is ground
    truth: no lamp, no percentile, no camera.
    """
    if not world.obstacles:
        return np.inf
    o = np.array([[b.x, b.y, b.radius,
                   1e9 if b.height is None else b.height, b.z]
                  for b in world.obstacles])
    bear = np.radians(np.linspace(-half_deg, half_deg, n_rays)) + heading
    dx, dy = np.cos(bear)[:, None], np.sin(bear)[:, None]     # (R, 1)
    ox, oy = p[0] - o[:, 0][None, :], p[1] - o[:, 1][None, :]  # (1, N)
    a = 1.0
    b = 2.0 * (dx * ox + dy * oy)
    c = ox * ox + oy * oy - (o[:, 2] ** 2)[None, :]
    disc = b * b - 4 * a * c
    ok = disc >= 0
    sq = np.sqrt(np.maximum(disc, 0.0))
    t = np.where(ok, (-b - sq) / 2.0, np.inf)
    t = np.where(ok & (t > 0), t, np.inf)
    # the ray is horizontal, so it only hits a cylinder spanning this height
    zlo, zhi = o[:, 4][None, :], (o[:, 4] + o[:, 3])[None, :]
    t = np.where((p[2] >= zlo) & (p[2] <= zhi), t, np.inf)
    return float(np.min(t))


def goal_bearing(world, p, heading):
    """Bearing to the NEAREST beacon, degrees, + is left.  Ground truth."""
    best, bb = np.inf, 0.0
    for t in world.all_targets():
        d = math.hypot(t[0] - p[0], t[1] - p[1])
        if d < best:
            best = d
            bb = math.degrees((math.atan2(t[1] - p[1], t[0] - p[0])
                               - heading + math.pi) % (2 * math.pi) - math.pi)
    return bb


def collect(seeds, n_clips, seed0=0):
    net_free = None
    lums, scents, masses, bearings, invtaus = [], [], [], [], []
    rs = np.random.RandomState(seed0)
    t0 = time.perf_counter()
    per = max(n_clips // len(seeds), 1)
    for si in seeds:
        w, _, _ = room(si)
        # EyeInput needs a net only for its shape; build the binding alone
        if net_free is None:
            from core.eye_subnet import build_eye_subnet
            net, ann, sub, info = build_eye_subnet()
            net_free = EyeInput(net, ann, info)
        inp = net_free
        for c in range(per):
            while True:
                p = np.array([rs.uniform(-15, 15), rs.uniform(-15, 15), 2.0])
                if not w.collides(p, clearance=1.2):
                    break
            ag = Agent(world=w, start=tuple(p),
                       heading=float(rs.uniform(-np.pi, np.pi)))
            u = float(rs.uniform(-R_MAX, R_MAX))
            cl, cs, cm, cb, ct = [], [], [], [], []
            for f in range(FRAMES):
                r = inp.cam.sense(ag.world, ag.p, ag.heading)
                flat = r["unlit"].ravel()
                lum = flat[inp._patch].mean(axis=1)[inp._uinv]
                cl.append((lum - lum.mean()).astype(np.float32))
                cs.append(r["scent"])
                cm.append(r["scent_mass"])
                cb.append(goal_bearing(ag.world, ag.p, ag.heading))
                rng = true_range(ag.world, ag.p, ag.heading)
                ct.append(V_CRUISE / max(rng, 0.3) if np.isfinite(rng) else 0.0)
                ag.step(u, V_CRUISE, 0.0, DT)
                if ag.collided:
                    break
            if len(cl) < FRAMES:
                continue
            # A bearing wraps: +179 and -179 point almost the same way but
            # are numerically opposite, and a correlation loss reads that as
            # a huge error.  The eye only spans +-145 deg anyway, so a beacon
            # behind the agent carries no information the circuit could use.
            # Judged on the LAST frame, which is the one the loss reads.
            if abs(cb[-1]) > 120.0:
                continue
            lums.append(np.stack(cl))
            scents.append(np.array(cs, np.float32))
            masses.append(np.array(cm, np.float32))
            bearings.append(np.array(cb, np.float32))
            invtaus.append(np.array(ct, np.float32))
        print("  seed %d: %d clips (%.0f s)"
              % (si, len(lums), time.perf_counter() - t0), flush=True)
    return (np.stack(lums), np.stack(scents), np.stack(masses),
            np.stack(bearings), np.stack(invtaus))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--clips", type=int, default=1200)
    ap.add_argument("--out", default="train")
    args = ap.parse_args(argv)

    lum, scent, mass, brg, itau = collect(args.seeds, args.clips)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / ("%s.npz" % args.out)
    np.savez_compressed(p, lum=lum, scent=scent, mass=mass,
                        bearing=brg, inv_tau=itau, seeds=np.array(args.seeds))
    print("%d clips x %d frames x %d ommatidia -> %s (%.0f MB)"
          % (lum.shape[0], lum.shape[1], lum.shape[2], p.name,
             p.stat().st_size / 1e6))
    print("  bearing  %+.1f .. %+.1f deg" % (brg.min(), brg.max()))
    print("  inv_tau  %.3f .. %.3f 1/s" % (itau.min(), itau.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
