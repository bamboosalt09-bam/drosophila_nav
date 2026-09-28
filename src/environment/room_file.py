"""Rooms live in files.

Generating one costs about 90 seconds and the result never changes.

`_try_room` runs seven flyability BFS searches -- one per beacon plus one for
the start -- on a 0.3 m grid over a 40 m room, i.e. 134 x 134 cells, and
every cell calls `world.collides` against 161 obstacles in a Python loop.
That is up to 2.9 M distance evaluations per search, seven searches per
attempt, and up to `max_tries` attempts until a room appears with enough
reachable beacons and at least one that is not.

Measured on 2026-09-21: 90 s of every 120 s run, against 1.2 s for the
connectome and 1.0 s for the rendering.  Nothing in the experiment was slow;
the room was being rebuilt from scratch every time.

Write it once:

    .venv/Scripts/python.exe -m environment.room_file 0 1 2

and every run after that loads a file.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOMS = Path(__file__).resolve().parents[2] / "results" / "rooms"


def path_for(seed: int) -> Path:
    return ROOMS / ("room%d.npz" % seed)


# Flyability margin for the "reachable" flag: body radius 0.25 m plus 0.25.
# Rooms were GENERATED against 1.4 m, a margin taken from the 1.27 m turning
# radius at 2 m/s.  The drone layer now slows to hold a curve instead of
# swinging wide, so that argument no longer holds -- and in room 0 the
# connectome reached a beacon flagged walled-off at 1.4 m, through a gap the
# flag said it could not fly.
FLY_CLEARANCE = 0.5
START = (0.0, 0.0, 2.0)


def reflag(seed: int, clearance: float = FLY_CLEARANCE):
    """Recompute and save a room's reachable flags at `clearance`."""
    from environment.target_world import TargetWorld, shortest_path
    w, h, old = load(seed)
    new = [shortest_path(TargetWorld(target=t, obstacles=w.obstacles,
                                     bounds=w.bounds),
                         START, cell=0.3, clearance=clearance) is not None
           for t in w.all_targets()]
    save(seed, w, h, new)
    return old, new


def save(seed: int, world, heading: float, reachable: List[bool]) -> Path:
    """Write one room.  Obstacles become (N, 5); NaN height means a sphere."""
    obs = np.array([[o.x, o.y, o.radius, o.z,
                     np.nan if o.height is None else o.height]
                    for o in world.obstacles], dtype=np.float64).reshape(-1, 5)
    ROOMS.mkdir(parents=True, exist_ok=True)
    p = path_for(seed)
    np.savez(p, obstacles=obs,
             targets=np.array(world.all_targets(), dtype=np.float64),
             heading=np.float64(heading),
             reachable=np.array(reachable, dtype=bool),
             bounds=np.array(world.bounds, dtype=np.float64),
             target_radius=np.float64(world.target_radius))
    return p


def load(seed: int) -> Tuple[object, float, List[bool]]:
    """Read one room, in the shape `room_world` returns."""
    from environment.target_world import Obstacle, TargetWorld

    p = path_for(seed)
    if not p.exists():
        raise FileNotFoundError(
            "no room file %s -- generate it with\n"
            "    .venv/Scripts/python.exe -m environment.room_file %d" % (p, seed))
    z = np.load(p)
    obs = [Obstacle(x=float(r[0]), y=float(r[1]), radius=float(r[2]),
                    z=float(r[3]),
                    height=None if np.isnan(r[4]) else float(r[4]))
           for r in z["obstacles"]]
    tg = [np.asarray(t, dtype=float) for t in z["targets"]]
    w = TargetWorld(target=tg[0], extra_targets=tg[1:], obstacles=obs,
                    bounds=tuple(float(v) for v in z["bounds"]),
                    target_radius=float(z["target_radius"]))
    return w, float(z["heading"]), [bool(b) for b in z["reachable"]]


def ensure(seed: int, **kw):
    """Load the room, generating and saving it the first time only."""
    try:
        return load(seed)
    except FileNotFoundError:
        from environment.target_world import room_world
        out = room_world(seed=seed, **kw)
        save(seed, *out)
        return out


def main(argv=None) -> int:
    import sys
    import time
    from environment.target_world import room_world

    seeds = [int(a) for a in (argv if argv is not None else sys.argv[1:])] or [0]
    for s in seeds:
        t = time.perf_counter()
        w, h, r = room_world(seed=s)
        p = save(s, w, h, r)
        print("room %d: %d obstacles, %d beacons, %d reachable, heading %+.4f"
              "  (%.0f s)  -> %s"
              % (s, len(w.obstacles), len(w.all_targets()), sum(r), h,
                 time.perf_counter() - t, p.name))
        # the file must reproduce what was generated, or it is not a cache
        w2, h2, r2 = load(s)
        assert len(w2.obstacles) == len(w.obstacles)
        assert h2 == h and r2 == r
        assert np.allclose(np.array(w2.all_targets()),
                           np.array(w.all_targets()))
        for a, b in zip(w2.obstacles, w.obstacles):
            assert (a.x, a.y, a.radius, a.z, a.height) == \
                   (b.x, b.y, b.radius, b.z, b.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
