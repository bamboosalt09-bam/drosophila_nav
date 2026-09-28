"""Room 6, 1 beam: why does the planner collide?  Log its last steps."""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import numpy as np

import cx_vp_room as C
import sim.drone_layer as DL
from cx_room import planner_for
from environment.room_file import ensure as room
from sensors.rangefinder import LAYOUTS

w, h, reach = room(6)
log, agents = [], []
_f = DL.follow
def spy(u, pts, rear, mem=None, pose=None):
    k = u / DL.V_CRUISE
    out = _f(u, pts, rear, mem, pose)
    log.append((u, DL.hit_distance(pts, k), DL.hit_distance(pts, 0.0),
                "dir" in (mem or {}), out[0], out[1], len(pts)))
    return out
C.follow = spy
_A = C.Agent
class A(_A):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        agents.append(self)
C.Agent = A

res = C.fly_reference(w, h, LAYOUTS[1], planner=planner_for)
ag = agents[-1]
print("planner, 1 beam:", {k: res[k] for k in ("found_ok" if False else "found",
                                               "collided", "steps")})
path = ag.path_array()
p = path[-1]
near = min(w.obstacles, key=lambda o: o.distance_to(p))
bear = math.degrees((math.atan2(near.y - p[1], near.x - p[0]) - ag.heading
                     + math.pi) % (2 * math.pi) - math.pi)
print("hit obstacle at bearing %+.0f deg from the nose (0 = straight ahead)" % bear)
print("\n step  u_int(deg/s)  d_path  d_ahead  sacc  yaw(deg/s)  v   echoes")
for i, (u, dp, da, sc, r, v, n) in enumerate(log[-25:]):
    print("%5d %12.0f %7.2f %8.2f %5d %11.0f %5.2f %5d"
          % (len(log) - 25 + i, math.degrees(u), min(dp, 9.99), min(da, 9.99),
             sc, math.degrees(r), v, n))
