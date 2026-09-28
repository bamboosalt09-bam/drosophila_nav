"""A 3D environment with a target to reach and obstacles in the way.

The body is a drone, not a fly, so the agent is the standard unicycle-with-
altitude that master doc section 22 describes as the step past 1-DOF yaw:

    state   x y z psi v r
    control yaw rate command, forward speed command, climb rate command

Yaw goes through the SAME `YawPlant` Stage 1 swept 252 bodies with, so
"드론 자체 거동 허용 범위" means exactly what it meant there -- `r_max`,
`alpha_max`, `tau_r` -- and every body condition from Stage 1 transfers
unchanged.  Speed and climb get their own rate limits because a drone has
them, but they are deliberately simple: the study is about steering.

Obstacles are vertical cylinders and spheres.  That is not a poverty of
imagination, it is so the cameras can be projected analytically: a rendered
blob and an analytically projected one give the same centroid, and the
analytic version is ~100x cheaper (0.28 ms against 12 ms per camera,
measured).  Rendering belongs to the renderer, which master doc section 20
keeps separate from the simulation for exactly this reason.

ponytail: no aerodynamics, no wind, no attitude dynamics -- forbidden by
master doc section 41 and not what is being measured.  Add them the day the
question is about flight mechanics rather than steering.
"""
from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from body.yaw_plant import YawPlant, YawPlantParams


@dataclass(frozen=True)
class Obstacle:
    """A vertical cylinder, or a sphere when `height` is None."""

    x: float
    y: float
    radius: float
    z: float = 0.0
    height: Optional[float] = None      # None -> sphere centred at (x, y, z)

    def distance_to(self, p: np.ndarray) -> float:
        """Signed distance from point `p` to the surface; negative is inside."""
        if self.height is None:
            return float(np.linalg.norm(p - np.array([self.x, self.y, self.z]))
                         - self.radius)
        radial = math.hypot(p[0] - self.x, p[1] - self.y) - self.radius
        if self.z <= p[2] <= self.z + self.height:
            return radial
        dz = (self.z - p[2]) if p[2] < self.z else (p[2] - self.z - self.height)
        return math.hypot(max(radial, 0.0), dz) if radial > 0 else dz


@dataclass
class AgentLimits:
    """What the vehicle can do.  Yaw limits live in `YawPlantParams`."""

    v_max: float = 2.0              # m/s forward
    # Reverse is allowed again now that the range-sensor ring covers the
    # rear sector.  It was disabled while the only sensing faced forward,
    # because backing up then meant accelerating blind.  Capped well below
    # forward speed; the speed law only uses it when the rear is clear.
    v_min: float = -0.5             # m/s, reverse
    v_accel_max: float = 4.0        # m/s^2
    climb_max: float = 1.0          # m/s
    climb_accel_max: float = 2.0    # m/s^2


@dataclass
class TargetWorld:
    """Target, obstacles, bounds.  Knows nothing about controllers."""

    target: np.ndarray = field(
        default_factory=lambda: np.array([20.0, 0.0, 2.0]))
    # Extra targets, for the search task.  `target` stays the first one so
    # every existing single-target caller is untouched; `all_targets()` is
    # what the scene and the cue sensor read.
    extra_targets: List[np.ndarray] = field(default_factory=list)
    target_radius: float = 0.6
    obstacles: List[Obstacle] = field(default_factory=list)
    bounds: Tuple[float, float, float, float] = (-5.0, 35.0, -20.0, 20.0)
    z_bounds: Tuple[float, float] = (0.3, 8.0)
    reach_radius: float = 1.0

    def collides(self, p: np.ndarray, clearance: float = 0.25) -> bool:
        if not (self.z_bounds[0] <= p[2] <= self.z_bounds[1]):
            return True
        x0, x1, y0, y1 = self.bounds
        if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1):
            return True
        return any(o.distance_to(p) < clearance for o in self.obstacles)

    def all_targets(self) -> List[np.ndarray]:
        return [self.target] + list(self.extra_targets)

    def reached(self, p: np.ndarray) -> bool:
        return bool(np.linalg.norm(p - self.target) < self.reach_radius)

    def reached_any(self, p: np.ndarray):
        """Index of the target reached, or None.  Beacons are vertical, so
        the test is horizontal distance -- altitude is held anyway."""
        for i, t in enumerate(self.all_targets()):
            if math.hypot(p[0] - t[0], p[1] - t[1]) < self.reach_radius:
                return i
        return None

    def straight_line_length(self, start: np.ndarray) -> float:
        return float(np.linalg.norm(self.target - start))


# Scene luminances.  The target is BRIGHT and everything else is dark, which
# is the premise the whole cue design already rested on -- "목표물이 광도가
# 높다고 가정한 것임" -- but which the scene itself never implemented: the
# target was appended to the obstacle list and rendered as the same 0.0
# silhouette, so the eyes could not tell the goal from a pillar.  With the
# lamina transmitting contrast, a dark sky makes the target the one strong
# positive deviation in the visual field and the obstacles weak negative ones.
# Nothing emits onto anything else: this is analytic ray casting, so a bright
# target casts no light and creates no reflection.
SKY_LUMINANCE = 0.15
OBSTACLE_LUMINANCE = 0.0
TARGET_LUMINANCE = 1.0
# The target is a vertical luminous beacon, not a glowing ball.  As a sphere
# of radius 0.6 it subtends 8.6 deg at 8 m and only cells near the horizon
# can see it: of 1,779 injected lamina cells just 60 lie within +-5 deg of
# the horizon across the frontal +-50 deg, and at some bearings the target
# fell between them entirely -- measured 0 bright cells at +20 deg against 7
# at -40.  That sampling gap, not the circuit, chopped the bearing response
# into a non-monotonic staircase.  A vertical beacon is sampled by every
# elevation row, 756 cells over the same frontal window, and it is what a
# high-luminance marker physically looks like anyway.  Physics is untouched:
# reaching and collision still use the 3D distance to the point.
TARGET_HEIGHT = 6.0


def world_scene(world: "TargetWorld", position: np.ndarray,
                include_target: bool = True, sky: float = SKY_LUMINANCE,
                obstacle: float = OBSTACLE_LUMINANCE,
                target: float = TARGET_LUMINANCE):
    """What the eyes see from `position`: a bright target against a dark sky,
    with obstacles darker still.

    Returns the same kind of callable `sensors.flywire_eye.bar_scene` does --
    (world azimuth, elevation) in degrees -> luminance -- so it drops straight
    into `eye.sample()` and the existing lamina injection.

    Analytic ray casting, no rendering.  Obstacles are cylinders and spheres
    precisely so this stays a closed-form quadratic per object: ~6,199 rays
    against ~14 objects is a handful of vectorised numpy ops, against 12 ms
    for a rendered camera (measured).

    Unlike the earlier version this resolves the NEAREST hit rather than any
    hit, which it has to now that objects differ in brightness: a target
    behind a pillar must read as the pillar, the way the attraction cue
    sensor already reports it occluded.

    ponytail: no shading and no texture -- three flat luminances.  Add
    greyscale the day the task needs to tell two obstacles apart.
    """
    p = np.asarray(position, dtype=float)
    objs = [(o, obstacle) for o in world.obstacles]
    if include_target:
        for t in world.all_targets():
            objs.append((Obstacle(x=float(t[0]), y=float(t[1]), z=0.0,
                                  radius=world.target_radius,
                                  height=TARGET_HEIGHT),
                         target))

    def luminance(azimuth_deg, elevation_deg):
        az = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
        el = np.deg2rad(np.asarray(elevation_deg, dtype=float))
        dx, dy, dz = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
        best_t = np.full(az.shape, np.inf)
        lum = np.full(az.shape, float(sky))

        for o, val in objs:
            ox, oy, oz = p[0] - o.x, p[1] - o.y, p[2] - o.z
            t_hit = np.full(az.shape, np.inf)
            if o.height is None:                       # sphere
                b = 2 * (dx * ox + dy * oy + dz * oz)
                c = ox * ox + oy * oy + oz * oz - o.radius ** 2
                disc = b * b - 4 * c                   # a == 1, d is a unit vector
                ok = disc >= 0
                if not ok.any():
                    continue
                t = (-b - np.sqrt(np.maximum(disc, 0))) / 2
                t_hit = np.where(ok & (t > 0), t, np.inf)
            else:                                      # vertical cylinder
                a = dx * dx + dy * dy
                b = 2 * (dx * ox + dy * oy)
                c = ox * ox + oy * oy - o.radius ** 2
                disc = b * b - 4 * a * c
                ok = (disc >= 0) & (a > 1e-12)
                if not ok.any():
                    continue
                sq = np.sqrt(np.maximum(disc, 0))
                for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
                    zh = p[2] + t * dz
                    valid = ok & (t > 0) & (zh >= o.z) & (zh <= o.z + o.height)
                    t_hit = np.where(valid, np.minimum(t_hit, t), t_hit)
            closer = t_hit < best_t
            best_t = np.where(closer, t_hit, best_t)
            lum = np.where(closer, val, lum)
        return lum

    return luminance


def corridor_world(seed: int = 0, n_obstacles: int = 14,
                   spread: float = 9.0) -> TargetWorld:
    """Obstacles scattered between start and target.

    Seeded so a condition can be repeated exactly: paired trials across
    controllers are the whole basis of the comparison, the same way Stage 1
    paired its noise seeds.
    """
    # CACHED, because generating this room costs about 90 seconds and the
    # result is a pure function of the arguments.
    #
    # `_try_room` runs seven flyability BFS searches -- one per beacon plus
    # one for the start -- on a 0.3 m grid over a 40 m room, i.e. 134 x 134
    # cells, and every cell calls `world.collides` against 161 obstacles in
    # a Python loop.  That is up to 2.9 M distance evaluations per search,
    # seven searches per attempt, and up to `max_tries` attempts until a
    # room appears with enough reachable beacons and at least one that is
    # not.  Measured: it was 90 s of every 120 s run, dwarfing both the
    # connectome (1.2 s) and the rendering (1.0 s), and it was being redone
    # from scratch every single time for a room that never changes.
    import pickle
    key = (seed, half, n_targets, n_walls, tuple(float(v) for v in start),
           min_reachable, max_tries)
    cache_dir = Path(__file__).resolve().parents[2] / "results" / "rooms"
    cf = cache_dir / ("room_%s.pkl" % "_".join(str(k) for k in key)
                      .replace(" ", "").replace("(", "").replace(")", "")
                      .replace(",", "-"))
    if cf.exists():
        with cf.open("rb") as fh:
            return pickle.load(fh)

    for attempt in range(max_tries):
        out = _try_room(seed * 100 + attempt, half, n_targets, n_walls, start)
        if out is not None and sum(out[2]) >= min_reachable                 and not all(out[2]):
            cache_dir.mkdir(parents=True, exist_ok=True)
            with cf.open("wb") as fh:
                pickle.dump(out, fh)
            return out
    raise RuntimeError("no room with %d+ reachable and >=1 unreachable"
                       % min_reachable)


def _try_room(seed, half, n_targets, n_walls, start):
    rng = np.random.RandomState(seed)
    obs = []
    for _ in range(n_obstacles):
        x = rng.uniform(4.0, 17.0)
        y = rng.uniform(-spread, spread)
        obs.append(Obstacle(x=float(x), y=float(y),
                            radius=float(rng.uniform(0.4, 1.1)),
                            z=0.0, height=float(rng.uniform(3.0, 8.0))))
    return TargetWorld(obstacles=obs)


def is_reachable(world: TargetWorld, start=(0.0, 0.0, 2.0),
                 cell: float = 0.5) -> bool:
    """Is there ANY collision-free path from start to the target?"""
    return shortest_path(world, start, cell) is not None


def shortest_path(world: TargetWorld, start=(0.0, 0.0, 2.0),
                  cell: float = 0.5, clearance: float = 0.25):
    """The grid-shortest collision-free path, or None if the map is sealed.

    A cluttered map can simply be sealed, and then every arm fails for a
    reason that has nothing to do with any of them.  Grid BFS at the flight
    altitude, using the world own collision test so clearance is identical
    to what the agent experiences.

    ponytail: 2D grid at fixed altitude, 4-connected.  The agent holds
    altitude, so a 3D search would explore a dimension it cannot use; make
    this 3D the day climb becomes a control output.
    """
    from collections import deque
    x0, x1, y0, y1 = world.bounds
    z = float(start[2])
    nx, ny = int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1

    def free(i, j):
        # `clearance` is what makes this a FLYABILITY test rather than a
        # topology test.  At 2 m/s and 90 deg/s the minimum turning radius is
        # 1.27 m, so a corridor the grid can thread at 0.25 m clearance is
        # one the vehicle cannot actually fly.  Generating mazes against the
        # loose test produced maps where even the full-information planner
        # reached 1 of 4 -- an unsolvable map, not a hard one.
        return not world.collides(np.array([x0 + i * cell, y0 + j * cell, z]),
                                  clearance=clearance)

    def to_ij(p):
        return (int(round((p[0] - x0) / cell)), int(round((p[1] - y0) / cell)))

    si, sj = to_ij(np.asarray(start, dtype=float))
    ti, tj = to_ij(world.target)
    if not free(si, sj):
        return None
    prev = {(si, sj): None}
    q = deque([(si, sj)])
    goal_cells = max(int(world.reach_radius / cell), 1)
    while q:
        i, j = q.popleft()
        if abs(i - ti) + abs(j - tj) <= goal_cells:
            out, n = [], (i, j)
            while n is not None:
                out.append(np.array([x0 + n[0] * cell, y0 + n[1] * cell, z]))
                n = prev[n]
            return list(reversed(out))
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if (0 <= a < nx and 0 <= b < ny and (a, b) not in prev
                    and free(a, b)):
                prev[(a, b)] = (i, j)
                q.append((a, b))
    return None


def cluttered_world(seed: int = 0, n_obstacles: int = 45,
                    start=(0.0, 0.0, 2.0), max_tries: int = 40):
    """A dense field the target has to be FOUND in, not just flown to.

    `corridor_world` put 2-14 pillars between a start and a target 12 m dead
    ahead, and measurement showed what that really was: the target was inside
    the camera frame at t=0 in 96% of episodes and the median start bearing
    was 14 deg.  Nothing had to search for anything -- it was "fly at the
    thing you can already see", and all three arms drew nearly the same
    straight line whenever no pillar happened to sit on it.

    Here the target is 22-32 m away at ANY bearing, the heading starts
    anywhere in the full circle, and the field is dense enough to occlude.
    Finding the target is part of the task.

    Returns (world, heading).  Only maps with a proven collision-free path
    are returned, so a failure is always the controller's.
    """
    for t in range(max_tries):
        rng = np.random.RandomState(seed * 1000 + t)
        th = rng.uniform(-np.pi, np.pi)
        d = rng.uniform(22.0, 32.0)
        tgt = np.array([d * np.cos(th), d * np.sin(th), 2.0])
        obs = []
        guard = 0
        while len(obs) < n_obstacles and guard < 40 * n_obstacles:
            guard += 1
            x = rng.uniform(-34.0, 34.0)
            y = rng.uniform(-34.0, 34.0)
            r = float(rng.uniform(0.4, 1.6))
            # keep the start and the goal themselves clear, or the map is
            # unsolvable for reasons that are not navigation
            if math.hypot(x - start[0], y - start[1]) < 3.0 + r:
                continue
            if math.hypot(x - tgt[0], y - tgt[1]) < 2.5 + r:
                continue
            obs.append(Obstacle(x=x, y=y, radius=r, z=0.0,
                                height=float(rng.uniform(3.0, 8.0))))
        w = TargetWorld(target=tgt, obstacles=obs,
                        bounds=(-38.0, 38.0, -38.0, 38.0))
        if is_reachable(w, start):
            return w, float(rng.uniform(-np.pi, np.pi))
    raise RuntimeError("no solvable map after %d tries" % max_tries)


def maze_world(seed: int = 0, n_walls: int = 14, n_pillars: int = 90,
               start=(0.0, 0.0, 2.0), max_tries: int = 60):
    """Dense clutter WITH structure: walls, not just scattered pillars.

    `cluttered_world` blocks about 3% of its area with 45 independent
    pillars, so a straight line to the target is usually clear and the task
    rewards pointing at the goal.  Walls change the problem: a segment of
    touching pillars cannot be flown through at all, it has to be flown
    AROUND, and which way around is a decision that a purely reactive
    "steer toward the cue" cannot make correctly.

    Scattered pillars stay too -- a field of pure walls is a maze with
    corridors, which is a different and narrower task.

    Solvability is proven by grid BFS at a finer 0.3 m cell, because a 0.5 m
    grid will happily thread a gap a real vehicle cannot use.

    ponytail: BFS ignores the minimum turning radius (1.27 m at 2 m/s and
    90 deg/s), so "solvable" here means a path exists, not that it is
    flyable.  The full-information planner in the experiment is the honest
    check on that, and if it starts failing the maps have gone too far.
    """
    for t in range(max_tries):
        rng = np.random.RandomState(seed * 1000 + t)
        th = rng.uniform(-np.pi, np.pi)
        d = rng.uniform(24.0, 34.0)
        tgt = np.array([d * np.cos(th), d * np.sin(th), 2.0])
        obs = []

        def clear_of_endpoints(x, y, r):
            return (math.hypot(x - start[0], y - start[1]) > 3.5 + r
                    and math.hypot(x - tgt[0], y - tgt[1]) > 3.0 + r)

        for _ in range(n_walls):
            x0 = rng.uniform(-30.0, 30.0)
            y0 = rng.uniform(-30.0, 30.0)
            ang = rng.uniform(-np.pi, np.pi)
            length = rng.uniform(6.0, 18.0)
            r = float(rng.uniform(0.7, 1.2))
            n = max(int(length / (1.4 * r)), 2)
            for k in range(n):
                f = k / (n - 1)
                x = x0 + f * length * math.cos(ang)
                y = y0 + f * length * math.sin(ang)
                if abs(x) > 34 or abs(y) > 34 or not clear_of_endpoints(x, y, r):
                    continue
                obs.append(Obstacle(x=x, y=y, radius=r, z=0.0,
                                    height=float(rng.uniform(4.0, 8.0))))

        guard = 0
        n_scat = 0
        while n_scat < n_pillars and guard < 40 * n_pillars:
            guard += 1
            x = rng.uniform(-32.0, 32.0)
            y = rng.uniform(-32.0, 32.0)
            r = float(rng.uniform(0.4, 2.2))
            if not clear_of_endpoints(x, y, r):
                continue
            obs.append(Obstacle(x=x, y=y, radius=r, z=0.0,
                                height=float(rng.uniform(3.0, 8.0))))
            n_scat += 1

        w = TargetWorld(target=tgt, obstacles=obs,
                        bounds=(-38.0, 38.0, -38.0, 38.0))
        # 1.4 m: the 1.27 m turning radius with a little margin
        if shortest_path(w, start, cell=0.3, clearance=1.4) is not None:
            return w, float(rng.uniform(-np.pi, np.pi))
    raise RuntimeError("no flyable maze after %d tries" % max_tries)


def _wall(obs, x0, y0, x1, y1, r=0.9, rng=None):
    """A line of touching pillars.  Walls are pillars so the eyes see them
    the same way they see everything else -- no special-case geometry."""
    n = max(int(math.hypot(x1 - x0, y1 - y0) / (1.3 * r)), 2)
    for k in range(n + 1):
        f = k / n
        h = 6.0 if rng is None else float(rng.uniform(5.0, 8.0))
        obs.append(Obstacle(x=x0 + f * (x1 - x0), y=y0 + f * (y1 - y0),
                            radius=r, z=0.0, height=h))


def room_world(seed: int = 0, half: float = 18.0, n_targets: int = 6,
               n_walls: int = 6, start=(0.0, 0.0, 2.0),
               min_reachable: int = 3, max_tries: int = 60):
    """A closed room, partitioned, with several beacons -- some unreachable.

    Every open-field task so far let a lost agent simply leave: the
    connectome's characteristic failure was flying to the edge of the world
    with the target 60 m behind it, which says nothing about navigation.  A
    sealed room removes that escape, so the only ways to fail are to hit
    something or to run out of time.

    Several targets turn a 0-or-1 outcome into a score, and because some of
    them are walled off, an arm that fixates on an unreachable beacon can be
    told apart from one that gives up and finds another.

    Returns (world, heading, reachable) where `reachable` is a boolean per
    target, computed by the same flyability BFS the agent must satisfy.

    ponytail: partitions are straight walls with gaps, not a generated maze.
    A real maze generator is the upgrade if the rooms turn out too easy.
    """
    for attempt in range(max_tries):
        out = _try_room(seed * 100 + attempt, half, n_targets, n_walls, start)
        if out is not None and sum(out[2]) >= min_reachable                 and not all(out[2]):
            return out
    raise RuntimeError("no room with %d+ reachable and >=1 unreachable"
                       % min_reachable)


def _try_room(seed, half, n_targets, n_walls, start):
    rng = np.random.RandomState(seed)
    obs = []
    # outer wall
    for (a, b, c, d) in ((-half, -half, half, -half), (half, -half, half, half),
                         (half, half, -half, half), (-half, half, -half, -half)):
        _wall(obs, a, b, c, d, r=1.0)
    # interior partitions, each with one gap so the room stays partly connected
    for _ in range(n_walls):
        horiz = rng.rand() < 0.5
        pos = rng.uniform(-half + 4, half - 4)
        lo, hi = sorted(rng.uniform(-half + 1, half - 1, size=2))
        if hi - lo < 6.0:
            hi = min(lo + 6.0, half - 1)
        gap_c = rng.uniform(lo + 2.5, hi - 2.5) if hi - lo > 5.0             else 0.5 * (lo + hi)
        # A doorway has to clear the 1.4 m flyability margin on both sides of
        # a 0.9 m pillar, so anything under ~4 m is a wall with a decorative
        # notch.  A quarter of the partitions are sealed on purpose: the task
        # is supposed to contain beacons that cannot be reached.
        gap_w = 0.0 if rng.rand() < 0.25 else float(rng.uniform(4.5, 6.5))
        for (s0, s1) in ((lo, gap_c - gap_w / 2), (gap_c + gap_w / 2, hi)):
            if s1 - s0 < 1.0:
                continue
            if horiz:
                _wall(obs, s0, pos, s1, pos, r=0.9, rng=rng)
            else:
                _wall(obs, pos, s0, pos, s1, r=0.9, rng=rng)

    def far_enough(x, y, pts, d):
        return all(math.hypot(x - q[0], y - q[1]) > d for q in pts)

    placed, guard = [], 0
    while len(placed) < n_targets and guard < 4000:
        guard += 1
        x, y = rng.uniform(-half + 2, half - 2, size=2)
        if math.hypot(x - start[0], y - start[1]) < 6.0:
            continue
        if not far_enough(x, y, placed, 6.0):
            continue
        if any(o.distance_to(np.array([x, y, 2.0])) < 2.0 for o in obs):
            continue
        placed.append((x, y))

    if len(placed) < n_targets:
        return None
    tgts = [np.array([x, y, 2.0]) for x, y in placed]
    w = TargetWorld(target=tgts[0], extra_targets=tgts[1:], obstacles=obs,
                    bounds=(-half - 2, half + 2, -half - 2, half + 2))
    reachable = []
    for t in tgts:
        probe = TargetWorld(target=t, obstacles=obs, bounds=w.bounds)
        reachable.append(shortest_path(probe, start, cell=0.3,
                                       clearance=1.4) is not None)
    if shortest_path(TargetWorld(target=np.array([start[0], start[1], 2.0]),
                                 obstacles=obs, bounds=w.bounds),
                     start, cell=0.3, clearance=1.4) is None:
        return None                      # the start itself is walled in
    return w, float(rng.uniform(-np.pi, np.pi)), reachable


# What a small quadrotor can actually do in yaw.  `YawPlantParams` defaults
# to r_max = alpha_max = inf and tau_r = 0 -- the Stage 0 unconstrained
# baseline -- and every experiment built its Agent without passing anything,
# so the central premise of this project, "드론 자체 거동 허용 범위 내에서만
# 이동", has never once been enforced.  The docstring claimed it was.
#
#   r_max      90 deg/s.  Navigation-rate yaw for a small multirotor; aerobatic
#              airframes go past 200 deg/s but nothing here is doing acro.
#   alpha_max  5 rad/s^2.  Quadrotor yaw torque comes from rotor drag, not
#              thrust differential, so it is far weaker than roll or pitch.
#   tau_r      0.08 s.  Motor spin-up plus the inner rate loop.  Must stay
#              above 2*dt_body (0.005 s) for the explicit integrator.
#
# Override per experiment with Agent(yaw_params=...); an unconstrained body is
# still one line away and is the right control to run alongside.
DRONE_YAW = YawPlantParams(r_max=math.radians(90.0), alpha_max=5.0,
                           tau_r=0.08, label="small_quadrotor_yaw")


class Agent:
    """Unicycle with altitude.  Yaw goes through a constrained drone plant."""

    def __init__(self, world: TargetWorld,
                 yaw_params: Optional[YawPlantParams] = None,
                 limits: Optional[AgentLimits] = None,
                 start: Sequence[float] = (0.0, 0.0, 2.0),
                 heading: float = 0.0):
        self.world = world
        self.limits = limits or AgentLimits()
        self.plant = YawPlant(params=yaw_params or DRONE_YAW)
        self.start = np.array(start, dtype=float)
        self.reset(heading)

    def reset(self, heading: float = 0.0) -> None:
        self.p = self.start.copy()
        self.plant.reset(psi=heading, r=0.0)
        self.v = 0.0
        self.climb = 0.0
        self.path: List[np.ndarray] = [self.p.copy()]
        self.collided = False

    @property
    def heading(self) -> float:
        return self.plant.psi

    def step(self, r_cmd: float, v_cmd: float, climb_cmd: float,
             dt: float) -> None:
        """One control cycle.  `r_cmd` goes through the constrained plant."""
        lim = self.limits
        self.plant.step(r_cmd, dt)

        # speed and climb: first-order rate limits, same idea as the yaw cap
        dv = np.clip(v_cmd - self.v, -lim.v_accel_max * dt,
                     lim.v_accel_max * dt)
        self.v = float(np.clip(self.v + dv, lim.v_min, lim.v_max))
        dc = np.clip(climb_cmd - self.climb, -lim.climb_accel_max * dt,
                     lim.climb_accel_max * dt)
        self.climb = float(np.clip(self.climb + dc, -lim.climb_max,
                                   lim.climb_max))

        psi = self.plant.psi
        self.p = self.p + np.array([self.v * math.cos(psi) * dt,
                                    self.v * math.sin(psi) * dt,
                                    self.climb * dt])
        self.path.append(self.p.copy())
        if self.world.collides(self.p):
            self.collided = True

    def path_array(self) -> np.ndarray:
        return np.array(self.path)


def demo() -> None:
    """Check the world, the constraint and the collision test all bite."""
    world = corridor_world(seed=0)
    print("world: target %s, %d obstacles"
          % (world.target.tolist(), len(world.obstacles)))

    # an unconstrained agent flying straight reaches the target
    free = Agent(world=TargetWorld(obstacles=[]))
    for _ in range(2000):
        free.step(0.0, 2.0, 0.0, 0.01)
        if free.world.reached(free.p):
            break
    assert free.world.reached(free.p), free.p
    print("unconstrained straight run reaches target at %s"
          % np.round(free.p, 2).tolist())

    # the yaw constraint must actually limit the turn rate
    tight = Agent(world=TargetWorld(obstacles=[]),
                  yaw_params=YawPlantParams(r_max=0.2, alpha_max=1.0,
                                            tau_r=0.1))
    for _ in range(200):
        tight.step(10.0, 1.0, 0.0, 0.01)      # ask for far more than r_max
    assert abs(tight.plant.r) <= 0.2 + 1e-9, tight.plant.r
    print("yaw rate asked 10.0, delivered %.3f (r_max 0.2) -- constraint bites"
          % tight.plant.r)

    # obstacles must be hittable, or the environment is not an environment
    blocked = Agent(world=world)
    for _ in range(3000):
        blocked.step(0.0, 2.0, 0.0, 0.01)
        if blocked.collided or blocked.world.reached(blocked.p):
            break
    print("flying blind straight through the corridor: collided=%s at %s"
          % (blocked.collided, np.round(blocked.p, 2).tolist()))
    assert blocked.collided, "a blind straight run should hit something"

    # a sphere and a cylinder must give sane signed distances
    sph = Obstacle(x=0.0, y=0.0, radius=1.0, z=0.0)
    assert abs(sph.distance_to(np.array([2.0, 0.0, 0.0])) - 1.0) < 1e-9
    assert sph.distance_to(np.zeros(3)) < 0
    cyl = Obstacle(x=0.0, y=0.0, radius=1.0, z=0.0, height=2.0)
    assert abs(cyl.distance_to(np.array([3.0, 0.0, 1.0])) - 2.0) < 1e-9
    assert cyl.distance_to(np.array([0.0, 0.0, 5.0])) > 0, "above the cylinder"

    # --- what the eyes see -------------------------------------------------
    import math as _m
    pillar = TargetWorld(target=np.array([50.0, 0.0, 2.0]),
                         obstacles=[Obstacle(x=10.0, y=0.0, radius=1.0,
                                             z=0.0, height=6.0)])
    az = np.linspace(-180, 180, 721)
    el = np.zeros_like(az)

    def dark_width(dist):
        scene = world_scene(pillar, np.array([10.0 - dist, 0.0, 2.0]),
                            include_target=False)
        lum = scene(az, el)
        return float((lum == 0).sum()) * (360.0 / 721)

    w10 = dark_width(10.0)
    expected = 2 * _m.degrees(_m.asin(1.0 / 10.0))
    print("pillar r=1 at 10 m: dark arc %.2f deg (geometry says %.2f)"
          % (w10, expected))
    assert abs(w10 - expected) < 1.5, (w10, expected)

    # looming: the same pillar must subtend more as it gets closer
    w5, w3 = dark_width(5.0), dark_width(3.0)
    print("  at 5 m %.2f deg, at 3 m %.2f deg -- looming" % (w5, w3))
    assert w3 > w5 > w10, (w10, w5, w3)

    # and the sky behind must stay bright
    behind = world_scene(pillar, np.array([0.0, 0.0, 2.0]),
                         include_target=False)(np.array([180.0]),
                                               np.array([0.0]))
    assert behind[0] == 1.0, behind

    # a ray over the top of the pillar must miss it
    over = world_scene(pillar, np.array([0.0, 0.0, 2.0]),
                       include_target=False)(np.array([0.0]),
                                             np.array([80.0]))
    assert over[0] == 1.0, "steep ray should clear a 6 m pillar at 10 m"
    print("demo ok")


if __name__ == "__main__":
    demo()
