"""One target-approach episode, and what the resulting path cost.

Same runner for both arms.  That is the point: "동일 학습 동일 환경" is only
true if there is literally one loop, one dt, one seed stream, one episode
budget, and the only thing swapped is the `Policy`.

    cue = sensor.sense(pose)            two cameras, target centroid
    r_cmd = policy.act(cue, heading)    fly core, or GRU
    agent.step(r_cmd, ...)              yaw goes through the drone's limits

Path efficiency is the deliverable, so it is measured here rather than
reconstructed later:

    detour          path length / straight-line distance.  1.0 is optimal,
                    and it is the number "경로의 효율성" most directly means
    turning         integrated |yaw rate| -- control effort, and what
                    separates a smooth approach from a wobbling one
    saturation      fraction of steps where the body could not deliver what
                    the brain asked.  Free: `YawPlant` already counts it, and
                    it IS the body-brain mismatch, not a proxy for it

ponytail: altitude is held and forward speed is constant.  Obstacles are
vertical cylinders so a fixed altitude still gives a real obstacle field, and
the study is about steering -- master doc section 22 puts planar navigation
before 3D.  Add climb control when the question needs a third axis; the agent
already takes `climb_cmd`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from environment.target_world import Agent, TargetWorld
from sensors.attraction_cue import AttractionCueSensor

DT = 0.1                 # s, the core's own 10 Hz cycle (Stage 0 calibration)
V_CRUISE = 2.0           # m/s


@dataclass
class EpisodeResult:
    path: np.ndarray                 # (n, 3)
    reached: bool
    collided: bool
    steps: int
    straight_line: float
    reach_radius: float
    yaw_rates: np.ndarray
    rate_saturation: float
    accel_saturation: float
    cue_valid_fraction: float

    @property
    def path_length(self) -> float:
        return float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum())

    @property
    def detour(self) -> float:
        """Path length over the SHORTEST path that counts as reaching.

        The denominator is `straight_line - reach_radius`, not the full
        straight line: success is being within `reach_radius` of the target,
        so an optimal run stops that far short and a naive ratio reads 0.953
        for a perfect flight.  Worse, the floor would then move with the start
        distance, which makes conditions incomparable -- the exact thing this
        metric exists to allow.  1.0 is optimal, higher is worse.

        NaN when the target was never reached: a short path that stopped at an
        obstacle is not efficient, and letting it score well would invert the
        metric.  Report `reached` alongside, always.
        """
        floor = self.straight_line - self.reach_radius
        if not self.reached or floor <= 0:
            return float("nan")
        return self.path_length / floor

    @property
    def turning(self) -> float:
        """Integrated |yaw rate|, rad.  Control effort."""
        return float(np.abs(self.yaw_rates).sum() * DT)

    @property
    def time_s(self) -> float:
        return self.steps * DT

    def summary(self) -> dict:
        return {"reached": self.reached, "collided": self.collided,
                "time_s": self.time_s, "path_length": self.path_length,
                "detour": self.detour, "turning": self.turning,
                "rate_saturation": self.rate_saturation,
                "accel_saturation": self.accel_saturation,
                "cue_valid_fraction": self.cue_valid_fraction}


def run_episode(policy, world: TargetWorld, agent: Agent,
                sensor: Optional[AttractionCueSensor] = None,
                max_steps: int = 100, dt: float = DT,
                v_cruise: float = V_CRUISE) -> EpisodeResult:
    """Fly until the target is reached, something is hit, or time runs out."""
    sensor = sensor or AttractionCueSensor(target_radius=world.target_radius)
    policy.reset()
    agent.plant.reset(psi=agent.heading, r=0.0)

    rates: List[float] = []
    n_valid = 0
    steps = 0
    for steps in range(1, max_steps + 1):
        cue = sensor.sense(agent.p, agent.heading, world.target,
                           world.obstacles)
        n_valid += int(cue.valid)
        r_cmd = policy.act(cue, agent.heading)
        agent.step(r_cmd, v_cruise, 0.0, dt)
        rates.append(agent.plant.r)
        if agent.collided or world.reached(agent.p):
            break

    return EpisodeResult(
        path=agent.path_array(),
        reached=bool(world.reached(agent.p)) and not agent.collided,
        collided=bool(agent.collided),
        steps=steps,
        straight_line=world.straight_line_length(agent.start),
        reach_radius=float(world.reach_radius),
        yaw_rates=np.array(rates),
        rate_saturation=float(agent.plant.rate_saturation_fraction),
        accel_saturation=float(agent.plant.accel_saturation_fraction),
        cue_valid_fraction=n_valid / max(steps, 1))


def demo() -> None:
    """Check the metrics separate a good run from a bad one."""
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "src"))

    from body.yaw_plant import YawPlantParams
    from core import calibration
    from core.westeinde2024 import WesteindeSteeringCore
    from decoder.steering import SOURCE_KAPPA_DEG_PER_STEP
    from environment.target_world import TargetWorld
    from sim.policies import FlyPolicy

    try:
        core = WesteindeSteeringCore(
            norm=calibration.load(repo / "configs/norm_constants.json"))
    except Exception:
        core = WesteindeSteeringCore()
    fly = FlyPolicy(core, SOURCE_KAPPA_DEG_PER_STEP)

    # open field, target off to one side: the fly has to steer to reach it
    world = TargetWorld(target=np.array([15.0, 8.0, 2.0]), obstacles=[])
    agent = Agent(world=world, start=(0.0, 0.0, 2.0), heading=0.0)
    good = run_episode(fly, world, agent)
    print("fly, open field   : %s" % {k: (round(v, 3) if isinstance(v, float)
                                          else v)
                                      for k, v in good.summary().items()})
    assert good.reached, "the fly should reach an unobstructed target"
    assert good.detour < 1.6, good.detour

    # an optimal run -- target dead ahead, no turning needed -- must read 1.0
    ahead = TargetWorld(target=np.array([15.0, 0.0, 2.0]), obstacles=[])
    opt = run_episode(fly, ahead, Agent(world=ahead, start=(0.0, 0.0, 2.0)))
    print("optimal straight  : detour %.4f (must be ~1.0)" % opt.detour)
    assert abs(opt.detour - 1.0) < 0.05, opt.detour

    # a policy that never turns must do worse on the same problem
    class Straight:
        def reset(self): pass
        def act(self, cue, heading): return 0.0
    agent.reset(heading=0.0)
    blind = run_episode(Straight(), world, agent)
    print("no-turn control   : reached=%s, detour=%s"
          % (blind.reached, blind.detour))
    assert not blind.reached, "flying straight past the target should fail"

    # a tightly limited body must saturate where a free one does not
    agent.reset(heading=0.0)
    free = run_episode(fly, world, agent)
    tight_agent = Agent(world=world, start=(0.0, 0.0, 2.0), heading=0.0,
                        yaw_params=YawPlantParams(r_max=0.05, alpha_max=0.5,
                                                  tau_r=0.2))
    tight = run_episode(fly, world, tight_agent)
    print("free body         : saturation %.3f, reached=%s, detour=%s"
          % (free.rate_saturation, free.reached, round(free.detour, 3)
             if free.reached else "n/a"))
    print("rate-limited body : saturation %.3f, reached=%s, detour=%s"
          % (tight.rate_saturation, tight.reached,
             round(tight.detour, 3) if tight.reached else "n/a"))
    assert tight.rate_saturation > free.rate_saturation, \
        "the constrained body must saturate more -- that is the mismatch"

    # detour must refuse to reward a run that failed
    assert np.isnan(blind.detour), "an unreached target cannot have a detour"
    print("demo ok")


if __name__ == "__main__":
    demo()
