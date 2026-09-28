"""The closed loop, done the way NeuroMechFly v2 is meant to be driven.

    flygym renders      ->  get_ommatidia_readouts()   721 per eye
    bridge joins        ->  ~23,720 columnar cells
    connectome runs     ->  166,700 neurons, dt 5 ms
    DESCENDING readout  ->  two numbers, [delta_L, delta_R]
    flygym's own CPG    ->  tripod rhythm, step kinematics, adhesion,
                            retraction and stumbling corrections
    50 physics steps    ->  the view changes, back to the top

What replaced what
------------------
The first version of this loop read 909 motor neurons into 24 joint angles and
drove the position actuators directly.  That could not work and the reason is
structural, not a tuning failure: walking is a phase-shifted rhythm, our rate
model converges to a fixed point (measured: per-step change 2.5e-05), and a
fixed point has no rhythm in it.

It also did not have to work.  NeuroMechFly v2's descending interface is TWO
NUMBERS.  The rhythm is the body's job -- that is what a central pattern
generator is -- and flygym ships one, together with the step kinematics,
adhesion, and the contact/leg-height corrections that keep the rhythm in touch
with the ground.  `reference/flygym_demo/` holds that code, taken from
NeLy-EPFL/flygym unmodified.

This does not weaken the project's rule.  The rule forbids a body plugin that
receives a goal or a heading error and acts as a navigation controller.  The
CPG receives neither: it gets a per-side drive amplitude and produces a gait.
Deciding WHERE to go remains entirely inside the connectome, and the only
things designated are still anatomical -- the eye at one end, the descending
neurons at the other.

The readout
-----------
`delta_L` and `delta_R` are the mean rate of the left and right
`descending_neuron` populations.  That is anatomy, not function: no cell type
is picked, no steering circuit is identified, all 1,314 are used.

ponytail: ipsilateral convention -- left descending neurons drive the left
CPG.  Insect DNs are largely ipsilateral so this is the right default, but it
is an assumption and `swap_sides=True` is the control that tests it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

BRAIN_DT_MS = 5.0          # the rate model's step
PHYSICS_DT_S = 1e-4        # MuJoCo's, from mujoco_globals.yaml
STEPS_PER_BRAIN_STEP = int(round(BRAIN_DT_MS / 1000.0 / PHYSICS_DT_S))   # 50


@dataclass
class WalkTrace:
    """What happened, per brain step."""

    descending: List[np.ndarray] = field(default_factory=list)
    position: List[np.ndarray] = field(default_factory=list)
    heading: List[np.ndarray] = field(default_factory=list)
    lum_mean: List[float] = field(default_factory=list)
    lum_std: List[float] = field(default_factory=list)

    def as_arrays(self):
        return {k: np.array(getattr(self, k)) for k in
                ("descending", "position", "heading", "lum_mean", "lum_std")}

    def yaw_deg(self) -> np.ndarray:
        h = np.array(self.heading)
        a = np.degrees(np.arctan2(h[:, 1], h[:, 0]))
        return (a - a[0] + 180.0) % 360.0 - 180.0


def build_walking_fly(name: str = "nmf"):
    """flygym's own locomotion fly, plus the eyes we need.

    `make_locomotion_fly` carries the parameters the walking examples are
    tuned with -- YAW_PITCH_ROLL axis order, soft leg joints (stiffness 0.05,
    damping 0.06) with stiff passive tarsi, kp 45, forcerange +-65, and leg
    adhesion at gain 40.  Adhesion is not optional: a fly without it slides.
    """
    from flygym_demo.complex_terrain import make_locomotion_fly

    fly = make_locomotion_fly(name=name)
    fly.add_vision()
    return fly


class ConnectomeWalkLoop:
    """Connectome decides where to go; flygym's CPG decides how to step."""

    def __init__(self, net, ann, lattice, omm_index, sim, fly, controller,
                 drive_gain: float = 0.20, descending_gain: float = 1.0,
                 swap_sides: bool = False, bias: float = 0.0):
        self.net, self.ann, self.lat = net, ann, lattice
        self.omm = omm_index
        self.sim, self.fly, self.ctrl = sim, fly, controller
        self.drive_gain = drive_gain
        self.descending_gain = descending_gain
        self.swap_sides = swap_sides
        self.bias = bias
        self.v = net.init_state(1)

        import pandas as pd
        idx = pd.Index(net.ids).get_indexer(lattice.root_id)
        self.rows, self.valid = idx, idx >= 0
        self.eye_is_left = lattice.eye == "left"

        sc = ann["super_class"].to_numpy(dtype="<U32")
        side = ann["side"].to_numpy(dtype="<U16")
        desc = sc == "descending_neuron"
        self.desc_l = np.flatnonzero(desc & (side == "left"))
        self.desc_r = np.flatnonzero(desc & (side == "right"))

        body_order = fly.get_bodysegs_order()
        self._thorax = body_order.index(
            type(fly).BODY_SEGMENT_CLASS("c_thorax"))

    def _visual_drive(self):
        readouts = self.sim.get_ommatidia_readouts(self.fly.name)   # (2, n, 2)
        per_eye = readouts.sum(axis=-1)
        lum = np.where(self.eye_is_left,
                       per_eye[0][self.omm], per_eye[1][self.omm])
        drive = torch.zeros(self.net.n, 1)
        sel = self.valid
        drive[self.rows[sel], 0] = torch.from_numpy(
            (lum[sel] * self.drive_gain).astype(np.float32))
        return drive, float(lum[sel].mean()), float(lum[sel].std())

    def descending_signal(self, rates: np.ndarray) -> np.ndarray:
        """Mean rate of each side's descending population, scaled."""
        l = float(rates[self.desc_l].mean()) if self.desc_l.size else 0.0
        r = float(rates[self.desc_r].mean()) if self.desc_r.size else 0.0
        if self.swap_sides:
            l, r = r, l
        return np.array([l, r]) * self.descending_gain + self.bias

    def step(self, trace: Optional[WalkTrace] = None) -> np.ndarray:
        """One brain step, then the 50 physics steps it covers."""
        from core.flywire_rate import activity
        from flygym_demo.complex_terrain import (
            HybridControllerObservation, apply_locomotion_action)

        drive, lmean, lstd = self._visual_drive()
        with torch.no_grad():
            self.v = self.net.step(self.v, drive, BRAIN_DT_MS)
            rates = activity(self.v).numpy().ravel()
        delta = self.descending_signal(rates)

        # the descending command is held constant while the body steps, the
        # same zero-order hold Stage 1 used between neural and body cycles
        for _ in range(STEPS_PER_BRAIN_STEP):
            obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
            action = self.ctrl.step(delta, obs)
            apply_locomotion_action(self.sim, self.fly.name, action)
            self.sim.step()

        if trace is not None:
            pos = np.asarray(self.sim.get_body_positions(self.fly.name))
            trace.descending.append(delta.copy())
            trace.position.append(pos[self._thorax].copy())
            trace.heading.append(obs.fly_heading.copy())
            trace.lum_mean.append(lmean)
            trace.lum_std.append(lstd)
        return delta


def make_sim_and_controller(steps_obj=None, spawn_z: float = 1.0, seed: int = 0):
    """A walking fly on flat ground, with its controller, ready to step."""
    from flygym import Simulation
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import (
        HybridTurningController, PreprogrammedSteps)

    fly = build_walking_fly()
    world = FlatGroundWorld()
    world.add_fly(fly, spawn_position=[0.0, 0.0, spawn_z],
                  spawn_rotation=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    sim.reset()
    ctrl = HybridTurningController(
        timestep=sim.mj_model.opt.timestep,
        preprogrammed_steps=steps_obj or PreprogrammedSteps())
    ctrl.reset(seed=seed)
    return sim, fly, ctrl


def demo(n_steps: int = 40) -> None:
    """Close the loop and report what the connectome actually commands."""
    import sys
    import time
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "reference"))

    from flygym.vision.retina import Retina

    import sensors.flywire_eye as eye
    import sensors.flygym_bridge as bridge
    from core.malecns import load_malecns
    from core.flywire_rate import FlyWireRate

    assert STEPS_PER_BRAIN_STEP == 50, STEPS_PER_BRAIN_STEP

    t0 = time.perf_counter()
    ids, out, ann, _ = load_malecns(w_scale=0.005, symmetrise=True)
    net = FlyWireRate(out_csr=out, ids=ids, ann=ann)
    lat = eye.load_malecns_eye(ann)
    omm, _ = bridge.join(lat, Retina())
    print("brain %d neurons, %d columns (%.1f s)"
          % (net.n, len(omm), time.perf_counter() - t0))

    sim, fly, ctrl = make_sim_and_controller()
    loop = ConnectomeWalkLoop(net, ann, lat, omm, sim, fly, ctrl)
    print("reading %d left / %d right descending neurons"
          % (len(loop.desc_l), len(loop.desc_r)))

    trace = WalkTrace()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        loop.step(trace)
    el = time.perf_counter() - t0
    bio = n_steps * BRAIN_DT_MS / 1000.0
    a = trace.as_arrays()
    d = a["descending"]
    moved = np.linalg.norm(a["position"][-1][:2] - a["position"][0][:2])

    print("%d brain steps (%.0f ms fly time) in %.0f s -> %.0f s per "
          "biological second" % (n_steps, bio * 1000, el, el / bio))
    print("  descending L %.3e..%.3e  R %.3e..%.3e"
          % (d[:, 0].min(), d[:, 0].max(), d[:, 1].min(), d[:, 1].max()))
    print("  L-R difference %.3e..%.3e" % ((d[:, 1] - d[:, 0]).min(),
                                           (d[:, 1] - d[:, 0]).max()))
    print("  moved %.3f mm, yaw %.1f deg, z %.3f"
          % (moved, trace.yaw_deg()[-1], a["position"][-1][2]))
    assert len(trace.descending) == n_steps
    print("demo ok")


if __name__ == "__main__":
    demo()
