"""The closed loop: connectome brain, flygym body.

    flygym renders      ->  get_ommatidia_readouts()   721 per eye
    bridge joins        ->  ~23,720 columnar cells
    connectome runs     ->  166,700 neurons, dt 5 ms
    muscles read out    ->  24 joint commands, 6 legs x 4 joints
    flygym integrates   ->  50 physics steps at 1e-4 s
    the view changes    ->  back to the top

Nothing between the eye and the muscles is designated.

Timescales divide exactly, which is the whole of mapping 3: MuJoCo's timestep
is 1e-4 s and the rate model's is 5 ms, so one brain step is fifty physics
steps.  The brain's command is held constant across them -- zero-order hold,
the same arrangement Stage 1 used between the neural cycle and the body.

flygym's leg actuators are POSITION actuators: the input is a target angle.
So the motor readout enters as a DEVIATION from the neutral pose, which is
also the honest reading -- motor neuron drive moves a joint away from rest,
it does not specify an absolute angle.  `actuator_gain` converts our
dimensionless readout into radians and has to be calibrated against the body.

ponytail: the readout drives each joint's PITCH dof only, which is the
flexion/extension axis our antagonist pairs actually name.  Roll and yaw stay
at neutral; wire them when a muscle pair is found that means them.
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
class LoopTrace:
    """What happened, per brain step."""

    joint_commands: List[Dict[str, float]] = field(default_factory=list)
    position: List[np.ndarray] = field(default_factory=list)
    lum_mean: List[float] = field(default_factory=list)
    lum_std: List[float] = field(default_factory=list)

    def as_arrays(self):
        return {"position": np.array(self.position),
                "lum_mean": np.array(self.lum_mean),
                "lum_std": np.array(self.lum_std)}


def build_fly(name: str = "nmf", kp: float = 30.0):
    """A NeuroMechFly with legs that move and eyes that see.

    Both have to be asked for: a bare `NeuroMechFly` has no joints, no
    actuators and no eye cameras.
    """
    from flygym.compose import NeuroMechFly
    from flygym.anatomy import Skeleton, AxisOrder, ActuatedDOFPreset, JointPreset
    from flygym.compose.pose import KinematicPosePreset

    fly = NeuroMechFly(name=name)
    skel = Skeleton(axis_order=AxisOrder.ROLL_PITCH_YAW,
                    joint_preset=JointPreset.LEGS_ACTIVE_ONLY)
    fly.add_joints(skel, KinematicPosePreset.NEUTRAL)
    dofs = skel.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(dofs, "position", KinematicPosePreset.NEUTRAL, kp=kp)
    fly.add_vision()
    return fly


class ConnectomeFlyLoop:
    """Drive a flygym body with the connectome, and feed its view back."""

    def __init__(self, net, ann, lattice, omm_index, sim, fly,
                 drive_gain: float = 0.20, actuator_gain: float = 0.30):
        self.net, self.ann, self.lat = net, ann, lattice
        self.omm = omm_index          # column -> ommatidium, from flygym_bridge
        self.sim, self.fly = sim, fly
        self.drive_gain = drive_gain
        self.actuator_gain = actuator_gain
        self.v = net.init_state(1)

        import pandas as pd
        idx = pd.Index(net.ids).get_indexer(lattice.root_id)
        self.rows = idx
        self.valid = idx >= 0
        self.eye_is_left = lattice.eye == "left"
        self._keys, self._neutral = self._build_actuator_map()

    def _visual_drive(self):
        """Ommatidia readouts -> a drive vector over the whole network."""
        readouts = self.sim.get_ommatidia_readouts(self.fly.name)   # (2, n, 2)
        # the two channels are yellow/pale and only one is non-zero per
        # ommatidium, so summing them recovers that ommatidium's reading
        per_eye = readouts.sum(axis=-1)
        lum = np.where(self.eye_is_left,
                       per_eye[0][self.omm], per_eye[1][self.omm])

        drive = torch.zeros(self.net.n, 1)
        sel = self.valid
        drive[self.rows[sel], 0] = torch.from_numpy(
            (lum[sel] * self.drive_gain).astype(np.float32))
        return drive, float(lum[sel].mean()), float(lum[sel].std())

    def step(self, trace: Optional[LoopTrace] = None) -> Dict[str, float]:
        """One brain step, then the physics steps it covers."""
        from decoder.steering import joint_commands
        from core.flywire_rate import activity

        drive, lmean, lstd = self._visual_drive()
        with torch.no_grad():
            self.v = self.net.step(self.v, drive, BRAIN_DT_MS)
            rates = activity(self.v).numpy().ravel()

        cmds = joint_commands(rates, self.ann)
        self._apply(cmds)
        for _ in range(STEPS_PER_BRAIN_STEP):
            self.sim.step()

        if trace is not None:
            trace.joint_commands.append(cmds)
            pos = self.sim.get_body_positions(self.fly.name)
            trace.position.append(np.asarray(pos).mean(axis=0))
            trace.lum_mean.append(lmean)
            trace.lum_std.append(lstd)
        return cmds

    def _apply(self, cmds: Dict[str, float]) -> None:
        """Motor readout as a deviation from the neutral pose."""
        from flygym.compose.fly.base_fly import ActuatorType

        vec = self._neutral.copy()
        for i, key in enumerate(self._keys):
            if key is not None:
                vec[i] += self.actuator_gain * cmds.get(key, 0.0)
        self.sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, vec)

    def _build_actuator_map(self):
        """Which actuator each joint command drives, and the neutral angles.

        Actuator dof names are `parent-child-axis`, e.g.
        `lf_trochanterfemur-lf_tibia-pitch`.  Our command keys are the child
        segment, so the join is exact -- no substring guessing.
        """
        from flygym.compose.fly.base_fly import ActuatorType

        dofs = self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        neutral = self.fly.jointdof_to_neutralaction_by_type[ActuatorType.POSITION]
        keys = []
        for d in dofs:
            parts = d.name.split("-")
            keys.append(parts[1] if len(parts) == 3 and parts[2] == "pitch"
                        else None)
        return keys, np.array([neutral[d] for d in dofs], dtype=float)


def demo(n_steps: int = 20) -> None:
    """Close the loop for a few brain steps and check it actually moves."""
    import sys
    import time

    sys.path.insert(0, "src")
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from flygym.vision.retina import Retina
    from flygym import Simulation

    import sensors.flywire_eye as eye
    import sensors.flygym_bridge as bridge
    from core.malecns import load_malecns
    from core.flywire_rate import FlyWireRate
    from decoder.steering import joint_commands

    assert STEPS_PER_BRAIN_STEP == 50, STEPS_PER_BRAIN_STEP

    t0 = time.perf_counter()
    ids, out, ann, _ = load_malecns(w_scale=0.005, symmetrise=True)
    net = FlyWireRate(out_csr=out, ids=ids, ann=ann)
    lat = eye.load_malecns_eye(ann)
    omm, err = bridge.join(lat, Retina())
    print("brain %d neurons, %d columns joined (%.1f s)"
          % (net.n, len(omm), time.perf_counter() - t0))

    world = FlatGroundWorld()
    fly = build_fly()
    world.add_fly(fly, spawn_position=(0.0, 0.0, 0.5),
                  spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    print("body: %d actuators, timestep %.0e s"
          % (sim.mj_model.nu, sim.mj_model.opt.timestep))
    assert sim.mj_model.nu == 42, sim.mj_model.nu

    loop = ConnectomeFlyLoop(net, ann, lat, omm, sim, fly)
    matched = {k for k in loop._keys if k is not None}
    reachable = matched & set(joint_commands(np.zeros(net.n), ann))
    print("  %d of 24 joint commands reach an actuator" % len(reachable))
    assert len(reachable) == 24, sorted(matched)

    trace = LoopTrace()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        loop.step(trace)
    el = time.perf_counter() - t0
    bio = n_steps * BRAIN_DT_MS / 1000.0
    print("%d brain steps (%.0f ms biological) in %.1f s -> %.0f s per "
          "biological second" % (n_steps, n_steps * BRAIN_DT_MS, el, el / bio))

    a = trace.as_arrays()
    moved = np.linalg.norm(a["position"][-1] - a["position"][0])
    print("  body moved %.3f mm" % moved)
    print("  luminance mean %.4f..%.4f, spatial std %.4f..%.4f"
          % (a["lum_mean"].min(), a["lum_mean"].max(),
             a["lum_std"].min(), a["lum_std"].max()))
    cmd = np.array([list(c.values()) for c in trace.joint_commands])
    print("  joint commands %.4f..%.4f, per-step change %.2e"
          % (cmd.min(), cmd.max(), np.abs(np.diff(cmd, axis=0)).mean()))
    assert len(trace.joint_commands) == n_steps
    assert a["lum_std"].max() > 0, "the eye sees a uniform field"
    print("demo ok")


if __name__ == "__main__":
    demo()
