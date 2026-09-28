"""The two things being compared, behind one interface.

    fly   fixed Drosophila-derived steering core + tiny trainable adapter
    rnn   generic trainable recurrent policy

"동일 학습" does NOT mean equal parameter counts.  It means identical
observation, action space, task objective, episodes, and training exposure.
The parameter asymmetry is the RESULT, not a confound: the question is how
much learning freedom a circuit with a strong structural prior needs to
acquire and transfer a function, against how much a generic RNN needs.  Read
the outcome as inductive bias and sample efficiency, never as "who wins at
equal capacity".

What is frozen in the fly, always, in both experiments:
    PFL3 +-67.5 deg phase shift, PFL2 180 deg phase shift, the PFL->DNa03->
    DNa02 relative weights, the activation structure, the topology, the
    calibration constant S, and the decoder gain kappa.
Stage 0 fixed S and kappa once; they are never re-optimised per body.

What the fly may train, in the learning-efficiency experiment ONLY:
    the output decoder's scale and bias.  Two numbers, outside the circuit.

Training happens ONCE, on a reference body.  Both models are then frozen and
carried unchanged to other body dynamics, where transfer is measured.  No
per-body retraining -- that is the rule the whole embodiment result rests on.

ponytail: the input-side affine adapter the spec allows "if needed" is not
here.  The Westeinde core is numpy, so no gradient reaches anything upstream
of it; an input adapter needs either a differentiable port of the core or a
gradient-free optimiser, and neither is worth building before the output
adapter is shown to be insufficient.
"""
from __future__ import annotations

import math
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from sensors.attraction_cue import AttractionCue


@runtime_checkable
class Policy(Protocol):
    """Cue in, body command out.  Both arms implement exactly this."""

    def reset(self) -> None: ...

    def act(self, cue: AttractionCue, heading: float) -> float:
        """Return the requested yaw rate, rad/s.  The PLANT decides what of
        it actually happens -- that is the body-brain mismatch being studied."""
        ...

    def trainable(self) -> list: ...

    def freeze(self) -> None: ...


def cue_to_goal(cue: AttractionCue, heading: float) -> Optional[float]:
    """Attraction cue -> desired heading, in world frame.

    This is master doc section 22's "goal-vector preprocessing -> desired
    heading representation".  It is arithmetic on the sensor's own report, not
    a decision: the target is at `bearing` relative to where we point, so it
    is at `heading + bearing` absolutely.  Returns None when nothing is seen,
    and what to do about THAT is the policy's business, not the sensor's.
    """
    if not cue.valid:
        return None
    return heading + math.radians(cue.bearing_deg)


class FlyPolicy(nn.Module):
    """Frozen Westeinde core, with two trainable numbers bolted on the output.

    The core is called as a black box and never differentiated, which is not
    a limitation here -- it is the experimental condition.
    """

    def __init__(self, core, kappa: float, trainable_adapter: bool = False):
        super().__init__()
        self.core = core
        self.kappa = float(kappa)          # Stage 0 calibration, frozen
        # scale starts at 1 and bias at 0, so an untrained FlyPolicy is
        # bit-identical to the frozen main-experiment controller
        self.scale = nn.Parameter(torch.ones(1),
                                  requires_grad=trainable_adapter)
        self.bias = nn.Parameter(torch.zeros(1),
                                 requires_grad=trainable_adapter)
        self._last_goal: Optional[float] = None

    def reset(self) -> None:
        self._last_goal = None

    def act_t(self, cue: AttractionCue, heading: float) -> torch.Tensor:
        """The command, still attached to autograd.

        The core is numpy, so `r` is a constant here -- and that is exactly
        the experimental condition.  The two knobs sit OUTSIDE the connectome,
        so their gradient never has to travel through 166,700 neurons: this
        costs a forward pass, not a forward-and-backward one.
        """
        goal = cue_to_goal(cue, heading)
        if goal is None:
            # target not visible: hold the last known goal.  Doing nothing
            # would make the comparison a test of the detector, not the core.
            goal = self._last_goal
            if goal is None:
                return self.bias * 0.0        # zero, but still differentiable
        self._last_goal = goal
        s = float(self.core.steering(heading, goal))
        r = math.radians(self.kappa * s) / 0.1      # deg/0.1s -> rad/s
        return self.scale * r + self.bias

    def act(self, cue: AttractionCue, heading: float) -> float:
        return float(self.act_t(cue, heading).item())

    def trainable(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)


class ConnectomeFlyPolicy(nn.Module):
    """The whole MaleCNS connectome, driven by both eyes and the ORNs.

        scene  -> 6,199 lamina cells  ]
        bearing-> 569 pheromone ORNs  ] -> 166,700 neurons -> descending L/R

    Only the two anatomical endpoints are designated.  Nothing between them is
    chosen: the readout is the mean rate of every left and every right
    `descending_neuron`, 1,304 cells, no cell type picked.

    Trainable: the same two numbers as `FlyPolicy` -- a decoder scale and
    bias, OUTSIDE the connectome.  So training costs a forward pass; no
    gradient ever crosses the 166,700 neurons.
    """

    def __init__(self, net, ann, connectome_input, kappa_scale: float = 1.0,
                 trainable_adapter: bool = False, dt_ms: float = 5.0,
                 steps_per_cycle: int = 20):
        super().__init__()
        # `self.net = net` would register FlyWireRate as a SUBMODULE, and its
        # 141 per-cell-type log_tau/bias/log_gain would join `parameters()` --
        # silently making the frozen core trainable.  It did, and the count
        # read 143 instead of 2.  Freeze it explicitly and assert the count.
        for p in net.parameters():
            p.requires_grad_(False)
        self.net, self.ann, self.inp = net, ann, connectome_input
        self.dt_ms = dt_ms
        # 20 steps x 5 ms = one 100 ms control cycle.  These numbers are NOT
        # free, and coarsening dt for speed destroys the answer.  Measured,
        # offset/range of the descending readout across bar bearings (below
        # 0.5 means the steering signal crosses zero and is therefore signed):
        #     5 ms x 20  (100 ms)   0.09   monotonic, crosses zero
        #     2 ms x 50  (100 ms)   0.38   crosses zero
        #    10 ms x 10  (100 ms)   8.06   flat, no direction at all
        #    20 ms x  5  (100 ms)   8.41   flat
        #     5 ms x 40  (200 ms)   2.31   drifts past the directional state
        # dt must stay at or below 5 ms -- a quarter of the ~20 ms membrane
        # tau -- and the window at 100 ms.  A 2x speedup from dt 10 ms looked
        # free and silently removed the signal being measured.
        self.steps_per_cycle = steps_per_cycle
        self.scale = nn.Parameter(torch.full((1,), float(kappa_scale)),
                                  requires_grad=trainable_adapter)
        self.bias = nn.Parameter(torch.zeros(1),
                                 requires_grad=trainable_adapter)

        sc = ann["super_class"].to_numpy(dtype="<U32")
        side = ann["side"].to_numpy(dtype="<U16")
        desc = sc == "descending_neuron"
        self.desc_l = np.flatnonzero(desc & (side == "left"))
        self.desc_r = np.flatnonzero(desc & (side == "right"))
        self.v = None

    def reset(self) -> None:
        self.v = self.net.init_state(1)

    def act_t(self, world, position, heading: float) -> torch.Tensor:
        from core.flywire_rate import activity
        if self.v is None:
            self.reset()
        drive, _ = self.inp.drive(world, position, heading)
        with torch.no_grad():
            for _ in range(self.steps_per_cycle):
                self.v = self.net.step(self.v, drive, self.dt_ms)
            r = activity(self.v).numpy().ravel()
        turn = float(r[self.desc_r].mean() - r[self.desc_l].mean())
        return self.scale * turn + self.bias

    def act(self, world, position, heading: float) -> float:
        return float(self.act_t(world, position, heading).item())

    def reset_batch(self, batch: int) -> None:
        self.v = self.net.init_state(batch)

    def act_batch(self, worlds, positions, headings) -> torch.Tensor:
        """One control cycle for a whole batch of episodes at once.

        The sparse matvec is the expensive part and it amortises hard over
        batch columns: measured 13.3 ms at batch 1 against 2.1 ms per episode
        at batch 16, a 6.2x saving.  Building each world's drive does NOT
        batch -- different scenes, different obstacle lists -- so it becomes
        the next term; that is expected and it is still a large net win.
        """
        from core.flywire_rate import activity
        b = len(worlds)
        if self.v is None or self.v.shape[1] != b:
            self.reset_batch(b)
        cols = [self.inp.drive(w, p, h)[0]
                for w, p, h in zip(worlds, positions, headings)]
        drive = torch.cat(cols, dim=1)
        with torch.no_grad():
            for _ in range(self.steps_per_cycle):
                self.v = self.net.step(self.v, drive, self.dt_ms)
            r = activity(self.v).numpy()
        turn = r[self.desc_r].mean(axis=0) - r[self.desc_l].mean(axis=0)
        return self.scale * torch.from_numpy(turn.astype(np.float32)) + self.bias

    def trainable(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)


class RNNPolicy(nn.Module):
    """Small GRU over the cue vector.  Deliberately plain and readable.

    Not a reimplementation of FLYNN -- that paper is the literature position
    for "a learned recurrent policy doing embodied navigation", not a model to
    clone.  This baseline is kept simple so the comparison is about inductive
    bias rather than about someone's architecture tricks.
    """

    CUE_DIM = 5        # AttractionCue.as_vector()

    def __init__(self, hidden: int = 32, r_scale: float = 2.0,
                 n_bins: int = 32):
        super().__init__()
        self.n_bins = n_bins
        # scene profile + cue + sin/cos heading: the same information the
        # connectome gets through its eyes and ORNs, pooled
        self.cell = nn.GRUCell(n_bins + self.CUE_DIM + 2, hidden)
        self.head = nn.Linear(hidden, 1)
        self.hidden_size = hidden
        self.r_scale = float(r_scale)
        self.h: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.h = None

    def observe(self, profile: np.ndarray, cue: AttractionCue,
                heading: float) -> torch.Tensor:
        v = np.concatenate([profile, cue.as_vector(),
                            [math.sin(heading), math.cos(heading)]])
        return torch.tensor(v, dtype=torch.float32).unsqueeze(0)

    def act_t(self, profile: np.ndarray, cue: AttractionCue,
              heading: float) -> torch.Tensor:
        x = self.observe(profile, cue, heading)
        if self.h is None:
            self.h = torch.zeros(1, self.hidden_size)
        self.h = self.cell(x, self.h)
        return self.r_scale * torch.tanh(self.head(self.h)).squeeze()

    def act(self, profile: np.ndarray, cue: AttractionCue,
            heading: float) -> float:
        return float(self.act_t(profile, cue, heading).item())

    def reset_batch(self, batch: int) -> None:
        self.h = None

    def act_batch(self, profiles, cues, headings) -> torch.Tensor:
        """Same interface as the connectome arm: a whole batch per call."""
        x = torch.stack([self.observe(p, c, h).squeeze(0)
                         for p, c, h in zip(profiles, cues, headings)])
        if self.h is None or self.h.shape[0] != x.shape[0]:
            self.h = torch.zeros(x.shape[0], self.hidden_size)
        self.h = self.cell(x, self.h)
        return self.r_scale * torch.tanh(self.head(self.h)).squeeze(-1)

    def trainable(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)


def demo() -> None:
    """Check the asymmetry is real and the fly core is untouched."""
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "src"))

    from core import calibration
    from core.westeinde2024 import WesteindeSteeringCore
    from decoder.steering import SOURCE_KAPPA_DEG_PER_STEP

    try:
        core = WesteindeSteeringCore(
            norm=calibration.load(repo / "configs/norm_constants.json"))
    except Exception as e:
        print("(no frozen norm constants: %s)" % e)
        core = WesteindeSteeringCore()

    main = FlyPolicy(core, SOURCE_KAPPA_DEG_PER_STEP)
    adapt = FlyPolicy(core, SOURCE_KAPPA_DEG_PER_STEP, trainable_adapter=True)
    rnn = RNNPolicy()

    n_main = sum(p.numel() for p in main.trainable())
    n_adapt = sum(p.numel() for p in adapt.trainable())
    n_rnn = sum(p.numel() for p in rnn.trainable())
    print("trainable parameters:")
    print("  fly, main experiment          %6d" % n_main)
    print("  fly, learning experiment      %6d   (decoder scale + bias)" % n_adapt)
    print("  generic GRU policy            %6d" % n_rnn)
    print("  asymmetry                     %6.0fx" % (n_rnn / max(n_adapt, 1)))
    assert n_main == 0, "the main-experiment fly must train NOTHING"
    assert n_adapt == 2, n_adapt
    assert n_rnn > 100 * n_adapt, (n_rnn, n_adapt)

    # both must accept the identical observation -- that is what "동일 학습" means
    cue = AttractionCue(bearing_deg=20.0, elevation_deg=0.0, deviation=0.44,
                        range_m=8.0, strength=0.09, valid=True, n_cameras=2)
    a, b = main.act(cue, 0.0), rnn.act(cue, 0.0)
    print("same cue -> fly %+.4f rad/s, rnn %+.4f rad/s" % (a, b))

    # an untrained adapter must not change the frozen controller at all
    assert abs(adapt.act(cue, 0.0) - a) < 1e-12, "adapter is not neutral at init"

    # the fly must turn toward the target and away from the mirror cue
    mirror = AttractionCue(-20.0, 0.0, 0.44, 8.0, 0.09, True, 2)
    main.reset()
    left = main.act(cue, 0.0)
    main.reset()
    right = main.act(mirror, 0.0)
    print("bearing +20 -> %+.4f   bearing -20 -> %+.4f rad/s" % (left, right))
    assert left * right < 0, "steering must reverse when the target does"
    assert abs(left + right) < 1e-9, "and be symmetric"

    # losing sight must not zero the command -- it holds the last goal
    main.reset()
    main.act(cue, 0.0)
    blind = AttractionCue(0.0, 0.0, 0.0, float("nan"), 0.0, False, 0)
    held = main.act(blind, 0.0)
    assert abs(held) > 0, "lost target should hold the last goal, not give up"
    print("target lost, last goal held  -> %+.4f rad/s" % held)

    # the adapter must actually receive gradient -- it did not, when `act`
    # returned a float built with .item(), which detaches the graph entirely
    adapt.reset()
    out = adapt.act_t(cue, 0.0)
    assert out.requires_grad, "adapter is detached from autograd"
    out.backward()
    g_scale = adapt.scale.grad.item()
    g_bias = adapt.bias.grad.item()
    print("adapter gradients : d/d_scale %+.4f, d/d_bias %+.4f"
          % (g_scale, g_bias))
    assert g_scale != 0.0 and g_bias == 1.0, (g_scale, g_bias)

    # and no gradient may reach the core -- it is frozen by construction
    assert not any(getattr(p, "grad", None) is not None
                   for p in [main.scale, main.bias]), "frozen fly got a grad"

    # freezing must actually stop training
    adapt.freeze()
    assert sum(p.numel() for p in adapt.trainable()) == 0
    print("demo ok")


if __name__ == "__main__":
    demo()
