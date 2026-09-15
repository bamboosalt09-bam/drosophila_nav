"""3D viewer: replays a LoopResult.  Reads simulation output, changes nothing.

Draws what the study is about (handoff doc section 40): where the body points,
where the goal is, what the brain asked for, and what the body actually did.
In Stage 0 the last two coincide; with a constrained body they separate, and
that gap is the picture.

    PYTHONPATH=src python -m render.viewer3d          # writes a demo gif

# ponytail: clears and redraws the axes every frame, ~25 ms each.  Fine for a
# few hundred frames; switch to blitting only if a long replay gets annoying.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def _arrow(ax, angle, length, color, label, lw=3):
    ax.quiver(0, 0, 0, length * np.cos(angle), length * np.sin(angle), 0,
              color=color, lw=lw, arrow_length_ratio=0.15, label=label)


def animate(result, path, T_core=0.1, fps=10, stride=1):
    """Render `result` (a sim.closed_loop.LoopResult) to a gif at `path`."""
    psi, goal = result.psi, result.goal
    frames = range(0, len(result.steering), stride)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw(k):
        ax.clear()
        # where the brain wants to be one cycle from now, and where the body
        # will actually be: the same arrow in Stage 0, two arrows with a body
        _arrow(ax, goal, 1.25, "C2", "goal", lw=2)
        _arrow(ax, psi[k] + result.r_brain[k] * T_core, 1.0, "C3", "brain wants")
        _arrow(ax, psi[k] + result.r[k + 1] * T_core, 1.0, "C0", "body will do")
        _arrow(ax, psi[k], 0.85, "k", "heading", lw=4)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(-0.05, 0.35)
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 0.35))
        ax.view_init(elev=26, azim=-62)
        ax.legend(loc="upper left", fontsize=8)
        err = np.rad2deg(np.arctan2(np.sin(psi[k] - goal), np.cos(psi[k] - goal)))
        ax.set_title(
            "t = %5.1f s   error = %+7.1f deg\n"
            "brain %+8.0f deg/s    body %+8.0f deg/s"
            % (k * T_core, err, np.rad2deg(result.r_brain[k]),
               np.rad2deg(result.r[k + 1])), fontsize=10, family="monospace")

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 // fps)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path, len(list(frames))


if __name__ == "__main__":
    import json
    import sys

    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO / "src"))

    from body.yaw_plant import YawPlant, from_neural_scale
    from core.calibration import load
    from core.westeinde2024 import CoreParams, WesteindeSteeringCore
    from decoder.steering import SteeringDecoder
    from environment.heading_task import HeadingTask
    from plugins.rate_clip import RateClipPlugin
    from sensors.ideal_heading import IdealHeadingSensor
    from sim.closed_loop import run_closed_loop

    norm = load(REPO / "configs" / "norm_constants.json")
    R = json.loads((REPO / "configs" / "neural_command_scale.json")
                   .read_text(encoding="utf-8"))["body_scale_rad_per_s"]
    core = WesteindeSteeringCore(CoreParams(), norm)
    dec = SteeringDecoder()

    # a body slow enough that brain and body visibly disagree
    p = from_neural_scale(R, r_max_ratio=0.25, alpha_max_ratio=0.5, tau_ratio=2.0)
    task = HeadingTask(initial_heading=np.deg2rad(120.0), duration_s=6.0)
    res = run_closed_loop(core, dec, RateClipPlugin(r_max=p.r_max), YawPlant(p),
                          IdealHeadingSensor(), task)

    out, n = animate(res, REPO / "results" / "render" / "demo_constrained.gif")
    assert out.exists() and out.stat().st_size > 0, "no gif written"
    assert n == task.n_cycles, (n, task.n_cycles)
    print("wrote %s (%d frames, %.1f MB)"
          % (out, n, out.stat().st_size / 1e6))
