"""Stage 1 control: does the sweep result survive with noise turned ON?

The main sweep runs noise-free (handoff doc section 24, to isolate the body
effect).  That protocol choice turns out to decide one of the headline
results, so it needs its own control:

  1. With noise ON, does the ideal body still fail at e0 = +-90 deg?
     (Stage 0 finding F1, the period-2 limit cycle.)
     -> If not, the "rescued by body" result is an artifact of running
        noise-free, because there is nothing left to rescue.

  2. With noise ON, does the r_max x tau_r structure still hold?

Usage:
    python experiments/stage1_noise_control.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from body.ideal_yaw import IdealYawBody
from body.yaw_plant import YawPlant, from_neural_scale
from core.calibration import load
from core.westeinde2024 import CoreParams, WesteindeSteeringCore
from decoder.steering import SteeringDecoder
from environment.heading_task import (CALIBRATION_ERRORS_DEG, HeadingTask,
                                      TEST_ERRORS_DEG)
from eval import metrics as mx
from eval.feasibility import is_task_feasible
from plugins.passthrough import PassthroughPlugin
from plugins.rate_clip import RateClipPlugin
from sensors.ideal_heading import IdealHeadingSensor
from sim.closed_loop import run_closed_loop
from sim.noise import command_noise_rad_per_s

SEEDS = range(8)
ERRORS = list(CALIBRATION_ERRORS_DEG) + list(TEST_ERRORS_DEG)
CRIT = mx.SuccessCriterion()
DURATION = 15.0


def noise_for(task, seed):
    return command_noise_rad_per_s(task.n_cycles, task.T_core_s, seed)


def main() -> int:
    norm = load(REPO / "configs" / "norm_constants.json")
    R = json.loads((REPO / "configs" / "neural_command_scale.json")
                   .read_text(encoding="utf-8"))["body_scale_rad_per_s"]
    core = WesteindeSteeringCore(CoreParams(), norm)
    dec = SteeringDecoder()
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "seeds": len(list(SEEDS)), "initial_errors_deg": ERRORS}

    # ---- 1. does F1 survive noise? --------------------------------------
    print("=== 1. ideal body with noise ON ===")
    baseline = {}
    for e0 in ERRORS:
        ok = 0
        for s in SEEDS:
            t = HeadingTask(initial_heading=np.deg2rad(e0), duration_s=DURATION)
            r = run_closed_loop(core, dec, PassthroughPlugin(), IdealYawBody(),
                                IdealHeadingSensor(), t, command_noise=noise_for(t, s))
            ok += mx.compute(r, CRIT).success
        baseline[e0] = ok / len(list(SEEDS))
        print("  e0=%+7.1f  success %.2f" % (e0, baseline[e0]))
    out["ideal_body_success_with_noise"] = baseline
    f1_survives = any(v < 1.0 for v in baseline.values())
    print("  F1 (ideal-body failure) survives noise: %s" % f1_survives)
    out["f1_survives_noise"] = bool(f1_survives)

    # ---- 2. does the r_max x tau structure survive? ---------------------
    print("=== 2. success rate, noise OFF vs ON ===")
    print("  %8s %6s | %10s %10s" % ("r_max/R", "tau/T", "noise OFF", "noise ON"))
    grid = {}
    for rr in (0.01, 0.05, 0.25, 1.0):
        for tr in (1.0, 8.0, 32.0):
            p = from_neural_scale(R, rr, 4.0, tr)
            res = {}
            for label, seeds in (("off", [None]), ("on", list(SEEDS))):
                ok = n = 0
                for e0 in ERRORS:
                    if not is_task_feasible(p, np.deg2rad(e0), CRIT.timeout_s,
                                            CRIT.tolerance_rad).feasible:
                        continue
                    for s in seeds:
                        t = HeadingTask(initial_heading=np.deg2rad(e0),
                                        duration_s=DURATION)
                        pl, b = RateClipPlugin(r_max=p.r_max), YawPlant(p)
                        r = run_closed_loop(core, dec, pl, b, IdealHeadingSensor(),
                                            t, command_noise=None if s is None
                                            else noise_for(t, s))
                        ok += mx.compute(r, CRIT, plugin=pl, body=b).success
                        n += 1
                res[label] = ok / n if n else float("nan")
            grid["r%g_tau%g" % (rr, tr)] = res
            print("  %8.2f %6.0f | %10.2f %10.2f" % (rr, tr, res["off"], res["on"]))
    out["success_grid"] = grid

    # ---- verdict ---------------------------------------------------------
    # r_max alone, at the shortest lag: if these are all equal, the apparent
    # "faster body is worse" main effect is really an r_max x tau interaction.
    at_tau1 = [grid["r%g_tau1" % rr]["on"] for rr in (0.01, 0.05, 0.25, 1.0)]
    interaction_only = max(at_tau1) - min(at_tau1) < 0.05
    out["r_max_effect_is_interaction_only"] = bool(interaction_only)
    print("=== verdict ===")
    print("  rescue result valid:            %s" % f1_survives)
    print("  r_max effect is interaction only: %s (spread at tau=T is %.2f)"
          % (interaction_only, max(at_tau1) - min(at_tau1)))

    d = REPO / "results" / "stage1_noise_control"
    d.mkdir(parents=True, exist_ok=True)
    (d / "noise_control.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("  written -> %s" % (d / "noise_control.json").relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
