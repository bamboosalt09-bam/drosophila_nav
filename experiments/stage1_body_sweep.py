"""Stage 1, step 2: body-constraint sweep (handoff doc sections 16, 17, 27-30).

Condition A (ideal body, passthrough plugin) versus condition B (constrained
yaw plant, rate-clip plugin), with the neural core and the decoder FROZEN and
identical in every cell of the sweep.  That is the whole point: anything that
changes between cells is body, not brain.

Sweep axes are dimensionless (section 17), anchored to
R_max = kappa*max|steering|.  The RANGES differ from the document's proposal
because a probe showed the proposed ones sit entirely inside the tolerant
region; see the R_MAX_RATIOS comment below.

Every trial is classified before it is counted (section 27).  Three classes
are kept out of the body-tolerance success rate:

    BODY_INFEASIBLE   the plant provably cannot turn far enough in time
    BASELINE_FAILURE  the ideal body already failed from this initial heading
                      (Stage 0 finding F1: the period-2 limit cycle)
    RESCUED_BY_BODY   the ideal body failed here but the CONSTRAINED body
                      succeeded -- a result in its own right, reported
                      separately rather than averaged into anything

Noise is off (section 24) so that the body effect is isolated.

Usage:
    python experiments/stage1_body_sweep.py
    python experiments/stage1_body_sweep.py --quick --tag quick
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from body.ideal_yaw import IdealYawBody
from body.yaw_plant import YawPlant, from_neural_scale
from core.calibration import calibrate, load
from core.westeinde2024 import CoreParams, WesteindeSteeringCore
from decoder.steering import SteeringDecoder
from environment.heading_task import (CALIBRATION_ERRORS_DEG, HeadingTask,
                                      TEST_ERRORS_DEG)
from eval import classify as cls
from eval import metrics as mx
from eval.feasibility import is_task_feasible
from plugins.passthrough import PassthroughPlugin
from plugins.rate_clip import RateClipPlugin
from sensors.ideal_heading import IdealHeadingSensor
from sim.closed_loop import run_closed_loop

NORM_JSON = REPO / "configs" / "norm_constants.json"
SCALE_JSON = REPO / "configs" / "neural_command_scale.json"

# Handoff document section 17 proposes ratios of 0.25 - 4 on all three axes.
# Measured against this model those are all far inside the tolerant region:
# at the most constrained corner of that grid the feasibility margin is still
# 40-85x, because R_max = 2000 deg/s is ~100x what the task needs (180 deg in
# 10 s is 18 deg/s on average).  The grid below was chosen from a probe that
# located the actual boundaries; it keeps the document's dimensionless scheme
# and only moves the ranges to where the transitions are.
R_MAX_RATIOS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.25, 1.0)
ALPHA_RATIOS = (0.005, 0.01, 0.02, 0.05, 0.25, 4.0)
TAU_RATIOS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "no-commit"
    except Exception:
        return "unavailable"


def build_parser():
    p = argparse.ArgumentParser(description="Stage 1 body-constraint sweep")
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--tolerance-deg", type=float, default=15.0)
    p.add_argument("--hold", type=float, default=1.0)
    p.add_argument("--quick", action="store_true",
                   help="3x3x3 grid and calibration errors only")
    p.add_argument("--tag", default="default")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    outdir = REPO / "results" / "stage1_sweep" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    norm = load(NORM_JSON) if NORM_JSON.exists() else calibrate(CoreParams(),
                                                                verbose=False)
    scale = json.loads(SCALE_JSON.read_text(encoding="utf-8"))
    R = float(scale["body_scale_rad_per_s"])

    core = WesteindeSteeringCore(CoreParams(), norm)
    decoder = SteeringDecoder()
    T = decoder.T_core_s
    crit = mx.SuccessCriterion(tolerance_rad=np.deg2rad(args.tolerance_deg),
                               timeout_s=args.timeout, hold_s=args.hold)

    if args.quick:
        r_ratios, a_ratios, t_ratios = (0.005, 0.02, 0.25), (0.01, 0.05, 4.0), (1.0, 8.0, 32.0)
        errors = list(CALIBRATION_ERRORS_DEG)
    else:
        r_ratios, a_ratios, t_ratios = R_MAX_RATIOS, ALPHA_RATIOS, TAU_RATIOS
        errors = list(CALIBRATION_ERRORS_DEG) + list(TEST_ERRORS_DEG)

    combos = list(itertools.product(r_ratios, a_ratios, t_ratios))
    print("=== Stage 1 step 2: body-constraint sweep (tag: %s) ===" % args.tag)
    print("  body scale R_max = %.1f deg/s, T_core = %.2f s" % (np.rad2deg(R), T))
    print("  %d body conditions x %d initial errors = %d trials"
          % (len(combos), len(errors), len(combos) * len(errors)))
    print("  success: |e| <= %.0f deg within %.0f s, held %.0f s; noise off"
          % (args.tolerance_deg, args.timeout, args.hold))

    # ---- condition A baseline, per initial error -------------------------
    baseline = {}
    for e0 in errors:
        task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(e0),
                           duration_s=args.duration)
        res = run_closed_loop(core, decoder, PassthroughPlugin(), IdealYawBody(),
                              IdealHeadingSensor(), task)
        m = mx.compute(res, crit)
        baseline[e0] = {"success": m.success, "settling_s": m.settling_time_s,
                        "iae": m.iae_deg_s, "zero_crossings": m.zero_crossings}
    n_base_ok = sum(b["success"] for b in baseline.values())
    print("  condition A (ideal body): %d / %d initial errors succeed"
          % (n_base_ok, len(errors)))
    failed_e0 = [e for e, b in baseline.items() if not b["success"]]
    if failed_e0:
        print("    already failing with a perfect body (finding F1): %s"
              % sorted(failed_e0))

    # ---- sweep -----------------------------------------------------------
    rows = []
    t0 = time.perf_counter()
    for i, (rr, ar, tr) in enumerate(combos):
        params = from_neural_scale(R, r_max_ratio=rr, alpha_max_ratio=ar,
                                   tau_ratio=tr, T_core=T)
        for e0 in errors:
            task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(e0),
                               duration_s=args.duration)
            plugin = RateClipPlugin(r_max=params.r_max)
            body = YawPlant(params)
            res = run_closed_loop(core, decoder, plugin, body,
                                  IdealHeadingSensor(), task)
            m = mx.compute(res, crit, plugin=plugin, body=body)
            feas = is_task_feasible(params, np.deg2rad(e0), args.timeout,
                                    crit.tolerance_rad)
            outcome = cls.classify(m, feas, e0, baseline[e0]["success"],
                                   settle_grace_s=args.duration)

            row = {"r_max_ratio": rr, "alpha_ratio": ar, "tau_ratio": tr,
                   "e0_deg": e0, "outcome": outcome.value,
                   "feasible": feas.feasible, "feas_margin": feas.margin,
                   "baseline_success": baseline[e0]["success"],
                   "baseline_settling_s": baseline[e0]["settling_s"],
                   **m.as_dict()}
            # success at the other tolerances, from the same trajectory
            for tol in (10.0, 20.0):
                alt = mx.compute(res, mx.SuccessCriterion(
                    tolerance_rad=np.deg2rad(tol), timeout_s=args.timeout,
                    hold_s=args.hold))
                row["success_tol%d" % int(tol)] = alt.success
            rows.append(row)

        if (i + 1) % max(1, len(combos) // 10) == 0:
            el = time.perf_counter() - t0
            print("  ... %d/%d body conditions (%.0f s elapsed, ~%.0f s left)"
                  % (i + 1, len(combos), el, el * (len(combos) - i - 1) / (i + 1)))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sweep_trials.csv", index=False)
    print("--- %d trials in %.0f s ---" % (len(df), time.perf_counter() - t0))

    # ---- outcome composition --------------------------------------------
    counts = df["outcome"].value_counts()
    print("--- outcome composition (all trials) ---")
    for name, n in counts.items():
        print("  %-26s %5d  (%.1f%%)" % (name, n, 100 * n / len(df)))

    n_rescued = int((df["outcome"] == cls.Outcome.RESCUED_BY_BODY.value).sum())
    n_baseline_fail = int(df["baseline_success"].eq(False).sum())
    if n_baseline_fail:
        print("  RESCUED BY THE BODY: %d / %d trials where the IDEAL body "
              "failed (%.1f%%)" % (n_rescued, n_baseline_fail,
                                   100 * n_rescued / n_baseline_fail))

    attributable = df[~df["outcome"].isin(
        [o.value for o in cls.NOT_ATTRIBUTABLE] + [cls.Outcome.RESCUED_BY_BODY.value])]
    print("  trials usable for body-tolerance claims: %d / %d (%.1f%%)"
          % (len(attributable), len(df), 100 * len(attributable) / len(df)))

    # ---- success rate per body condition --------------------------------
    def rate(sub):
        """Success rate over trials the body could do and the ideal body did do."""
        att = sub[~sub["outcome"].isin(
            [o.value for o in cls.NOT_ATTRIBUTABLE]
            + [cls.Outcome.RESCUED_BY_BODY.value])]
        if len(att) == 0:
            return np.nan
        return float(att["outcome"].isin([o.value for o in cls.SUCCESSFUL]).mean())

    grid = (df.groupby(["r_max_ratio", "alpha_ratio", "tau_ratio"])
              .apply(rate, include_groups=False).rename("success_rate").reset_index())
    grid.to_csv(outdir / "success_rate_grid.csv", index=False)

    print("--- success rate over ATTRIBUTABLE trials, by axis ---")
    for axis in ("r_max_ratio", "alpha_ratio", "tau_ratio"):
        agg = grid.groupby(axis)["success_rate"].mean()
        print("  %-12s %s" % (axis, "  ".join("%.2f:%.2f" % (k, v)
                                              for k, v in agg.items())))

    # which axis separates success from failure most strongly (section 30)
    spread = {axis: float(grid.groupby(axis)["success_rate"].mean().max()
                          - grid.groupby(axis)["success_rate"].mean().min())
              for axis in ("r_max_ratio", "alpha_ratio", "tau_ratio")}
    worst = max(spread, key=spread.get)
    print("  axis with the largest effect: %s (range %.2f)" % (worst, spread[worst]))

    # ---- figures ---------------------------------------------------------
    def heat(ax, piv, title, xlabel, ylabel):
        im = ax.imshow(piv.values, origin="lower", vmin=0, vmax=1,
                       cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([("%g" % c) for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([("%g" % r) for r in piv.index])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        for (yi, xi), v in np.ndenumerate(piv.values):
            if np.isfinite(v):
                ax.text(xi, yi, "%.2f" % v, ha="center", va="center", fontsize=7)
        return im

    pairs = [(("tau_ratio", "r_max_ratio"), "alpha_ratio"),
             (("tau_ratio", "alpha_ratio"), "r_max_ratio"),
             (("alpha_ratio", "r_max_ratio"), "tau_ratio")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, ((xa, ya), marg) in zip(axes, pairs):
        piv = grid.pivot_table(index=ya, columns=xa, values="success_rate",
                               aggfunc="mean")
        im = heat(ax, piv, "marginalised over %s" % marg,
                  xa + "  (x T or x R_max/T)", ya + "  (x R_max)")
    fig.colorbar(im, ax=axes, shrink=0.85, label="success rate")
    fig.suptitle("Stage 1 / sweep: heading-recovery success of the FIXED fly core "
                 "vs body limits")
    fig.savefig(outdir / "fig1_phase_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    order = [o.value for o in (cls.Outcome.SUCCESS, cls.Outcome.SUCCESS_NEAR_INFEASIBLE,
                               cls.Outcome.RESCUED_BY_BODY,
                               cls.Outcome.SLOW_BUT_STABLE,
                               cls.Outcome.SATURATION_DOMINATED,
                               cls.Outcome.PERSISTENT_OSCILLATION,
                               cls.Outcome.OVERSHOOT_DOMINANT,
                               cls.Outcome.CONTROLLER_INSTABILITY,
                               cls.Outcome.FAILURE_OTHER,
                               cls.Outcome.BODY_INFEASIBLE,
                               cls.Outcome.BASELINE_FAILURE)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, axis in zip(axes, ("r_max_ratio", "alpha_ratio", "tau_ratio")):
        comp = (df.groupby([axis, "outcome"]).size().unstack(fill_value=0))
        comp = comp.reindex(columns=[c for c in order if c in comp.columns])
        comp = comp.div(comp.sum(axis=1), axis=0)
        bottom = np.zeros(len(comp))
        for col in comp.columns:
            ax.bar([str(i) for i in comp.index], comp[col].values, bottom=bottom,
                   label=col)
            bottom += comp[col].values
        ax.set_xlabel(axis)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("fraction of trials")
    axes[-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Stage 1 / sweep: outcome composition (classified before counting)")
    fig.savefig(outdir / "fig2_outcome_composition.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- provenance ------------------------------------------------------
    prov = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "experiments/stage1_body_sweep.py",
        "git_commit": git_commit(),
        "core_params": CoreParams().as_dict(),
        "decoder": decoder.as_dict(),
        "body_scale_rad_per_s": R,
        "body_scale_deg_per_s": float(np.rad2deg(R)),
        "grid": {"r_max_ratio": list(r_ratios), "alpha_ratio": list(a_ratios),
                 "tau_ratio": list(t_ratios)},
        "initial_errors_deg": errors,
        "success_criterion": crit.as_dict(),
        "tolerance_sensitivity_deg": [10.0, 15.0, 20.0],
        "noise": "none (doc section 24: isolate the body effect)",
        "classification_thresholds": cls.thresholds_as_dict(),
        "baseline_condition_A": {str(k): v for k, v in baseline.items()},
        "outcome_counts": {str(k): int(v) for k, v in counts.items()},
        "attributable_fraction": float(len(attributable) / len(df)),
        "n_rescued_by_body": n_rescued,
        "n_baseline_failures": n_baseline_fail,
        "axis_effect_range": spread,
        "largest_effect_axis": worst,
        "n_trials": int(len(df)),
    }
    (outdir / "provenance.json").write_text(json.dumps(prov, indent=2),
                                            encoding="utf-8")

    print("--- written to %s ---" % outdir)
    for f in sorted(outdir.iterdir()):
        print("  " + f.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
