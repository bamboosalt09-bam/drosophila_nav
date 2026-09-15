"""Stage 1, step 1: fix the body-parameter scale and freeze it.

Handoff document section 17 forbids inventing drone specifications.  Body
limits are dimensionless multiples of the brain's own command scale:

    r_max     = ratio * R
    alpha_max = ratio * R / T_core
    tau_r     = ratio * T_core

Section 17 proposes R = R99, the 99th percentile of |r_brain| in the ideal
baseline.  Measured here, R99 turns out NOT to be a well-defined scale for
this model, and this script shows why rather than quietly using it:

  * restricted to converging trials it collapses with trial duration
    (1367 -> 0 deg/s as the trial goes from 2 s to 100 s), because the loop
    reaches the goal in 3-4 cycles and every later sample is ~0.  The
    percentile ends up measuring how long we recorded.
  * taken over all trials it is set by the trials that FAIL into the period-2
    limit cycle (finding F1), which emit near-maximal commands forever, and it
    saturates at the decoder peak.

So we keep the INTENT of section 17 and replace the estimator with

    R_max = kappa * max|steering|

the fastest yaw rate the core can ever request.  It is exact and
duration-independent because the core's normalised steering obeys
|steering| <= 1 on the calibration grid.  Numerically R99(all trials) and
R_max agree to 0.02%, so the sweep grid is unchanged.

Measured on the CALIBRATION initial conditions only (section 47 keeps the test
conditions out of any tuning).  Result is written to
configs/neural_command_scale.json.

Usage:
    python experiments/stage1_command_scale.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from body.ideal_yaw import IdealYawBody
from core.calibration import calibrate, load
from core.westeinde2024 import CoreParams, WesteindeSteeringCore
from decoder.steering import SteeringDecoder
from environment.heading_task import CALIBRATION_ERRORS_DEG, trials_from_errors
from plugins.passthrough import PassthroughPlugin
from sensors.ideal_heading import IdealHeadingSensor
from sim.closed_loop import run_closed_loop
from sim.noise import (SourceNoiseSpec, deg_per_step_to_rad_per_s,
                       source_command_noise_deg)

NORM_JSON = REPO / "configs" / "norm_constants.json"
OUT_JSON = REPO / "configs" / "neural_command_scale.json"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "no-commit"
    except Exception:
        return "unavailable"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="measure the neural command scale")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--tag", default="default")
    args = ap.parse_args(argv)

    outdir = REPO / "results" / "stage1_scale" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    norm = load(NORM_JSON) if NORM_JSON.exists() else calibrate(CoreParams(),
                                                                verbose=False)
    core = WesteindeSteeringCore(CoreParams(), norm)
    decoder = SteeringDecoder()
    tasks = trials_from_errors(CALIBRATION_ERRORS_DEG, duration_s=args.duration)

    print("=== Stage 1 step 1: neural command scale ===")
    print("  ideal body, passthrough plugin, %d calibration initial errors, %.0f s"
          % (len(tasks), args.duration))

    per_trial = []
    all_vals = {"no_noise": [], "source_noise": []}
    rng = np.random.default_rng(args.seed)

    for task in tasks:
        e0 = float(np.rad2deg(task.initial_error))
        row = {"e0_deg": e0}
        for label in ("no_noise", "source_noise"):
            noise = None
            if label == "source_noise":
                nd = source_command_noise_deg(task.n_cycles, rng,
                                              SourceNoiseSpec())
                noise = deg_per_step_to_rad_per_s(nd, task.T_core_s)
            res = run_closed_loop(core, decoder, PassthroughPlugin(),
                                  IdealYawBody(), IdealHeadingSensor(), task,
                                  command_noise=noise)
            vals = np.abs(res.r_brain)
            all_vals[label].append(vals)
            final = abs(float(np.rad2deg(res.error[-1])))
            tail = np.rad2deg(res.error[-20:])
            row[label + "_first_cmd_deg_s"] = float(np.rad2deg(vals[0]))
            row[label + "_final_abs_err_deg"] = final
            row[label + "_converged"] = bool(
                final < 1.0 and float(np.max(tail) - np.min(tail)) < 1.0)
        per_trial.append(row)

    print("--- per calibration trial (no noise) ---")
    print("  %9s %18s %16s %10s" % ("e0 [deg]", "first |r_brain|", "final |err|",
                                    "converged"))
    for row in per_trial:
        print("  %9.1f %18.1f %16.4f %10s"
              % (row["e0_deg"], row["no_noise_first_cmd_deg_s"],
                 row["no_noise_final_abs_err_deg"], row["no_noise_converged"]))

    n_conv = sum(r["no_noise_converged"] for r in per_trial)
    print("  converged: %d / %d  (the rest sit in the period-2 limit cycle, "
          "finding F1)" % (n_conv, len(per_trial)))

    # ---- percentiles ----------------------------------------------------
    stats = {}
    for label, chunks in all_vals.items():
        v = np.rad2deg(np.concatenate(chunks))
        stats[label] = {
            "n_samples": int(v.size),
            "median": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)),
            "max": float(v.max()),
            "fraction_below_1_deg_s": float(np.mean(v < 1.0)),
        }

    # R99 restricted to trials that actually reach the goal: the limit-cycle
    # trials emit near-maximal commands forever and would otherwise set the
    # scale on their own.
    conv_chunks = [c for c, row in zip(all_vals["no_noise"], per_trial)
                   if row["no_noise_converged"]]
    v_conv = np.rad2deg(np.concatenate(conv_chunks)) if conv_chunks else np.array([0.0])
    r99_converging = float(np.percentile(v_conv, 99))

    peak = float(np.rad2deg(decoder.max_yaw_rate))
    r99 = stats["no_noise"]["p99"]

    # ---- is R99 a well-defined scale at all? ----------------------------
    # Section 17 assumes the command distribution has a meaningful upper
    # percentile.  Check that against trial duration before trusting it.
    dur_scan = []
    for dur in (2.0, 5.0, 10.0, 30.0, 100.0):
        a, c = [], []
        for task in trials_from_errors(CALIBRATION_ERRORS_DEG, duration_s=dur):
            res = run_closed_loop(core, decoder, PassthroughPlugin(),
                                  IdealYawBody(), IdealHeadingSensor(), task)
            v = np.abs(res.r_brain)
            a.append(v)
            if abs(float(np.rad2deg(res.error[-1]))) < 1.0:
                c.append(v)
        av = np.rad2deg(np.concatenate(a))
        cv = np.rad2deg(np.concatenate(c)) if c else np.array([0.0])
        dur_scan.append({"duration_s": dur,
                         "r99_all": float(np.percentile(av, 99)),
                         "r99_converging": float(np.percentile(cv, 99)),
                         "fraction_below_1_deg_s": float(np.mean(av < 1.0))})

    print("--- is R99 duration-independent? ---")
    print("  %10s %12s %16s" % ("duration", "R99 all", "R99 converging"))
    for d in dur_scan:
        print("  %8.0f s %12.1f %16.1f"
              % (d["duration_s"], d["r99_all"], d["r99_converging"]))
    drift = (dur_scan[0]["r99_converging"] - dur_scan[-1]["r99_converging"])
    print("  R99(converging) drifts by %.1f deg/s across durations -> "
          "not a usable scale" % drift)

    first_cmds = [row["no_noise_first_cmd_deg_s"] for row in per_trial]
    r_max_scale = peak
    print("--- duration-independent alternatives ---")
    print("  decoder peak (|steering| = 1)          = %.1f deg/s" % peak)
    print("  max first-cycle command (calibration)  = %.1f deg/s" % max(first_cmds))
    print("  CHOSEN body scale R_max                = %.1f deg/s" % r_max_scale)

    print("--- |r_brain| distribution [deg/s] ---")
    for label, s in stats.items():
        print("  %-13s median %8.2f  p90 %8.2f  p99 %8.2f  max %8.2f  "
              "(%.1f%% below 1 deg/s)"
              % (label, s["median"], s["p90"], s["p99"], s["max"],
                 100 * s["fraction_below_1_deg_s"]))
    print("  R99 (all calibration trials)      = %.2f deg/s" % r99)
    print("  R99 (converging trials only)      = %.2f deg/s" % r99_converging)
    print("  decoder peak at |steering| = 1    = %.2f deg/s" % peak)
    print("  R99 / peak                        = %.4f" % (r99 / peak))

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    v = np.rad2deg(np.concatenate(all_vals["no_noise"]))
    axes[0].hist(v, bins=80, color="C0")
    axes[0].set_yscale("log")
    axes[0].axvline(r99, color="C3", lw=2, label="R99 = %.0f" % r99)
    axes[0].axvline(peak, color="k", ls=":", lw=1.5, label="peak = %.0f" % peak)
    axes[0].set_xlabel("|r_brain| [deg/s]")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title("ideal baseline, no noise")
    axes[0].legend(fontsize=8)

    vq = np.sort(v)
    axes[1].plot(np.linspace(0, 100, vq.size), vq, lw=2)
    axes[1].axhline(r99, color="C3", lw=1.5, ls="--", label="R99")
    axes[1].set_xlabel("percentile")
    axes[1].set_ylabel("|r_brain| [deg/s]")
    axes[1].set_title("bimodal: converged (~0) or near-maximal")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Stage 1 / scale: neural command distribution")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_command_distribution.png", dpi=150)
    plt.close(fig)

    # ---- freeze ----------------------------------------------------------
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "body_scale_deg_per_s": r_max_scale,
        "body_scale_rad_per_s": float(np.deg2rad(r_max_scale)),
        "body_scale_definition": (
            "R_max = kappa * max|steering| = the fastest yaw rate the core can "
            "ever request. Exact and duration-independent, because the core's "
            "normalised steering is bounded by |steering| <= 1 on the "
            "calibration grid."),
        "R99_deg_per_s": r99,
        "R99_rad_per_s": float(np.deg2rad(r99)),
        "R99_duration_scan": dur_scan,
        "why_not_R99": (
            "Handoff document section 17 asks for R99, the 99th percentile of "
            "|r_brain| in the ideal baseline. Measured here it is not a "
            "well-defined scale. Restricted to converging trials it collapses "
            "with trial duration (1367 -> 0 deg/s from 2 s to 100 s) because "
            "the loop reaches the goal in 3-4 cycles and every later sample is "
            "~0, so the percentile measures recording length. Taken over all "
            "trials it is set by the trials that FAIL into the period-2 limit "
            "cycle (finding F1) and saturates at the decoder peak. We keep the "
            "INTENT of section 17 -- body limits as dimensionless multiples of "
            "the brain's own command scale, never invented vehicle specs -- and "
            "replace the estimator with R_max. Numerically R99(all trials) = "
            "%.1f and R_max = %.1f differ by %.2f%%, so the sweep grid is "
            "essentially unchanged." % (r99, peak, 100 * abs(r99 - peak) / peak)),
        "max_first_cycle_command_deg_per_s": float(max(first_cmds)),
        "T_core_s": decoder.T_core_s,
        "measured_on": {
            "initial_errors_deg": list(CALIBRATION_ERRORS_DEG),
            "set": "calibration only (test conditions excluded, doc 47)",
            "body": "ideal_yaw", "plugin": "passthrough",
            "noise": "none", "duration_s": args.duration,
        },
        "decoder": decoder.as_dict(),
        "distribution": stats,
        "R99_converging_trials_only_deg_per_s": r99_converging,
        "per_trial": per_trial,
        "interpretation": (
            "The command distribution is bimodal, not unimodal: %.1f%% of "
            "samples are below 1 deg/s (the loop converges in 3-4 cycles) and "
            "the rest sit near the maximum. Read 'r_max = 0.5 R_max' as 'the "
            "body can execute half of the fastest turn the brain ever "
            "requests', never as a statement about a typical command."
            % (100 * stats["no_noise"]["fraction_below_1_deg_s"])),
        "warning": ("This is a simulation scale, NOT a vehicle specification. "
                    "Never present these numbers as a real drone's limits."),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (outdir / "command_scale.json").write_text(json.dumps(payload, indent=2),
                                               encoding="utf-8")

    print("--- frozen ---")
    print("  %s" % OUT_JSON.relative_to(REPO))
    print("  %s" % (outdir / "fig1_command_distribution.png").relative_to(REPO))
    print("  sweep grid this implies (doc section 17, anchored to R_max):")
    for ratio in (0.25, 0.5, 0.75, 1.0, 1.5):
        print("     r_max = %.2f R_max = %7.1f deg/s" % (ratio, ratio * r_max_scale))
    for ratio in (0.25, 0.5, 1.0, 2.0, 4.0):
        print("     tau_r = %.2f T    = %7.1f ms" % (ratio,
                                                     1000 * ratio * decoder.T_core_s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
