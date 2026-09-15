"""Stage 0, part 2: ideal closed loop.

Three things are established here, in order:

  1. Reproduction of the source's closed-loop simulation (notebook cell 32):
     50 runs per PFL scalar S, 1000 timesteps at 10 Hz, starting anti-goal at
     180 deg, with the source's low-pass Gaussian command noise.  The reported
     statistic is the heading consistency rho.

  2. Agreement between that literal replica and OUR scheduler
     (decoder -> passthrough plugin -> ideal body -> ideal sensor).  They
     should agree up to the source's 1-degree heading quantisation.  This is
     what licenses using our scheduler for every later condition: condition A
     run through the general machinery is the source's ideal-body loop.

  3. The deterministic basin of attraction, with noise switched off.  With the
     source gain k = 200 the loop contracts to the goal from small errors but
     falls into a period-2 limit cycle from large ones.  Knowing where that
     boundary sits BEFORE a body is attached is what will let us tell a
     body-induced failure from one the core already had (handoff doc
     section 27).

Usage
-----
    python experiments/stage0_ideal_loop.py
    python experiments/stage0_ideal_loop.py --runs 10 --tag quick
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
import pandas as pd

from body.ideal_yaw import IdealYawBody
from core.calibration import calibrate, load
from core.westeinde2024 import CoreParams, WesteindeSteeringCore
from decoder.steering import SteeringDecoder
from environment.heading_task import (CALIBRATION_ERRORS_DEG, HeadingTask,
                                      STRESS_ERRORS_DEG, TEST_ERRORS_DEG)
from plugins.passthrough import PassthroughPlugin
from sensors.ideal_heading import IdealHeadingSensor
from sim.closed_loop import resultant_length, run_closed_loop
from sim.noise import (SourceNoiseSpec, deg_per_step_to_rad_per_s,
                       source_command_noise_deg)
from sim.source_closed_loop import run_source_loop, steering_lookup_table
from utils.angles import wrap_deg

NORM_JSON = REPO / "configs" / "norm_constants.json"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "no-commit"
    except Exception:
        return "unavailable"


def build_parser():
    p = argparse.ArgumentParser(description="Stage 0 ideal closed loop")
    p.add_argument("--runs", type=int, default=50, help="source uses 50")
    p.add_argument("--steps", type=int, default=1000, help="source uses 1000")
    p.add_argument("--n-scalars", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--basin-step-deg", type=float, default=1.0)
    p.add_argument("--basin-steps", type=int, default=200)
    p.add_argument("--tag", default="default")
    return p


def classify_tail(err_deg_tail: np.ndarray, tol: float = 1.0):
    """Classify the last part of a deterministic trajectory."""
    spread = float(np.max(err_deg_tail) - np.min(err_deg_tail))
    final = float(np.mean(np.abs(err_deg_tail)))
    if final <= tol and spread <= tol:
        return "converged"
    if spread <= tol:
        return "fixed_point_off_goal"
    return "oscillation"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    outdir = REPO / "results" / "stage0_loop" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    norm = load(NORM_JSON) if NORM_JSON.exists() else calibrate(CoreParams(),
                                                                verbose=True)
    decoder = SteeringDecoder()
    spec = SourceNoiseSpec()
    T = decoder.T_core_s
    S_vals = np.linspace(0.0, 1.0, args.n_scalars, endpoint=True)
    hd_grid = np.arange(-180.0, 181.0, 1.0)   # official HD grid

    print("=== Stage 0 part 2: ideal closed loop (tag: %s) ===" % args.tag)
    print("  decoder: k = %.0f deg/step, T_core = %.2f s, peak %.0f deg/s"
          % (decoder.kappa_deg_per_step, T, np.rad2deg(decoder.max_yaw_rate)))
    print("  %d runs x %d scalars x %d steps" % (args.runs, S_vals.size, args.steps))

    # =====================================================================
    # 1 + 2. source reproduction and scheduler agreement
    # =====================================================================
    rho_source = np.zeros((args.runs, S_vals.size))
    rho_ours = np.zeros((args.runs, S_vals.size))
    traj_example = {}
    max_traj_dev = 0.0

    cores = {}
    tables = {}
    for si, S in enumerate(S_vals):
        cores[si] = WesteindeSteeringCore(CoreParams(pfl_scalar_S=float(S)), norm)
        tables[si] = steering_lookup_table(cores[si], hd_grid, 0.0)

    rng = np.random.default_rng(args.seed)
    for run in range(args.runs):
        # one noise realisation per run, shared across S values (as in cell 32)
        noise_deg = source_command_noise_deg(args.steps, rng, spec)
        noise_rate = deg_per_step_to_rad_per_s(noise_deg[1:], T)

        for si, S in enumerate(S_vals):
            src = run_source_loop(tables[si], hd_grid, n_steps=args.steps,
                                  k=decoder.kappa_deg_per_step,
                                  noise_deg=noise_deg, initial_hd_deg=180.0)
            rho_source[run, si] = resultant_length(np.deg2rad(src.hd_deg))

            task = HeadingTask(goal=0.0, initial_heading=np.pi,
                               duration_s=(args.steps - 1) * T, T_core_s=T)
            res = run_closed_loop(cores[si], decoder, PassthroughPlugin(),
                                  IdealYawBody(), IdealHeadingSensor(), task,
                                  command_noise=noise_rate)
            ours_deg = np.rad2deg(res.psi)
            rho_ours[run, si] = resultant_length(res.psi)

            dev = float(np.max(np.abs(wrap_deg(ours_deg - src.hd_deg))))
            max_traj_dev = max(max_traj_dev, dev)
            if run == 0:
                traj_example[si] = (src.hd_deg, ours_deg)

        if args.runs >= 10 and (run + 1) % max(1, args.runs // 5) == 0:
            print("  ... %d/%d runs" % (run + 1, args.runs))

    print("--- heading consistency rho (mean +- sd over runs) ---")
    print("  %6s %22s %22s" % ("S", "source replica", "our scheduler"))
    for si, S in enumerate(S_vals):
        print("  %6.2f   %8.4f +- %-8.4f   %8.4f +- %-8.4f"
              % (S, rho_source[:, si].mean(), rho_source[:, si].std(),
                 rho_ours[:, si].mean(), rho_ours[:, si].std()))

    rho_diff = float(np.max(np.abs(rho_source - rho_ours)))
    print("  max |rho difference| between the two loops: %.4f" % rho_diff)
    print("  max trajectory deviation: %.3f deg "
          "(source snaps heading to a 1 deg grid)" % max_traj_dev)

    # =====================================================================
    # 2b. deterministic agreement, and what the source's 1 deg grid costs
    # =====================================================================
    print("--- deterministic agreement (noise off, S = 1) ---")
    core1 = cores[S_vals.size - 1]
    table1 = tables[S_vals.size - 1]

    def _det_pair(e0_deg: float, n_steps: int = 200):
        src = run_source_loop(table1, hd_grid, n_steps=n_steps,
                              k=decoder.kappa_deg_per_step,
                              initial_hd_deg=float(e0_deg))
        task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(float(e0_deg)),
                           duration_s=(n_steps - 1) * T, T_core_s=T)
        res = run_closed_loop(core1, decoder, PassthroughPlugin(),
                              IdealYawBody(), IdealHeadingSensor(), task)
        ours_deg = np.rad2deg(res.psi)
        return src.hd_deg, ours_deg, float(np.max(np.abs(
            wrap_deg(ours_deg - src.hd_deg))))

    regular = list(CALIBRATION_ERRORS_DEG) + list(TEST_ERRORS_DEG)
    det_devs = {e0: _det_pair(e0)[2] for e0 in regular}
    det_worst = max(det_devs.values())
    print("  over %d calibration/test initial errors: max deviation %.2f deg"
          % (len(regular), det_worst))

    # The separatrix is handled separately on purpose (handoff doc section 48):
    # near the unstable anti-goal equilibrium a 1 degree difference in heading
    # decides WHICH attractor is reached, so the two loops can end up in
    # different places without either being wrong.
    sep_devs = {e0: _det_pair(e0)[2] for e0 in STRESS_ERRORS_DEG}
    print("  near the anti-goal separatrix: %s"
          % {("%.0f" % k): round(v, 2) for k, v in sep_devs.items()})

    # =====================================================================
    # 3. deterministic basin of attraction
    # =====================================================================
    print("--- deterministic basin (noise off, S = 1) ---")
    e0s = np.arange(-179.0, 179.5, args.basin_step_deg)
    basin = []
    for e0 in e0s:
        task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(float(e0)),
                           duration_s=(args.basin_steps - 1) * T, T_core_s=T)
        res = run_closed_loop(core1, decoder, PassthroughPlugin(),
                              IdealYawBody(), IdealHeadingSensor(), task)
        tail = res.error_deg[-50:]
        basin.append({"e0_deg": float(e0), "outcome": classify_tail(tail),
                      "final_abs_err_deg": float(np.mean(np.abs(tail))),
                      "tail_spread_deg": float(np.max(tail) - np.min(tail))})
    basin_df = pd.DataFrame(basin)
    conv = basin_df["outcome"] == "converged"
    frac = float(conv.mean())
    if conv.any():
        edge = float(np.max(np.abs(basin_df.loc[conv, "e0_deg"])))
        # largest |e0| such that everything inside also converged
        inside = basin_df[np.abs(basin_df["e0_deg"]) <= edge]
        contiguous = float(np.max(np.abs(
            basin_df.loc[conv, "e0_deg"]))) if inside["outcome"].eq("converged").all() else np.nan
    else:
        edge = contiguous = np.nan
    print("  converged from %.1f%% of initial errors" % (100 * frac))
    print("  outermost converging |e0| = %.1f deg" % edge)
    print("  contiguous basin edge     = %s"
          % ("%.1f deg" % contiguous if np.isfinite(contiguous) else "not contiguous"))
    print("  outcome counts: %s" % basin_df["outcome"].value_counts().to_dict())
    osc = basin_df["outcome"] == "oscillation"
    if osc.any():
        # where does the loop end up when it does not reach the goal?
        task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(90.0),
                           duration_s=(args.basin_steps - 1) * T, T_core_s=T)
        res = run_closed_loop(core1, decoder, PassthroughPlugin(),
                              IdealYawBody(), IdealHeadingSensor(), task)
        tail = res.error_deg[-20:]
        cycle = {"upper_deg": float(np.max(tail)), "lower_deg": float(np.min(tail)),
                 "period_steps": 2 if np.sign(tail[-1]) != np.sign(tail[-2]) else None}
        print("  non-goal attractor from e0 = 90 deg: alternates %.1f / %.1f deg"
              % (cycle["upper_deg"], cycle["lower_deg"]))
    else:
        cycle = None
    basin_df.to_csv(outdir / "basin_of_attraction.csv", index=False)

    # =====================================================================
    # figures
    # =====================================================================
    def finish(fig, name, title):
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(outdir / name, dpi=150)
        plt.close(fig)

    t_axis = np.arange(args.steps) * T
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for si, ax in enumerate(axes.ravel()):
        if si >= S_vals.size:
            ax.axis("off")
            continue
        src_hd, our_hd = traj_example[si]
        ax.plot(t_axis, src_hd, lw=2.2, alpha=0.45, label="source replica")
        ax.plot(t_axis, our_hd, lw=0.9, ls="--", color="C3", label="our scheduler")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("S = %.1f   (rho %.3f)" % (S_vals[si], rho_ours[0, si]),
                     fontsize=9)
        ax.set_ylim(-180, 180)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("head direction [deg]")
    axes[1, 0].set_ylabel("head direction [deg]")
    axes[0, 0].legend(fontsize=7)
    finish(fig, "fig1_heading_trajectories.png",
           "Stage 0 / loop fig1: heading over time, run 0, goal = 0 deg")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].errorbar(S_vals, rho_source.mean(0), yerr=rho_source.std(0),
                     marker="o", lw=2, capsize=3, label="source replica")
    axes[0].errorbar(S_vals, rho_ours.mean(0), yerr=rho_ours.std(0),
                     marker="s", lw=1.2, ls="--", capsize=3, label="our scheduler")
    axes[0].set_xlabel("PFL scalar S")
    axes[0].set_ylabel("heading consistency rho")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[1].plot(S_vals, np.abs(rho_source - rho_ours).max(0), marker="o",
                 color="C3")
    axes[1].set_xlabel("PFL scalar S")
    axes[1].set_ylabel("max |rho difference|")
    axes[1].set_title("loop agreement (max %.4f)" % rho_diff)
    for ax in axes:
        ax.grid(alpha=0.3)
    finish(fig, "fig2_rho_vs_S.png",
           "Stage 0 / loop fig2: goal-holding vs PFL scalar (%d runs)" % args.runs)

    colors = {"converged": "C2", "oscillation": "C3",
              "fixed_point_off_goal": "C1"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for outcome, sub in basin_df.groupby("outcome"):
        axes[0].plot(sub["e0_deg"], sub["final_abs_err_deg"], ".",
                     ms=4, color=colors.get(outcome, "C7"), label=outcome)
    axes[0].set_xlabel("initial heading error [deg]")
    axes[0].set_ylabel("final |error| [deg]")
    axes[0].set_yscale("symlog", linthresh=1e-3)
    axes[0].legend(fontsize=8)
    axes[1].scatter(basin_df["e0_deg"], np.ones(len(basin_df)),
                    c=[colors.get(o, "C7") for o in basin_df["outcome"]],
                    marker="|", s=400)
    axes[1].set_xlabel("initial heading error [deg]")
    axes[1].set_yticks([])
    axes[1].set_title("outcome map (noise off, S = 1)")
    for ax in axes:
        ax.set_xticks(np.arange(-180, 181, 45))
        ax.grid(alpha=0.3)
    finish(fig, "fig3_basin_of_attraction.png",
           "Stage 0 / loop fig3: deterministic basin with k = %.0f"
           % decoder.kappa_deg_per_step)

    # =====================================================================
    # checks and provenance
    # =====================================================================
    # S = 0 must switch the controller off entirely: with no PFL drive there is
    # no steering command at all, so that condition is an open-loop random walk
    # driven by the noise.  Its rho is a no-control baseline, not goal holding.
    s0_max = float(np.max(np.abs([cores[0].evaluate_deg(float(h), 0.0).steering
                                  for h in np.arange(-180.0, 181.0, 5.0)])))
    checks = [
        ("loops agree on rho (< 0.05)", rho_diff < 0.05),
        ("deterministic loops agree away from the separatrix (< 2 deg)",
         det_worst < 2.0),
        ("S = 0 produces zero steering (open loop)", s0_max < 1e-12),
        ("rho increases with S",
         bool(rho_ours.mean(0)[-1] > rho_ours.mean(0)[0] + 0.1)),
        ("S = 1 holds the goal (rho > 0.8)", bool(rho_ours.mean(0)[-1] > 0.8)),
        ("a finite basin exists", bool(0.0 < frac < 1.0)),
    ]
    print("--- checks ---")
    all_ok = True
    for name, ok in checks:
        all_ok &= bool(ok)
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))

    pd.DataFrame({"S": S_vals,
                  "rho_source_mean": rho_source.mean(0),
                  "rho_source_sd": rho_source.std(0),
                  "rho_ours_mean": rho_ours.mean(0),
                  "rho_ours_sd": rho_ours.std(0)}).to_csv(
        outdir / "rho_vs_S.csv", index=False)

    prov = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "experiments/stage0_ideal_loop.py",
        "git_commit": git_commit(),
        "core_source_sha256_16": hashlib.sha256(
            (REPO / "src" / "core" / "westeinde2024.py").read_bytes()
        ).hexdigest()[:16],
        "core_params": CoreParams().as_dict(),
        "decoder": decoder.as_dict(),
        "noise": spec.as_dict(),
        "seed": args.seed,
        "runs": args.runs, "steps": args.steps,
        "initial_hd_deg": 180.0,
        "plugin": "passthrough", "body": "ideal_yaw", "sensor": "ideal_heading",
        "rho_source_mean": rho_source.mean(0).tolist(),
        "rho_ours_mean": rho_ours.mean(0).tolist(),
        "max_rho_difference": rho_diff,
        "max_trajectory_deviation_deg_noisy": max_traj_dev,
        "deterministic_max_deviation_deg": det_worst,
        "deterministic_deviation_by_e0": {("%.0f" % k): v
                                          for k, v in det_devs.items()},
        "separatrix_deviation_by_e0": {("%.0f" % k): v
                                       for k, v in sep_devs.items()},
        "steering_max_at_S0": s0_max,
        "non_goal_attractor": cycle,
        "trajectory_deviation_note": (
            "Individual trajectories can differ by ~100 deg when the path "
            "passes the anti-goal separatrix, because the source loop snaps "
            "heading to a 1 deg grid and that decides which attractor is "
            "reached. Away from the separatrix the two loops agree to ~1 deg, "
            "and the rho statistics agree throughout."),
        "basin": {"converged_fraction": frac,
                  "outermost_converging_abs_e0_deg": edge,
                  "contiguous_edge_deg": (None if not np.isfinite(contiguous)
                                          else contiguous),
                  "counts": basin_df["outcome"].value_counts().to_dict()},
        "checks": {name: bool(ok) for name, ok in checks},
        "all_checks_passed": bool(all_ok),
        "python": platform.python_version(), "numpy": np.__version__,
    }
    (outdir / "provenance.json").write_text(json.dumps(prov, indent=2),
                                            encoding="utf-8")

    print("--- written to %s ---" % outdir)
    for f in sorted(outdir.iterdir()):
        print("  " + f.name)
    print("STAGE0-PART2: %s" % ("ALL CHECKS PASSED" if all_ok else "CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
