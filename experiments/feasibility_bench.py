"""Is the target study actually runnable on this machine?

The target, in the user's words: a complex 3D environment, simulated directly,
with the agent TRAINED in it, and path efficiency analysed -- fly-derived core
against a conventional AI baseline, both inside a drone's feasible dynamics.

Feasibility is arithmetic, not opinion: cost per step x steps per episode x
episodes.  So measure every component's real cost and multiply.  Nothing here
is estimated from how heavy something looks.

Components measured:
    core        Westeinde reduced core, forward
    core        MaleCNS connectome rate model, forward and BACKWARD
                (backward is what training costs, and it is never the same)
    sensing     analytic scene sampling (what Stage 1 uses)
    sensing     flygym compound-eye render (measured elsewhere at ~33 ms)
    body        abstract yaw plant step
    body        3D point-mass flight plant step

Then: what a training run costs at each combination.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch


def bench(fn, n, warmup=3):
    """Median seconds per call, after warm-up.

    Median, not mean: the first call to almost everything in this project has
    turned out to be an outlier -- lazy renderer construction, JIT, allocator
    warm-up -- and a mean over few calls hides it.
    """
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def fmt(s):
    return "%8.3f ms" % (1e3 * s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-connectome", action="store_true")
    args = ap.parse_args(argv)

    res = {}
    print("=== component costs (median per call) " + "=" * 30)

    # --- Westeinde reduced core -----------------------------------------
    from core.westeinde2024 import WesteindeSteeringCore
    from core import calibration

    try:
        core = WesteindeSteeringCore(
            norm=calibration.load(REPO / "configs/norm_constants.json"))
    except Exception as e:
        print("(no frozen norm constants: %s)" % e)
        core = WesteindeSteeringCore()
    h = 0.3
    res["core: Westeinde reduced"] = bench(lambda: core.steering(h, 0.0), 200)
    print("%-34s %s" % ("core: Westeinde reduced", fmt(res["core: Westeinde reduced"])))

    # --- abstract yaw plant ---------------------------------------------
    from body.yaw_plant import YawPlant, YawPlantParams
    plant = YawPlant(params=YawPlantParams())
    res["body: abstract yaw plant"] = bench(lambda: plant.step(1.0, 0.1), 2000)
    print("%-34s %s" % ("body: abstract yaw plant",
                        fmt(res["body: abstract yaw plant"])))

    # --- analytic scene sampling ----------------------------------------
    import sensors.flywire_eye as eye
    scene = eye.bar_scene(0.0, bar_width_deg=15.0, contrast=1.0, background=0.3)
    az = np.linspace(-80, 80, 6199)
    el = np.zeros_like(az)
    res["sensing: analytic scene"] = bench(lambda: scene(az, el), 200)
    print("%-34s %s" % ("sensing: analytic scene",
                        fmt(res["sensing: analytic scene"])))

    # --- 3D point-mass flight plant (what a drone body would be) --------
    state = np.zeros(6)

    def flight_step():
        # pos(3) + vel(3), yaw-rate-commanded heading, Euler
        state[:3] += state[3:] * 0.005
        state[3:] += np.array([0.1, 0.0, 0.0]) * 0.005
    res["body: 3D point-mass flight"] = bench(flight_step, 5000)
    print("%-34s %s" % ("body: 3D point-mass flight",
                        fmt(res["body: 3D point-mass flight"])))

    print("%-34s %s  (measured in loop_profile.py)"
          % ("sensing: flygym compound eye", "  33.2 ms"))
    res["sensing: flygym eye"] = 0.0332

    # --- the connectome, forward AND backward ---------------------------
    if not args.skip_connectome:
        from core.flywire_rate import FlyWireRate, activity
        from core.malecns import load_malecns
        t0 = time.perf_counter()
        ids, out, ann, _ = load_malecns(w_scale=0.005, symmetrise=True)
        net = FlyWireRate(out_csr=out, ids=ids, ann=ann)
        print("\n(connectome loaded in %.0f s: %d neurons, %d edges, "
              "%d trainable params)"
              % (time.perf_counter() - t0, net.n, net.n_edges,
                 net.n_trainable()))
        v = net.init_state(1)
        drive = torch.zeros(net.n, 1)

        def fwd():
            with torch.no_grad():
                net.step(v, drive, 5.0)
        res["core: connectome forward"] = bench(fwd, 20)
        print("%-34s %s" % ("core: connectome forward",
                            fmt(res["core: connectome forward"])))

        def fwd_bwd():
            net.zero_grad(set_to_none=True)
            vv = net.init_state(1)
            for _ in range(4):            # 4-step truncated window
                vv = net.step(vv, drive, 5.0)
            activity(vv).sum().backward()
        res["core: connectome fwd+bwd x4"] = bench(fwd_bwd, 5)
        print("%-34s %s   (= %s per step)"
              % ("core: connectome fwd+bwd, 4 steps",
                 fmt(res["core: connectome fwd+bwd x4"]),
                 fmt(res["core: connectome fwd+bwd x4"] / 4)))

    # --- what a training run would cost ---------------------------------
    print("\n=== training budget " + "=" * 47)
    print("assume: 10 s episodes, brain at 10 Hz (Stage 1's T_core) "
          "=> 100 steps/episode")
    steps_per_ep = 100

    def budget(label, per_step, n_ep):
        total = per_step * steps_per_ep * n_ep
        unit = "s"
        v = total
        if v > 3600:
            v, unit = v / 3600, "h"
        if unit == "h" and v > 48:
            v, unit = v / 24, "days"
        print("  %-42s %8.1f %s" % (label, v, unit))

    for n_ep in (1000, 10000):
        print("\n-- %d episodes --" % n_ep)
        budget("Westeinde core + analytic scene + yaw plant",
               res["core: Westeinde reduced"] + res["sensing: analytic scene"]
               + res["body: abstract yaw plant"], n_ep)
        budget("Westeinde core + flygym eye + yaw plant",
               res["core: Westeinde reduced"] + res["sensing: flygym eye"]
               + res["body: abstract yaw plant"], n_ep)
        if not args.skip_connectome:
            budget("connectome forward + analytic scene",
                   res["core: connectome forward"]
                   + res["sensing: analytic scene"], n_ep)
            budget("connectome TRAINING (fwd+bwd) + analytic scene",
                   res["core: connectome fwd+bwd x4"] / 4
                   + res["sensing: analytic scene"], n_ep)
            budget("connectome TRAINING + flygym eye",
                   res["core: connectome fwd+bwd x4"] / 4
                   + res["sensing: flygym eye"], n_ep)

    print("\nnote: the brain runs at 10 Hz here (Stage 1's core rate), not the "
          "200 Hz\nthe flygym loop used.  That is a 20x difference and it is a "
          "modelling choice,\nnot a speedup -- it is the rate the reduced core "
          "was calibrated at.")
    assert res["core: Westeinde reduced"] > 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
