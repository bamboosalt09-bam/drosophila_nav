"""Fly the projection-neuron subnetwork in a closed room, over sensor counts.

The fly PROPOSES, the drone disposes (sim.drone_layer.follow): the circuit
gives an intended turn, and one drone layer -- the same for every arm --
modifies it into a path the drone can fly without hitting anything.  Range
sensors are swept 0/1/3/6/12/24 to find how few the whole thing needs.
Inputs: docs/HANDOFF.md, "INPUTS -- the current spec".

34,125 neurons, the optic lobe replaced by a fisheye camera wired straight
into the visual projection neurons by their measured receptive fields.
54.5 ms per control step against 120 ms, and a readout four orders of
magnitude larger than the lamina-injected version.

Gain and zero come from a beacon swept round the drone in EMPTY space
(`calibrate`); the goal cue's PFL3 polarity is measured first (`wire_cue`).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import torch

from core.flywire_rate import activity
from core.vp_subnet import build_vp_subnet
from environment.room_file import ensure as room
from environment.target_world import Agent, TargetWorld
from sensors.fisheye import FisheyeCamera
from sensors.rangefinder import LAYOUTS, Rangefinder
from sensors.vp_input import VPInput, threat_level
from sim.drone_layer import beam_points, follow
from sim.episode import DT, V_CRUISE

from cx_room import START, planner_for

K_AVOID = 1.0                # 1/m: the centroid arm's threat bend
MAX_STEPS = 1200             # 120 s; at 60 s the connectome was often still en route
LIVE = None                  # --live: the file experiments/live_view.py watches
LIVE_EVERY = 5               # control steps between snapshots (0.5 s)


def _live(label, k, ag, w, sc, centres, u, cmd, found, final=False):
    """Write the flight's current state for the live viewer (--live)."""
    if LIVE is None or (k % LIVE_EVERY and not final):
        return
    import json
    import os
    snap = {"label": label, "t": round((k + 1) * DT, 1), "found": found,
            "final": final, "collided": bool(ag.collided),
            "path": ag.path_array()[:, :2].round(2).tolist(),
            "heading": float(ag.heading), "u_deg": math.degrees(u),
            "r_deg": math.degrees(cmd[0]), "v": cmd[1],
            "beams": [None if not np.isfinite(b) else round(float(b), 2)
                      for b in sc["beams"]],
            "centres": [float(c) for c in centres],
            "cue_deg": float(sc["scent"]) if sc["scent_mass"] > 0 else None,
            "targets": [[float(t[0]), float(t[1])] for t in ag.world.all_targets()],
            "all_targets": [[float(t[0]), float(t[1])] for t in w.all_targets()],
            "obstacles": [[o.x, o.y, o.radius] for o in w.obstacles]}
    tmp = str(LIVE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(snap, f)
    try:
        os.replace(tmp, LIVE)
    except PermissionError:
        # Windows: the viewer has the file open this instant.  Skip this
        # snapshot -- a live view must never stop a flight (it did, at 40 s).
        pass
# The number of sub-steps per control cycle now lives on the input, as
# `VPInput.n_sub`, because it is the length of the drive sequence.


def step_circuit(net, v, d):
    """Play the drive SEQUENCE, one column per sub-step.

    Was: one column held for all 20 sub-steps, i.e. a still photograph run
    to steady state.  At tau 20 ms and rho 0.50 the effective time constant
    is 40 ms, so 100 ms settles to 92% and every frame started from a state
    that had been washed clean.  Nothing temporal could survive that, which
    is why no looming signal existed for the circuit to steer on no matter
    how the readout was scaled.
    """
    with torch.no_grad():
        for k in range(d.shape[1]):
            v = net.step(v, d[:, k:k + 1], 5.0)
        return v, activity(v).numpy().ravel()


def steer(inp, r):
    """The steering readout: DNa01/02, right minus left."""
    return float(r[inp.steer_r].mean() - r[inp.steer_l].mean())


def calibrate_threat(net, inp, p0):
    """Escape-neuron rate per side with no threat and with threat 1.

    Threat 1 is DEFINED in the circuit, not by a pose in the world: every
    LC4/LPLC2 cell on that side driven at strength 1, i.e. every looming
    detector on that side seeing a surface at THREAT_R0 = 1.5 m.  It used to
    be a wall straight ahead, which after the per-cell change reached only
    the few cells looking within +-7.5 deg of the front beam -- the peak came
    out at 0.002, under the 0.01 guard, so for 1, 3 and 6 beams the threat
    readout was silently OFF, and at 12 beams a 0.03 peak over-scaled it
    until the drone crawled.  Defined this way the reference does not depend
    on the beam layout, and sparse coverage correctly reads as less threat.
    """
    from sensors.vp_input import THREAT_GAIN
    far = np.array([300.0, 0.0, 2.0])
    empty = TargetWorld(target=far, obstacles=[])
    side = inp.ann["side"].astype(str).to_numpy()[inp.thr_rows]

    def probe(extra):
        v = net.init_state(1)
        inp.reset()
        for _ in range(3):
            d, _ = inp.drive(empty, p0, 0.0)
            if len(extra):
                d[extra] += THREAT_GAIN
            v, r = step_circuit(net, v, d)
        return float(r[inp.esc_l].mean()), float(r[inp.esc_r].mean())

    base = probe(np.array([], dtype=int))
    peak_l = probe(inp.thr_rows[side == "left"])[0]
    peak_r = probe(inp.thr_rows[side == "right"])[1]
    return base, (peak_l, peak_r)


def threat_from(r, inp, base, peak):
    """(left, right) threat read off the escape descending neurons."""
    out = []
    for rows, b, pk in ((inp.esc_l, base[0], peak[0]),
                        (inp.esc_r, base[1], peak[1])):
        span = pk - b
        # a span under 0.01 means the threat input never reached the escape
        # neurons (0 beams gives 0.002 -- the wall's image alone; 24 give ~0.2)
        out.append(0.0 if span < 0.01
                   else max(0.0, (float(r[rows].mean()) - b) / span))
    return out


def calibrate(net, inp, info, w, p0):
    """Steering gain and zero from a BEACON-ONLY world, not from the room.

    This used to sweep headings in the cluttered room and fit the readout to
    `contrast_centroid`'s bearing -- the very quantity the reference arm
    steers on.  That makes the connectome's score a measure of how well it
    reproduces its rival, and it is not harmless: once the rangefinder
    channel was added the circuit began responding to obstacles, the fit
    read that as disagreement with the centroid, and it FLIPPED the gain
    sign.  Avoidance came out as attraction and the arm found 0 of 5.

    The anchor is the task instead.  A single beacon in empty space at a
    known bearing defines which way "toward the goal" is; the goal cue
    (PFL3) already carries that bearing, so the probe asks only whether the
    circuit's output agrees in sign and how large it is.  No obstacle
    appears in the calibration at all, so whatever the circuit does with
    obstacles is its own and is never scored against a formula.

    ponytail: two free numbers, sign and scale, from a clean probe.  If the
    day comes that these are trained rather than probed, delete this.
    """
    from environment.target_world import TargetWorld
    vals = []
    for brg in range(-120, 121, 15):
        a = math.radians(brg)
        tgt = np.array([p0[0] + 25.0 * math.cos(a),
                        p0[1] + 25.0 * math.sin(a), 2.0])
        beacon = TargetWorld(target=tgt, obstacles=[])
        v = net.init_state(1)
        inp.reset()
        for _ in range(3):
            d, tr = inp.drive(beacon, p0, 0.0)
            v, r = step_circuit(net, v, d)
        vals.append((tr["scent"], steer(inp, r)))
    g = np.array(vals)
    cc = float(np.corrcoef(g[:, 0], g[:, 1])[0, 1])
    rng = g[:, 1].max() - g[:, 1].min()
    order = np.argsort(g[:, 0])
    zero = float(np.interp(0.0, g[order, 0], g[order, 1]))
    return math.copysign(1.2 / (rng / 2), cc), cc, zero


def wire_cue(net, inp, info, p0, probe=0.02):
    """Which PFL3 side turns the drone LEFT, measured.  Returns deg/s per side.

    Uses a vision-only calibration (cue off, single circuit) to know which
    readout sign is a left turn, then drives each PFL3 side alone.
    """
    keep_gain = inp.cue_gain
    inp.cue_gain = 0.0
    g0, _, zero0 = calibrate(net, inp, info, None, p0)
    empty = TargetWorld(target=np.array([300.0, 0.0, 2.0]), obstacles=[])

    def u(rows):
        v = net.init_state(1)
        inp.reset()
        for _ in range(3):
            d, _ = inp.drive(empty, p0, 0.0)
            d[rows] += probe
            v, r = step_circuit(net, v, d)
        return math.degrees(g0 * (steer(inp, r) - zero0))

    u0 = u(np.array([], dtype=int))
    ul, ur = u(inp.pfl3_l) - u0, u(inp.pfl3_r) - u0
    if ur > ul:                       # right PFL3 turns the drone left
        inp.cue_left, inp.cue_right = inp.pfl3_r, inp.pfl3_l
    else:
        inp.cue_left, inp.cue_right = inp.pfl3_l, inp.pfl3_r
    inp.cue_gain = keep_gain
    return ul, ur


def _finish(ag, w, rem, found, first, k):
    """Beacon bookkeeping shared by both loops."""
    i = ag.world.reached_any(ag.p)
    if i is None:
        return found, first, False
    found += 1
    if first is None:
        first = k
    rem.pop(i)
    if rem:
        ag.world = TargetWorld(target=rem[0], extra_targets=rem[1:],
                               obstacles=w.obstacles, bounds=w.bounds)
    return found, first, not rem


def _row(ag, found, first, k, seen, stopped, corr, w):
    path = ag.path_array()
    # which of the ORIGINAL beacons the path came within reach of; the
    # caller splits them by the room's reachable flags, because a beacon
    # marked walled-off can still be reached -- the flag uses a flyability
    # margin (room_file.FLY_CLEARANCE) and the drone threads narrower gaps
    reached = [i for i, t in enumerate(w.all_targets())
               if np.min(np.linalg.norm(path[:, :2] - t[:2], axis=1))
               < w.reach_radius]
    return {"found": found, "first": first, "steps": k + 1,
            "path": path[:, :2].astype(np.float32),
            "reached": " ".join(map(str, reached)),
            "collided": bool(ag.collided), "cue_frac": seen / (k + 1),
            "stopped": stopped / (k + 1),
            "dist": float(np.linalg.norm(np.diff(path[:, :2], axis=0),
                                         axis=1).sum()),
            "corr": corr}


def fly(net, inp, info, w, heading, gain, zero, base, peak):
    v = net.init_state(1)
    inp.reset()
    ag = Agent(world=w, start=START, heading=heading)
    rem = list(w.all_targets())
    found, first, bs, us, seen, stopped = 0, None, [], [], 0, 0
    mem = {}                               # drone-layer state, per flight
    thr_sum = 0.0
    for k in range(MAX_STEPS):
        d, tr = inp.drive(ag.world, ag.p, ag.heading)
        v, r = step_circuit(net, v, d)
        # the INTENDED turn, not clipped: the drone layer decides what of it
        # the airframe can fly
        u = gain * (steer(inp, r) - zero)
        # the escape readout is RECORDED, not used for control: the fly
        # avoids through its own steering; this shows how alarmed it was
        thr_l, thr_r = threat_from(r, inp, base, peak)
        thr_sum += max(thr_l, thr_r)
        # a beacon in view = odour present.  `weight` was used here: the
        # signed centroid's, which walls feed too, so cue_frac read 1.0 in
        # every flight and corr compared the turn with the wall-laden
        # centroid rather than with the goal.
        if tr["scent_mass"] > 0:
            seen += 1
            bs.append(tr["scent"])
            us.append(u)
        r_cmd, v_cmd = follow(u, beam_points(tr["beams"],
                                             inp.cam.rangefinder.centres),
                              mem,
                              pose=(ag.p[0], ag.p[1], ag.heading))
        stopped += int(v_cmd <= 0.0)
        inp.speed = max(ag.v, 0.2)
        ag.step(r_cmd, v_cmd, 0.0, DT)
        label = "connectome, %d beams" % len(inp.cam.rangefinder.centres)
        _live(label, k, ag, w, tr, inp.cam.rangefinder.centres, u,
              (r_cmd, v_cmd), found)
        if ag.collided:
            break
        found, first, done = _finish(ag, w, rem, found, first, k)
        if done:
            break
    _live(label, k, ag, w, tr, inp.cam.rangefinder.centres, u,
          (r_cmd, v_cmd), found, final=True)
    corr = (float(np.corrcoef(bs, us)[0, 1])
            if len(bs) > 3 and np.std(us) > 0 else float("nan"))
    out = _row(ag, found, first, k, seen, stopped, corr, w)
    out["threat_mean"] = thr_sum / (k + 1)
    return out


def fly_reference(w, heading, centres, k_yaw=0.05, planner=None):
    """Same sensors, same drone layer; threat straight from the ranges."""
    cam = FisheyeCamera()
    cam.rangefinder = Rangefinder(cam.a, centres)
    ag = Agent(world=w, start=START, heading=heading)
    rem = list(w.all_targets())
    cmd = planner(ag.world) if planner else None
    found, first, seen, stopped = 0, None, 0, 0
    mem = {}
    for k in range(MAX_STEPS):
        sc = cam.sense(ag.world, ag.p, ag.heading)
        seen += int(sc["scent_mass"] > 0)
        if cmd:
            u = float(cmd(None, ag.heading, ag))
        else:
            # the centroid arm's OWN avoidance: steer toward the bright
            # centroid, away from the side that looms more.  This used to
            # live in the shared drone layer; it is this arm's decision.
            thr_l, thr_r = threat_level(sc["beams"], cam.rangefinder.centres,
                                        max(ag.v, 0.2))
            u = k_yaw * sc["bearing"] - K_AVOID * V_CRUISE * (thr_l - thr_r)
        r_cmd, v_cmd = follow(u, beam_points(sc["beams"],
                                             cam.rangefinder.centres),
                              mem,
                              pose=(ag.p[0], ag.p[1], ag.heading))
        stopped += int(v_cmd <= 0.0)
        ag.step(r_cmd, v_cmd, 0.0, DT)
        label = "%s, %d beams" % ("planner" if planner else "centroid",
                                  len(centres))
        _live(label, k, ag, w, sc, centres, u, (r_cmd, v_cmd), found)
        if ag.collided:
            break
        n0 = found
        found, first, done = _finish(ag, w, rem, found, first, k)
        if done:
            break
        if planner and found != n0:
            cmd = planner(ag.world)
    _live(label, k, ag, w, sc, centres, u, (r_cmd, v_cmd), found, final=True)
    return _row(ag, found, first, k, seen, stopped, float("nan"), w)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", type=int, default=0)
    ap.add_argument("--beams", default="24")   # fixed at 24 (user, 2026-09-28)
    ap.add_argument("--live", action="store_true",
                    help="write results/live.json for experiments/live_view.py")
    args = ap.parse_args(argv)
    global LIVE
    if args.live:
        LIVE = REPO / "results" / "live.json"

    t0 = time.perf_counter()
    net, ann, sub, info = build_vp_subnet()
    inp = VPInput(net, ann, info)
    # Nothing trains during a flight: expand per-type parameters once.
    net.freeze_params(5.0)
    print("%d neurons, %d edges (%.0f s); threat in: %d LC4/LPLC2 cells by "
          "direction; escape out L%d R%d; steer from DNa01/02 L%d R%d"
          % (net.n, sub.nnz, time.perf_counter() - t0, len(inp.thr_rows),
             len(inp.esc_l), len(inp.esc_r), len(inp.steer_l),
             len(inp.steer_r)), flush=True)
    p0 = np.array(START)
    w, h, reach = room(args.room)
    n_ok = sum(reach)
    # steering calibration is beacon-only -- no obstacle, so no echo in any
    # layout -- and is done once
    ul, ur = wire_cue(net, inp, info, p0)
    print("goal cue -> PFL3: left side alone turns %+.0f, right %+.0f deg/s"
          % (ul, ur), flush=True)
    gain, cc, zero = calibrate(net, inp, info, w, p0)
    print("steering calibration: sweep r %+.2f, gain %.2e" % (cc, gain),
          flush=True)

    rows, paths = [], {}
    print("\n%5s %-12s %8s %8s %9s %6s %8s %7s"
          % ("beams", "arm", "reach'ble", "walled", "collided", "steps",
             "stopped", "dist m"))
    for n in [int(x) for x in args.beams.split(",")]:
        inp.cam.rangefinder = Rangefinder(inp.cam.a, LAYOUTS[n])
        base, peak = calibrate_threat(net, inp, p0)
        t = time.perf_counter()
        res = [("connectome", fly(net, inp, info, w, h, gain, zero,
                                  base, peak))]
        res += [(nm, fly_reference(w, h, LAYOUTS[n], planner=pl))
                for nm, pl in (("planner", planner_for), ("centroid", None))]
        for nm, r in res:
            paths["b%d_%s" % (n, nm)] = r.pop("path")
            idx = [int(i) for i in r["reached"].split()]
            r.update({"room": args.room, "beams": n, "arm": nm,
                      "reachable": n_ok,
                      "found_ok": sum(reach[i] for i in idx),
                      "found_walled": sum(not reach[i] for i in idx)})
            rows.append(r)
            print("%5d %-12s %6d/%-2d %8d %9s %6d %7.0f%% %7.1f"
                  % (n, nm, r["found_ok"], n_ok, r["found_walled"],
                     r["collided"], r["steps"], 100 * r["stopped"],
                     r["dist"]), flush=True)
        print("      (%.0f s; escape base L%.3f R%.3f, peak L%.3f R%.3f)"
              % (time.perf_counter() - t, base[0], base[1], peak[0], peak[1]),
              flush=True)

    out = REPO / ("results/sweep_beams_room%d.csv" % args.room)
    pd.DataFrame(rows).to_csv(out, index=False)
    np.savez_compressed(REPO / ("results/paths_room%d.npz" % args.room),
                        **paths)
    print("wrote %s" % out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
