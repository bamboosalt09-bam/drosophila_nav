"""Why are the joint commands 1e-3?

Three candidate explanations, and they call for different fixes:

  A. scale       every rate in the network is tiny, so the difference is too.
                 w_scale was calibrated on central-brain heading variation and
                 never on motor output.  Fix: recalibrate.
  B. cancellation the agonist and antagonist groups are both active and nearly
                 equal, so the difference vanishes.  Fix: nothing -- that is a
                 fly standing still, and a real asymmetry has to come from a
                 stimulus.
  C. silence     the motor neurons receive essentially no synaptic input.
                 Fix: the wiring or the injection is wrong.

These are distinguishable, so measure rather than argue.  Everything is split
-- by anatomical stage, by segment, by side, by antagonist group -- because a
pooled motor rate is exactly the kind of number this project has been burned
by six times.

Run from the real closed loop, not a static scene, so the luminance is the
luminance the body actually produces.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym.vision.retina import Retina

import sensors.flywire_eye as eye
import sensors.flygym_bridge as bridge
from core.flywire_rate import FlyWireRate, activity
from core.malecns import load_malecns
from decoder.steering import JOINT_MUSCLES
from sim.flygym_loop import ConnectomeFlyLoop, build_fly

N_STEPS = 20
PCTL = (50, 90, 99, 100)


def pctl_line(label: str, r: np.ndarray) -> str:
    if r.size == 0:
        return "%-16s  (none)" % label
    q = np.percentile(r, PCTL)
    return ("%-16s n=%6d  median %.3e  p90 %.3e  p99 %.3e  max %.3e  "
            "frac>0 %.3f" % (label, r.size, q[0], q[1], q[2], q[3],
                             float((r > 1e-12).mean())))


def main() -> None:
    t0 = time.perf_counter()
    ids, out, ann, _ = load_malecns(w_scale=0.005, symmetrise=True)
    net = FlyWireRate(out_csr=out, ids=ids, ann=ann)
    lat = eye.load_malecns_eye(ann)
    omm, _ = bridge.join(lat, Retina())

    world = FlatGroundWorld()
    fly = build_fly()
    world.add_fly(fly, spawn_position=(0.0, 0.0, 0.5),
                  spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    loop = ConnectomeFlyLoop(net, ann, lat, omm, sim, fly)
    print("ready (%.1f s); running %d brain steps" %
          (time.perf_counter() - t0, N_STEPS), flush=True)

    for i in range(N_STEPS):
        loop.step()
    rates = activity(loop.v).numpy().ravel()

    sc = ann["super_class"].to_numpy(dtype="<U32")
    nm = ann["neuromere"].to_numpy(dtype="<U16")
    side = ann["side"].to_numpy(dtype="<U16")
    ctype = ann["cell_type"].to_numpy(dtype="<U48")
    motor = np.isin(sc, ("vnc_motor", "cb_motor", "vnc_efferent"))

    # the chain the signal has to travel, in anatomical order.  1,314
    # descending neurons are the ONLY brain -> nerve cord path, so if the
    # activity dies anywhere it will show as a step down at one of these.
    chain = ("ol_sensory", "ol_intrinsic", "visual_projection",
             "cb_intrinsic", "descending_neuron", "vnc_intrinsic",
             "vnc_motor", "cb_motor")
    print("\n--- rates along the chain " + "-" * 44)
    print(pctl_line("all", rates))
    for name in chain:
        print(pctl_line(name, rates[sc == name]))
    print(pctl_line("motor (all)", rates[motor]))

    # and where the input dies: same chain, but the synaptic input arriving
    W_in = out.tocsr()
    print("\n--- input arriving, same chain " + "-" * 39)
    for name in chain:
        rows = np.flatnonzero(sc == name)
        if rows.size == 0:
            continue
        inc = W_in[:, rows]
        p = inc.multiply(inc > 0).T.dot(rates)
        q = inc.multiply(inc < 0).T.dot(rates)
        print("%-20s exc median %.3e  inh median %.3e  net median %+.3e"
              % (name, np.median(p), np.median(q), np.median(p + q)))

    # C: does any synaptic input arrive at the motor neurons?  Excitatory and
    # inhibitory separately, or cancellation hides inside the sum.
    W = out.tocsr()
    r = rates
    rows = np.flatnonzero(motor)
    inc = W[:, rows]                      # out[i, j] = i -> j
    pos = inc.multiply(inc > 0).T.dot(r)
    neg = inc.multiply(inc < 0).T.dot(r)
    print("\n--- synaptic input arriving at %d motor neurons %s"
          % (len(rows), "-" * 18))
    print("  excitatory  median %.3e  max %.3e" %
          (np.median(pos), pos.max()))
    print("  inhibitory  median %.3e  min %.3e" %
          (np.median(neg), neg.min()))
    print("  net         median %.3e  |net| median %.3e" %
          (np.median(pos + neg), np.median(np.abs(pos + neg))))
    print("  motor neurons receiving nothing at all: %d of %d"
          % (int(((pos == 0) & (neg == 0)).sum()), len(rows)))

    # B: agonist against antagonist, one row per joint, never the difference
    # alone.  If both means are large and close, that is cancellation.
    print("\n--- antagonist groups " + "-" * 47)
    print("%-14s %5s %5s %11s %11s %11s" %
          ("joint", "n+", "n-", "mean+", "mean-", "diff"))
    leg_letter = {"T1": "f", "T2": "m", "T3": "h"}
    worst = 0.0
    for seg in ("T1", "T2", "T3"):
        for lr, pfx in (("left", "l"), ("right", "r")):
            base = motor & (nm == seg) & (side == lr)
            for joint, (plus, minus) in JOINT_MUSCLES.items():
                mp = base & np.array([any(k in t for k in plus)
                                      for t in ctype])
                mm = base & np.array([any(k in t for k in minus)
                                      for t in ctype])
                a = float(rates[mp].mean()) if mp.any() else 0.0
                b = float(rates[mm].mean()) if mm.any() else 0.0
                worst = max(worst, abs(a - b))
                print("%-14s %5d %5d %11.3e %11.3e %11.3e"
                      % ("%s%s_%s" % (pfx, leg_letter[seg], joint),
                         mp.sum(), mm.sum(), a, b, a - b))

    print("\nlargest |command| %.3e; median network rate %.3e"
          % (worst, np.median(rates)))
    # the three explanations are mutually exclusive; say which one the numbers
    # actually support rather than leaving it to the reader
    assert rates.size == net.n
    print("silent motor neurons imply C; large equal group means imply B; "
          "everything small together implies A")


if __name__ == "__main__":
    main()
