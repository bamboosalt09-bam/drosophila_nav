"""Watch a flight as it happens.

    .venv/Scripts/python.exe experiments/live_view.py
    .venv/Scripts/python.exe experiments/cx_vp_room.py --room 6 --live

Start them in either order.  The runner writes results/live.json every
0.5 s of flight time; this window redraws whenever it changes: the room,
the path so far (dark -> bright with time), the drone and its heading, the
rangefinder beams (red when closer than 1 m), the goal cue's bearing
(green) and the intended turn in the title.
"""
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LIVE = Path(__file__).resolve().parents[1] / "results" / "live.json"

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
seen = None
while plt.fignum_exists(fig.number):
    try:
        mt = os.path.getmtime(LIVE)
        if mt != seen:
            seen = mt
            with open(LIVE) as f:
                s = json.load(f)
            ax.clear()
            for x, y, r in s["obstacles"]:
                ax.add_patch(plt.Circle((x, y), r, color="0.6"))
            for t in s["all_targets"]:
                ax.plot(*t, "o", ms=16, mfc="none", mec="tab:blue")
            for t in s["targets"]:
                ax.plot(*t, "*", color="green", ms=14)
            P = np.array(s["path"])
            ax.scatter(P[:, 0], P[:, 1], c=np.arange(len(P)), cmap="plasma", s=3)
            x, y, h = P[-1, 0], P[-1, 1], s["heading"]
            for c, b in zip(s["centres"], s["beams"]):
                a = h + math.radians(c)
                d = 4.0 if b is None else b
                ax.plot([x, x + d * math.cos(a)], [y, y + d * math.sin(a)],
                        color="red" if d < 1.0 else "0.75", lw=1)
            if s["cue_deg"] is not None:
                a = h + math.radians(s["cue_deg"])
                ax.plot([x, x + 3 * math.cos(a)], [y, y + 3 * math.sin(a)],
                        color="green", lw=2)
            ax.plot(x, y, "ko", ms=7)
            ax.arrow(x, y, 1.2 * math.cos(h), 1.2 * math.sin(h), width=0.12,
                     color="k")
            ax.set_xlim(-19, 19)
            ax.set_ylim(-19, 19)
            ax.set_aspect("equal")
            state = ("COLLIDED" if s["collided"] else
                     "done" if s["final"] else "flying")
            ax.set_title("%s   t=%.1f s   found %d   %s\nintended turn %+.0f deg/s"
                         " -> flown %+.0f deg/s at %.1f m/s"
                         % (s["label"], s["t"], s["found"], state, s["u_deg"],
                            s["r_deg"], s["v"]), fontsize=10)
            fig.canvas.draw_idle()
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    plt.pause(0.2)
