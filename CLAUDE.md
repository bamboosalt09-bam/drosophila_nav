# drosophila_nav — read before doing anything

1. Read `docs/HANDOFF.md`: the goal section, then the **2026-09-28
   Structure B** section — its "INPUTS" and "DRONE LAYER" specs are the
   current state; the newest entry at its end is the open question.  Then
   "Working with this user".  Do not implement ahead of an open decision.
   The settled research design is `docs/master_handoff.md`.
2. Rules (details in HANDOFF):
   - Answer in Korean.  Give a recommendation, not a survey.
   - Diagnose by reading code.  Predict the number before any run.
   - One room (room 6) while debugging; four rooms only once it works.
   - Range sensor fixed at 24 beams (user, 2026-09-28).
   - Do not fix what was not asked.  "멈춰" = stop now.
   - Show results as images; save each run as `_vNN` before the next.
   - Commit and push to GitHub after each step; the repo is the record.
3. Live view (the user likes to watch): start
   `.venv/Scripts/python.exe experiments/live_view.py` (a window), then run
   `experiments/cx_vp_room.py --room 6 --live`.
4. Self-checks: `PYTHONPATH=src .venv/Scripts/python.exe -m sim.drone_layer`,
   `-m sensors.vp_input`, `-m sensors.rangefinder`.  A fresh clone runs
   without the 2 GB connectome (README "Start from a fresh clone").
