# drosophila_nav — read before doing anything

1. Read `docs/HANDOFF.md`: the goal section, then the **2026-09-28
   Structure B** section, then "Working with this user".  It ends with the
   decision that is waiting on the user; do not implement ahead of it.
2. Rules (details in HANDOFF):
   - Answer in Korean.  Give a recommendation, not a survey.
   - Diagnose by reading code.  Predict the number before any run.
   - One room (room 6) while debugging; four rooms only once it works.
   - Do not fix what was not asked.  "멈춰" = stop now.
   - Show results as images; save each run as `_vNN` before the next.
3. Live view (the user likes to watch): start
   `.venv/Scripts/python.exe experiments/live_view.py` (a window), then run
   `experiments/cx_vp_room.py ... --live`.
4. Self-checks: `PYTHONPATH=src .venv/Scripts/python.exe -m sim.drone_layer`
   and `-m sensors.rangefinder`.  Data (`data/`, ~2 GB) is not in git; see
   README "Data".
