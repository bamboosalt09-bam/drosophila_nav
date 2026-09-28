# Handoff — resume from here

`provenance/reproduction_adjustments.yaml` is the full record; this says where
the work stands and what to do next.

**Latest state: 2026-09-28, "Structure B" (below the goal section).**  Read
the goal section, then that section, then "Working with this user".  The
pending decision is at the end of the 2026-09-28 section: do not start coding
before the user answers it.

# THE RESEARCH GOAL — read this before doing anything

**This section exists because it was missing, and its absence cost a whole
session.**  An earlier version of this file opened with "can a fly connectome
produce navigation?", which is a fragment of the goal with the drone, the AI
comparison, the training and the path-efficiency analysis all dropped.  Working
from that fragment I built a walking NeuroMechFly, profiled it and optimised
it — none of which is on the research axis.  The goal below is quoted, not
paraphrased, for that reason.

Source of truth: the user's own `drosophila_navigation_handoff_master.md`
(60 sections, in Downloads) plus the user's messages, dated.

## The target, in the user's words

> 드론의 비행 경로를 초파리의 비행 경로와 동일시해서 초파리의 조향과 드론을
> 동일시하되, **드론 자체 거동 허용 범위 내에서만** 이동하게 할 때 얼마나
> **기존 인공지능에 비해서** 잘 이동하냐  *(2026-09-17)*

> **복잡한 3D 환경**에서 직접 시뮬레이션 되어 **직접 학습**시키면서 **경로의
> 효율성 분석**까지 하고 싶다  *(2026-09-17)*

> **동일 학습 동일 환경**에서 어떻게 학습이 최적화 되는가  *(2026-09-16)*

> 기존 인공지능 학습 경로와 똑같은 학습을 거칠 때 **학습 효율의 차이**
> *(2026-09-16)*

> 평면 이동 → 탐색·추적 → **학습형 AI 비교**  *(2026-09-16)*

So the deliverable is a **comparison**, with two measured layers:

    learning    how each controller's optimisation proceeds, same environment,
                same protocol -- convergence path, sample efficiency
    path        how efficient the resulting trajectory is

Why flight and not walking: master doc section 21 lists the bodies this study
is meant to span -- *fly-like body, **quadrotor-like body**, wheeled robot,
underwater vehicle*.  The target body lineage is a drone, a drone is a flying
body, and what transfers to a drone's yaw axis is FLIGHT steering.  Walking
steering is entangled with legs, stance and gait phase, and transfers to
nothing.

"드론 자체 거동 허용 범위" is already built: it is the ConstraintPlugin plus the
feasibility screen, `(r_max, alpha_max, tau_r)`.

## Sensing — the user's design (2026-09-17)

> 시각도 **카메라 3개(또는 2개)**로 대체할 생각이었음. 2개(또는 1개)의
> 카메라는 실제 시각 데이터, 나머지 하나는 **페로몬 데이터 입력**

NOT the fly's compound eye.  Plain cameras.  Two visual cameras because depth
has to come from somewhere — binocular, on the user's reasoning.

### The third input is NOT a chemical sensor.  Name it correctly.

The user's own correction, and it matters because the wrong name invites the
wrong implementation:

> 그건 그냥 "페로몬 추적 시스템"이라고 쓰면 약간 오해가 생깁니다. 실제
> 후각·페로몬 센서를 쓰는 구조가 아니라, 시각적으로 검출한 표적의 위치를
> 저차원화해서 초파리의 유인성 단서처럼 항법 회로에 공급하는 구조였죠.

**가상 유인 단서(virtual attraction cue) 모듈** — or in full, *시각 표적의
센트로이드 기반 가상 유인 단서 생성·추적 모듈*.

    3D 시각 입력 -> 표적 검출 -> 센트로이드 추적 -> 상대 방향/편차 추출
                 -> 가상 유인 단서 -> 초파리 항법 회로

The user's definition, to be carried verbatim into any write-up:

> 가상 유인 단서(virtual attraction cue) 모듈: 카메라 영상에서 표적을 검출하고
> 센트로이드를 추적한 뒤, 화면 중심에 대한 표적의 상대 방향·편차·검출 유효성
> 등을 저차원 신호로 변환한다. 이 신호는 실제 화학적 페로몬을 감지하는 것이
> 아니라, 시각적 표적 정보를 초파리의 유인성 감각 단서에 대응되는 형태로
> 단순화한 입력이다. 이후 이동 방향의 결정은 이 모듈이 아니라 초파리 유래
> 항법 회로가 수행한다.

Concretely: target centroid right of image centre -> an attraction cue to the
right, and vice versa.  Target size or detection confidence may set the cue's
strength.  "가상 페로몬" is a conceptual analogy and belongs in parentheses, not
in the module name.

### The boundary that must not be crossed

The module is a **sensory front-end**, not a navigation controller:

> "표적을 어디로 추적해야 하는지 결정하는 항법 controller"가 아니라, "시각
> 입력에서 표적 방향을 추출하여 생물학적 항법 회로에 전달하는 sensory
> front-end"입니다.

This is the same rule the master document already imposes on the body plugin
(section 3.5, goal-blind), now applied to the sensor side.  If the cue module
ever chooses a heading, the study stops measuring the fly circuit and starts
measuring the module.

**Terminology rule going forward:** write "시각 기반 가상 유인 단서 추적", not
"페로몬 추적".

### The master document already anticipated this module

Section 22, on extending past 1-DOF yaw:

> core가 직접 global position을 받아서는 안 된다. 가능한 구조:
> environment -> sensor / **goal-vector preprocessing** -> desired heading
> representation -> fly core -> steering

The virtual attraction cue module IS that goal-vector preprocessing stage.  The
design is not a new direction; it is the slot section 22 left open.

### Where it lands in the connectome, if the whole-connectome core is used

The cue is visual in origin, so it does not need an olfactory address.  But if
the cue is ever delivered as an attraction signal rather than a heading, the
anatomical addresses exist and need no functional designation, because the
glomerulus name IS the address:

    ORN          2,635   olfactory receptor neurons, named by glomerulus
      ORN_DA1      204   cVA pheromone (Or67d)
      ORN_VA1d     132   Or88a
      ORN_VA1v     130   Or47b
      ORN_DL3      103   cVA (Or65a/b/c)
      -------------------
                   569   pheromone-responsive, bilateral
    JO             672   Johnston's organ -- airflow.  A flying body has this
                         and so does a drone (airspeed / IMU).

Whether to inject the cue there or as a goal representation into the reduced
core is an open modelling choice, not a settled one.

## What is NOT the goal

Everything here is something a previous session drifted into:

- making a NeuroMechFly **walk**.  The body is an abstract yaw plant standing
  in for a drone (master doc section 10), not a detailed fly.
- whole-brain as the first step.  Master doc section 41 forbids it for the
  first implementation; section 58 puts it at **priority 9 of 9**.  The user
  did later authorise a whole-connectome core, but as a CORE swap, with the
  body unchanged.
- flapping-wing aerodynamics.  Section 41 forbids real quadrotor aerodynamics
  too; the plant is abstract on purpose.
- polishing the 3D visualisation.  The user: *"3D 영상의 완성도를 높이는 작업은
  현재의 과학적 불확실성을 해결하는 우선 과제는 아닙니다."*

## 2026-09-28 — Structure B: the fly proposes, the drone follows

### The premise, in the user's words — every design choice answers to this

> 초파리가 예상 경로를 지정해주면 그걸 드론이 현실적인 조건 하에서 경로를
> 수정해 따라가는 거  *(2026-09-28)*

> 나는 센서를 최소화해도 작동은 되게 하고 싶었거든.  *(2026-09-28)*

> 초파리는 벽에 앉아도 되지만 드론은 안되니까. 거리 센서 넣어봐  *(2026-09-28)*

I spent days making the fly circuit do collision avoidance by itself.  That
was the wrong division of labour.  The user chose **structure B** ("당연히 B로
가야하고"):

    fisheye + rangefinder ─► connectome (VP subnet) ─► intended turn u
                                                         │
                         drone layer (sim/drone_layer.py, SAME for every arm)
                         ─► yaw rate r, speed v within airframe limits

* The fly is the drone's onboard brain.  It says where it wants to go
  (intended turn, unlimited rate).  It does **not** get a flyable path handed
  to it and it does **not** need to avoid walls perfectly.
* The drone layer turns the intent into a feasible motion: speed from the
  predicted collision along the intended arc, stop, committed turn when
  blocked, back-off with a clear rear.  **It never chooses direction** —
  direction belongs to the proposer.  (Rejected alternative A: a virtual fly
  flying ahead of the drone.  Its camera would have to be where no camera is;
  impossible on a real drone.)
* Arms compared, all on the same drone layer and sensors: **connectome**,
  **planner** (knows the whole map — an upper bound, not an AI baseline),
  **centroid** (reactive: turn toward the bright blob, bend away from threat).

### What is in the loop now (read these files first)

| File | Role |
|---|---|
| `experiments/cx_vp_room.py` | the experiment.  `--room N --beams 0,1,3,6,12,24`.  Writes `results/sweep_beams_room{N}.csv`, `results/paths_room{N}.npz` (overwritten each run — copy to `_vNN` to keep) |
| `src/sim/drone_layer.py` | `follow(u, pts, rear, mem, pose)` — the drone layer.  `demo()` asserts its behaviour |
| `src/sensors/rangefinder.py` | HC-SR04-class ring: 15° cones, 0.02–4 m, 2 cm noise, beacons do not echo.  `LAYOUTS` 0/1/3/6/12/24.  Rear sector = beams with \|c\| ≥ 150° (only 6+ have one) |
| `src/sensors/vp_input.py` | camera + rangefinder → neurons; readouts |
| `src/sensors/fisheye.py` | `sense()`; odour cue = **strongest single blob** (winner-take-all) |
| `src/sensors/_fisheye_cpp/` | C++ renderer, `build_cpp.bat` (MSVC + pybind11) → `_fisheye.cp313-win_amd64.pyd` |
| `src/environment/room_file.py` | rooms as files, `results/rooms/room{N}.npz`; `FLY_CLEARANCE = 0.5`; rooms 0,5,6,7 re-flagged (old flags in `results/rooms_bak/`) |
| `src/core/subnet_file.py` | subnet pickle cache `results/subnet/vp_h4_rho0.50_*.pkl` (rebuilt on first run) |
| `experiments/diag/` | the diagnostics behind every number below (`diag_*.py`) and the figure scripts (`plot_v4.py`, `plot_room6.py BEFORE AFTER`) |

Circuit wiring (MaleCNS VP subnet, 34,125 neurons, 9,188 visual projection
neurons injected, ρ 0.50):

| Signal | Enters / read from | Notes |
|---|---|---|
| scene luminance | VP neurons by measured receptive field | unchanged from 09-20 |
| odour cue (virtual attraction cue) | ORN | strongest blob only.  The intensity-weighted mean pointed *between* beacons, 24–34° off the nearest in the first seconds; WTA: 83% → **100%** within 15° of the nearest |
| threat | **LC4 + LPLC2 (311 cells)**, each cell from the beam **nearest its own receptive direction** | looming, not nearness: `min((0.75 s / ttc)², 4)`, `ttc = range / (speed · cos bearing)`.  Side wall at 90° = 0 |
| intended turn | **DNa01/DNa02** (L2 / R2) only | the pooled 1,304-DN readout turned LC4 threat into a 4,000 °/s turn *toward* the wall |
| escape | DNp01/02/04/06/11 | **recorded only**, not used in control |
| gyro | Johnston's organ push-pull (user's choice) | measured ~0 effect on DNa01/02 (±0.3 °/s).  Left as is; ask before rerouting |
| ~~alternation push~~ | ~~PFL3~~ | removed in v12, see below |

Drone layer constants: `R_MAX 90°/s`, `D_STOP 0.45 m` (body 0.25 + 0.2),
`V_BACK −0.4 m/s` only if rear > D_STOP, `SWEPT 0.40 m` half-width,
`A_BRAKE 2 m/s²` (half the airframe's 4), `V_CRUISE 2 m/s`, speed
`v = min(2, √(2·2·(d_hit − 0.45)), R_MAX/|k|)`, in-place turn at the full
intended rate when v < 0.3, committed saccade until 0.75 m open ahead, 2 s
world-frame obstacle memory.  Flights are 120 s (1,200 steps).

### Version history (room 6 unless stated; "found" of the reachable beacons)

Keep this table growing.  Each version changed the listed things and nothing
else; every earlier CSV/NPZ is kept as `results/*_vNN.*`.

| Ver | Change | Connectome found, beams 0/1/3/6/12/24 |
|---|---|---|
| v1 | structure B, room 0, 60 s, pooled DN steering | 3/3/0/0/0/0 of 5 (3+ beams: stalled, turning into walls) |
| v2 | threat per cell by direction; steering from DNa01/02 | 1c/1/4/3/0/– (threat readout silently off; run cut by Python loss) |
| v3 | threat calibration defined in the circuit (L 0.229, R 0.177) | 3/3/3/2/1/2 of 5, **0 collisions** |
| v4 | **4 rooms (0,5,6,7)**, 120 s, D_STOP 0.8→0.45, reach clearance 1.4→0.5, reachable/walled counted apart | totals of 23: conn 6c/4/9/10/11/13 · planner 12c/15/15/15/15/14 · centroid 3c/7/10/8/8/9 |
| v5 | stalled → in-place turn at intended rate; committed saccade | 0c/**6**/5/3/1/2 (was 0c/0/2/0/1/2) |
| v6 | speed from predicted collision on the intended arc (was ±60° nearest range) | 5/5/5/5/5/5, 0 collisions, 184–203 m (was 44–151) |
| v7 | each LC4/LPLC2 cell takes its nearest beam (12 beams had fed 135/311 cells, 24 fed all 311 → threat doubled with beam count) | 5/5/5/5/5/5 |
| v8 | alternation v1: JO push after a 10 s leaky integral of 360° | 5/5/5/5/5/5 — **never fired** |
| v9 | threat = looming; drone layer's threat bend removed (centroid keeps its own, `K_AVOID`); odour WTA; alternation over a 20 s window; 2 s obstacle memory; `od_l/od_r` rename | 0c/5/**6**/**6**/5/5 (only v5 1-beam had reached all 6 before) |
| v10 | alternation push moved JO → PFL3 (JO measured ~0 on DNa) | 0c/5/6/5/5/3; fired 1/0/3/1/3 — corridor loop broken but sweeps wrecked |
| v11 | alternation only if the last full turn stayed within 3 m | 0c/5/6/6/5/5; fired 1/0/0/0/0 — 12-beam corridor loop not caught |

Figures for each row: `results/sweep_vNN_paths_room6.png` (all arms × beams)
and `results/sweep_vNN_connectome_*_vs_vNN_room6.png` (before/after);
v4: `results/sweep_v4_paths_room{0,5,6,7}.png`, `sweep_v4_summary.png`.

`c` = collided.  Planner and centroid from v9 on: 5 and 4 at every beam count
≥ 1, both collide at 0 beams.

### What is settled, with the number

* **Zero range sensors is not enough.**  v4: every arm collided in all four
  rooms at 0 beams.  The 0-beam successes in room 6 (v6–v8) were luck: fastest
  and straightest (1.69 m/s, 21°/m turning vs 29–35°/m with beams) and the
  *averaged* odour cue happened to keep it 0.82 m from walls; with WTA it
  grazed a pillar (0.27 m) and hit.  `diag/diag_zero.py`.
* **Rear beams matter more than beam count.**  1 and 3 beams have no rear
  beam, so no back-off: 70–82% of the flight stalled in corners (v4, rooms
  5–7) until the in-place-turn fix.  `diag/diag_trap.py`.
* **The in-place turn had been cut to 15% of intent** (`k × 0.3 m/s`).  Fixed
  in v5.  Before it (room 5, 3 beams) the longest stall was 58 s with 0° net turn.
* **More beams made it slower** because the old speed law braked for side walls
  inside ±60°.  Path-based speed (v6) removed the paradox.
* **Signal strength must not depend on beam count** (user: "신호로 들어가는
  세기는 동일하게 해야지").  Nearest-beam per cell (v7).
* **Reached beacons already disappear** (`_finish()` rebuilds the world).
  Lingering near them (conn 18%, centroid 17%, planner 8% of time within 3 m
  after 2 s) is corners/pockets, not attraction.
* **Paths are chaotic.**  12 vs 24 beams diverged 3.4 s after start.  One
  flight per condition cannot rank versions; differences of ±1 beacon are
  noise.
* **Training was tried and dropped** (before structure B): 327 cell-type parameters
  (log_tau, bias, log_gain × 109 groups) on the 68,045-neuron optic-lobe
  subnet.  Steering over-fitted (held-out r 0.175); looming never learned
  (r −0.085 → +0.029, then brake-only head also failed).  Gradient window was
  30 ms against a 100 ms frame — a known flaw if this is revisited.  That is
  why range comes from a sensor.
* **CO₂/olfactory neurons as avoidance input: rejected by the user** ("이산화탄소
  감지 뉴런 말고 다른 뉴런으로").  All VP types *attract* when excited; the
  sign-correct avoidance route is LC4/LPLC2 → DNa, which is what is used.

### OPEN — the decision waiting on the user (2026-09-28)

**The loop problem.**  After 5 beacons the 12/24-beam flights circle in the
left corridor of room 6 for the last ~40 s.  Two geometric triggers failed:
angle only (v10) fired on normal room-scale sweeps and broke good routes;
angle + 3 m radius (v11) missed the corridor loop, whose turns span 5.2–5.7 m.
Offline (`diag/diag_spread.py`): the start-to-end gap of each full turn is
0.0–2.5 m in the corridor loop and mostly 3–26 m in sweeps, but some sweeps
also close at 0.2–0.4 m (in-place rotations).  Any threshold is fitted to room 6.

The user asked: **"애초에 반경과 각변위로 설정하려는 시도가 잘못되었을지도...?"**
My answer (agreed in substance, awaiting their choice):

1. What I built is an anti-circling device, not spontaneous alternation.  Real
   alternation is a 1-bit memory at **choice points** ("last time I turned
   left → now right").  Candidate: at the moment a committed saccade starts,
   bias PFL3 opposite to the previous saccade.  No threshold.  May not break
   this loop (it circles without being blocked).
2. The loop is probably caused by the **odour cue pulling toward a beacon
   visible through the partition**; reversing the turn does not remove the
   pull (v10 reversed it and the loop came back).  Candidate: **ORN
   adaptation** — a cue held for a long time fades, recovers when the scene
   changes.  Targets the cause, no geometric threshold.

Recommended: remove the geometric alternation, try (2) alone on room 6, then
(1) separately.  **Nothing is implemented yet.**

**UPDATE, same day — (2) is refuted before running it.**
`diag/diag_odour.py` re-rendered the looping flights: after the fifth
beacon the odour cue is **exactly zero for the rest of the flight** (v11 12
beams: none after t = 65 s; v9 24 beams: none after t = 78.6 s).  Beacon 2 is
never visible from the corridor.  Nothing pulls; there is nothing to adapt
to.  The loop is what the circuit does **with no cue at all** — a standing
turn bias under vision + threat alone.  Next step proposed to the user:
log the intended turn during the loop (is it a constant offset?) before
choosing a fix.

**Observed (geometry only, `diag/diag_loop.py`, `plot_loop.py`):** the loop
is a clockwise orbit around the pillar column at x ≈ −13.5, not a corridor
bounce.  Speed 2 m/s throughout, nothing ever within 2 m ahead, so the drone
layer never stopped or saccaded — every turn is the circuit's.  96% of turns
are RIGHT; mean −22 °/s in the open, −46 °/s with a wall 2–4 m ahead.  v9
24 beams: 79% right.  Non-looping v11 3 beams: 48/52, mean ≈ 0.

**Code reading, candidate causes of the right bias (`diag/diag_lr.py`):**
1. Threat input is lopsided: LC4+LPLC2 with a receptive direction are
   165 left / 146 right (LC4 71 / 55; within ±30° of ahead 17 / 14).  Left
   cells see only the left half-field, and left threat turns the drone
   right (away).  Threat is injected per cell, not normalised per side, so a
   wall dead ahead drives the left side harder → turns right.  Every other
   lateral input (ORN 71/131, JO 14/5) IS normalised per side.
2. `calibrate()` takes `zero` at odour bearing 0 with a beacon 25 m ahead.
   With no beacon in view the ORN drive drops to 0 on both sides, so any
   part of `zero` that came from the ORN drive becomes a standing turn.  Size
   and sign unknown.  `calibrate_alt` already computes exactly this no-cue
   turn (`u0`) at run start but never prints it.
**Measured (`diag/diag_bias.py`, static, u in deg/s, + = left):** candidate
2 is refuted — no cue gives +2.9.  The bias is the **circuit's own mirror
asymmetry**, in threat AND vision AND odour:

| probe | u |
|---|---|
| equal threat on all LC4/LPLC2, 0.25 / 1.0 | −21 / −78 |
| left cells only / right only (0.25) | −81 / +62 (1.32×; counts are 1.13×) |
| pillar dead ahead 2 m | −35 |
| mirrored pillar pair | −23 (vision alone −17) |
| corridor, walls 1.5 m each side | −26 |
| pillar 45° left / right | −56 / +30 |
| beacon 60° left / right (odour + vision) | +77 / −61 |
| gyro, turning right at 90 °/s | +0.3 (JO still has no effect) |

The subnet IS built with `symmetrise=True` (`malecns.mirror_average` +
`balance_hemispheres`), but only cells with a known partner are averaged;
LC4/LPLC2 are 165 vs 146 and the VP cells are mostly unpaired, so the
residue stays.  **Tried and reverted:** re-weighting cells per 30° azimuth
band so each mirror pair turns equal and opposite.  The circuit is
nonlinear: the pair went −23 → −6 but the corridor −26 → +18 and the beacon
sum +16 → +56; the frontal threat band (−4.9 vs +0.2 °/s) cannot be fixed
by weights at all.

**v12 (cleanup, no behaviour change):** geometric alternation removed (it
never fired at 12 beams; the 12-beam room-6 flight is bit-identical to v11,
196.0213 m); dead lamp-range/τ/urgency code removed from `VPInput`
(computed every step, read by nobody); `cue_frac` and `corr` now use the
odour cue — `weight` was the wall-laden centroid's, so cue_frac read 1.0 in
every flight.  Corrected values for that flight: cue_frac 0.54, corr 0.22
(the intended turn tracks the goal's bearing only weakly — worth a look).

**Input equalised (user's instruction, "좌우 신호 세기를 동일하게 분배"):**
`vp_input.mirror_weights` — per 10° azimuth band, cells looking at +az and
−az share the same TOTAL drive (weight = mean count / own count), for
vision and threat; odour and gyro were already per-side normalised.  Scenes
after: no cue +3.2; pillar ahead −35 → **−20**; pair −23 → **−16**; pillar
45° L/R sum −27 → **−18**; but corridor −26 → **−34** and beacon ±60° sum
+16 → **+34**.  Equal input does not give a symmetric turn: what remains
is the circuit's own response asymmetry.  Kept, because equal input is the
correct rule regardless.

**v13 = equal input, flown in room 6** (`sweep_v13_*`): connectome
5/6/5/5/5/5 for 0/1/3/6/12/24 beams, no collisions (v11: 0c/5/6/6/5/5).
The first half of every route is now the same clockwise tour; 3 and 6
beams lost the 6th beacon, 1 beam got it.  The left-pillar orbit now
appears at 0, 3, 12 and 24 beams, turns 89–100% RIGHT after 70 s.
**It happens at 0 beams too, where the threat input is zero** — so the
orbit is driven by VISION's asymmetric response to walls (corridor −34
°/s), not by threat.  Input distribution cannot fix it; the mirror twin
(below) would.

**v14 = MIRROR TWIN, implemented and flown (user: "시각도 거울 쌍둥이로").**
`VPInput.twin = True` (set in `cx_vp_room.main`): `drive()` also builds
`d_mirror` — each vision cell reads the pixel at −az (the camera grid is
symmetric), threat cells take the beam nearest −az, odour frac and gyro
flip sign.  The runner steps both as batch 2 (`twin()`), steers on
`(seen − mirror)/2` (`steer()`), so `zero` is 0 by construction.  Static
check (`diag_bias.py`): symmetric scenes −1.8…+1.5 °/s (were −16…−34),
beacon ±60° exactly ±68.8; with twin off every number is unchanged.
Room 6 (`sweep_v14_*`), connectome 0c/3/5/5/5/**6** for 0/1/3/6/12/24
beams, no collisions from 1 beam up.  **The orbit is gone at every beam
count**; left share of turns after 70 s 24–55% (was 0–11%).  24 beams:
all 6 in 79 s over 100 m — the best flight so far.  1 beam dropped to 3
(roamed the top corridor); 0 beams collided at 3 s (no sensor, as in v4).
Room-6 sweep now ~9 min (connectome ~2× per step).  Single flights: the
next step is the multi-room, multi-heading evaluation.

**Why v14 looks "much weirder" (user) — the ODOUR CHANNEL IS DEAD.**
`diag_compare.py v13 v14`: goal visible longer (cue 0.61–0.75) but the turn
tracks it less (corr 3 beams 0.42 → 0.18) and zig-zags more.  In the
12-beam flight a beacon sat at +103…+113° for 6 s (t = 23–29 s) while the
drone flew away from it.  `diag_pose.py v14 12 T` at t = 22/24/27 s:
removing the odour changes nothing; odour alone gives +0.0.
`diag_orn.py`: odour alone, beacon 60° left, versus strength — 0.0 at the
masses a beacon has at 10–25 m, **at most +3.8 °/s** (strength 0.5), falling
again above that.  ORN barely reaches DNa01/02.  It was measured strong
only against the old pooled 1,304-DN readout; nobody re-measured after the
readout moved to DNa01/02 (v2) — the same silent death as the JO gyro.
So since v2 **all goal attraction has come from VISION of the bright
beacon**, which competes with walls; the right bias used to hide it.

Proposal to the user (open): deliver the virtual attraction cue as a
goal-bearing push-pull into **PFL3** (the central-complex goal → steering
path of Westeinde 2024, this project's own Stage 0 core; PFL3 one-sided
moves DNa01/02 by ~3,000 °/s at 1/cell), instead of ORN.  HANDOFF already
listed ORN vs goal representation as an open modelling choice.

(Superseded proposal, kept for the record:) a *mirror twin* — run a second copy of the
circuit on the mirror image of every input (camera flipped, beams flipped,
odour and gyro sign-flipped) and steer on `(u − u_mirror)/2`.  Exact
mirror symmetry for any scene by construction; the real fly is bilaterally
symmetric, and the residue is reconstruction asymmetry.  Cost: ~2× circuit
time (room-6 sweep ~5 → ~9 min).  Not implemented.  After that, finish 2D with
the proper evaluation: rooms 0/5/6/7 × 4 start headings × beams 1/6/24 ×
alternation on/off, then path efficiency (path length / shortest path).

Later, the user's curiosity: 3D.  Today it is 2D flight in a 3D world (climb
command always 0; horizontal beams only).  First step would be to measure
whether the circuit gives an up/down signal at all (DNs split by receptive
elevation, beacons above/below) before building anything.

### Working with this user — rules learned the hard way

These were stored in Claude's local memory on the original PC; they are
copied here so a new chat does not lose them.

* **Read the code before running anything.**  Write down the number you
  expect before launching a simulation; if you cannot predict it, the
  experiment is not designed yet.  Runs cost the user minutes; reading costs
  nothing.  n = 1–3 is not a measurement.
* **One room, fast cycles** while debugging (room 6 is the worst case and the
  current bench).  Widen to four rooms only when the behaviour works.
* **Do not fix what was not asked** ("고치라고 말도 안 했는데 왜 고치는데?").
  Propose, predict, wait for the yes.  When the user asks for a list of fixes
  to be done together (e.g. "V9에서 남은 문제점 좀 수정해보자"), do all of
  them *before* running.
* **"멈춰" means stop immediately** and listen.
* **Show results as images** (path grids, before/after) — the user reads
  paths, and caught several bugs from them.  Save every run as `_vNN` before
  the next one overwrites it.
* Answer in Korean; the user addresses decisions directly and expects a
  recommendation, not a survey.
* The master design document is
  `C:\Users\최성준\Downloads\drosophila_navigation_handoff_master.md` (not in
  this repo).  Its decisions are settled.

### Reproduce the current numbers

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m sim.drone_layer
PYTHONPATH=src .venv/Scripts/python.exe -m sensors.rangefinder
.venv/Scripts/python.exe experiments/cx_vp_room.py --room 6 > results/sweep_v12_room6.log
.venv/Scripts/python.exe experiments/diag/plot_room6.py v11 v12
```

The first two lines are the self-checks (both print "demo ok").  A full room-6 sweep takes
about 5 minutes; the first run on a new machine also rebuilds the subnet
cache.

## 2026-09-20 — sensing rebuilt end to end

The day started with the connectome unable to steer in closed loop and ended
with it flying a closed room without collision.  Six of my own hypotheses were
measured and rejected on the way; what follows is what survived.

### The one that explained everything: a sampling delay

A recurrent net has effective time constant `tau/(1-rho)`.  At rho 0.9 that is
200 ms against a 100 ms control cycle.  Measured step response:

| rho | settles in | value at 100 ms (% of final) |
|---|---|---|
| 0.90 | 695 ms | **-105%** |
| 0.50 | 85 ms | +97% |

The readout was sampled seven settling times early, while the transient was
still swinging through the opposite sign.  Every closed-loop inversion all
day was this.  **rho 0.50** fixed it: standing corr +0.959, flying +0.971,
where rho 0.9 gave +0.948 standing and **-0.503** flying.

Rejected before finding it: gain magnitude, gain sign, state reset, state
leak, DNa02 readout, pooled readout, vision off, readout adaptation.

### Sensing: three cameras of ray casting -> one fisheye image

| | before | after |
|---|---|---|
| lamina | 6,199 x 5 rays, 49.6 ms | image lookup, 11.5 ms |
| cue | 69 samples per target, 34.2 ms | read from the same image |
| cost with 6 targets | grows with target count | constant |
| target position | passed to the sensor | **never given** |

The old cue sensor took the target's world position as an argument, projected
it, and tested occlusion -- information a real drone cannot have, at a cost
proportional to how many targets exist.  The image has none of that.

### The centroid formula was being applied to the wrong thing

`u_c = sum(I_i u_i)/sum(I_i)` was applied to a THRESHOLDED detection, which
discards every wall at the first step.  Applied to the whole image with
`w = max(I - sky, 0)`:

| scene | blob (old) | rectified (new) |
|---|---|---|
| target +30, clear | +30.8 | +30.8 |
| target +30, wall on the right | +82.9 (signed) | **+30.8** |
| wall left, no target | none | pushes right |

Weight mass in "target +30 with a wall": target 35.7, wall 256 with signed
weights (wall wins 7:1), wall 0 with rectified.  Sky is flat and cancels
either way -- the user's original point.

### Injecting at the visual projection neurons instead of the lamina

The optic lobe is 44,654 cells whose dynamics this rate model has no reason
to reproduce, and measurement said it got the sign wrong: a wall closing from
12 m to 2 m on the LEFT made the steering readout go MORE positive, i.e. turn
into it, while growing 50-fold in magnitude.

Receptive fields are recovered from wiring, not assumed:
`direction(j) = sum_i w(i->j) direction(i) / sum_i w(i->j)` over lamina cells.
81% of projection neurons have a direction after 2 hops, 100% after 3,
spanning -137..+135 deg, and left-side cells average +55 deg while right-side
average -58 -- anatomy falling out of the wiring.

| | lamina injection | projection-neuron injection |
|---|---|---|
| neurons | 66,601 | 34,125 (optic lobe cut) |
| bearing sweep range | 4.4e-07 | **1.7e-04** |
| wall signal at 2 m | 7.5e-08 | 1.5e-04 |
| control step | 120 ms | 54.5 ms |
| offset/range | 0.59 | **0.05** |
| gain needed | -3e6 | **-6.5e3** |

And the sign problem dissolved: target left gives -8.6e-05, wall left gives
+1.5e-04 -- **opposite signs**, so one gain does approach AND avoidance.

**What this gives up:** the model no longer computes vision, it computes what
the brain does with extracted visual features.  The optic lobe became an
assumption.  Say so with any result that uses it.

### Two output channels, not one

A wall dead ahead is left-right symmetric and cannot appear in R-L at all --
measured, R-L moves 2.6e-09 as a wall closes 12 m to 3 m while R+L moves 5x.
Speed now scales with frontal blockage, for every arm, in the runner.  That
is what produces "stop, then turn".

### The world had to carry range

Flat surfaces have no feature to correspond between eyes or track between
frames: a disparity estimate on them returned the same 3.0-3.5 s from 20 m to
3 m, which was pixel quantisation.  Added world-fixed surface texture, then a
**lamp on the drone**, which turned out far better than stereo:

| distance | lamp contribution |
|---|---|
| 2 m | 0.151 |
| 5 m | 0.029 |
| 12 m | 0.002 |
| 18 m | 0.0005 |

300x over the working range, monotonic, no motion needed, no correspondence,
and it costs nothing because it is computed inside the render that already
runs.  Stereo still fails (correlation window includes sky, which has zero
disparity and dominates) and expansion-rate tau saturates in the near field.

### Where it stands, room 0 (161 obstacles, 5 reachable beacons)

| arm | found | collided | steps |
|---|---|---|---|
| planner (full information) | 3/5 | yes | 532 |
| **connectome VP** | **2/5** | **no** | 600 |
| centroid reference | 2/5 | no | 600 |

The connectome matches the reference and finds its first beacon sooner
(321 vs 356).  It has never collided in this configuration.

### Still open

- stereo correlation window must exclude sky
- tau from expansion saturates when the wall fills the field
- path efficiency is poor: 60 s for 2 beacons, large detours
- the lamp needs smoothing -- abrupt brightness steps should be filtered
- all of this is n=1; nothing here has been run over many rooms

## The closed-loop inversion was a sampling delay - 2026-09-20

Every closed-loop failure of the vision network traced to one thing.  The
readout encodes target bearing almost perfectly when the body is held still
(corr **+0.95 to +0.99**) and the correlation **inverts** as soon as the agent
flies (**-0.50 to -0.99**).  It inverted under gain up, gain down, gain sign
flipped, state reset, state leak, DNa02 readout, pooled readout, vision off,
and readout adaptation - five hypotheses measured and rejected.

A recurrent net has an effective time constant of `tau/(1-rho)`.  Measured
step response of this network:

| rho | effective tau | settles in | value at 100 ms (% of final) |
|---|---|---|---|
| 0.90 | 200 ms | 695 ms | **-105%** |
| 0.70 | 67 ms | 195 ms | +119% |
| 0.50 | 40 ms | 85 ms | +97% |
| 0.30 | 29 ms | 95 ms | +93% |

The control cycle is 100 ms.  At rho 0.9 the readout is sampled **seven
settling times too early**, while the transient is still swinging through the
opposite sign.  The command was the negative of what the circuit computes.
The circuit was never wrong - the sampling instant was.

**Fix: rho 0.50.**  Same room, everything else identical:

| rho | standing corr | flying corr | steps survived |
|---|---|---|---|
| 0.90 | +0.948 | **-0.503** | 21 |
| 0.50 | +0.959 | **+0.971** | 68 |
| 0.30 | +0.967 | +0.981 | 56 |

This also explains the earlier rho sweep that found 0.90-0.99
indistinguishable: all were far too slow, and only the untried direction -
downward - mattered.

**Rule this produces:** a recurrent readout must be checked for settling time
against the control period before any closed-loop conclusion is drawn.  An
open-loop sweep allowed to settle longer than the loop does is measuring a
different system.

**What remains:** steering now tracks the target (+0.971) and survives 3x
longer, but flies into walls - the same failure as the P reference.  Approach
works, avoidance does not, with the eyes on.  The cue/vision balance (2.59)
was measured at rho 0.9 in an empty scene and must be redone.

## The specified task: find the target in clutter — 2026-09-20, `experiments/cx_search.py`

Everything before this ran on `corridor_world`, which measurement showed was
not the task: target 12 m dead ahead, in frame at t=0 in 94% of episodes.
`cluttered_world` is 45 obstacles, target 22-32 m at ANY bearing, heading
anywhere in the circle, in frame at t=0 in 25%.  Every map proven solvable by
grid BFS first.

| arm | reached | collided | final dist | target seen |
|---|---|---|---|---|
| planner, full information | 0.875 | 0.125 | 1.81 m | 66% |
| P + oracle obstacles | 0.375 | 0.562 | 38.9 m | 33% |
| **connectome, eyes + cue** | **0.375** | 0.625 | 33.3 m | 27% |
| P blind, cue only | 0.125 | 0.875 | 36.6 m | 23% |

The blind reference falls from 0.79 to 0.125 as soon as the target is not
already in frame.  The connectome holds 3x that, and matches a controller
that is handed ground-truth obstacle positions.

**Search appears, and was never designed.**  In 2 of its 6 wins the
connectome found a target invisible from the start that no other arm found
(maps 14 and 8; in map 14 both references flew to y = -38, the opposite way).
Hypothesis: with no target in view the cue is silent but the eyes are not, so
obstacle silhouettes keep turning the agent and the turning sweeps the field
of view.  **Unverified** — the vision-off control has not been run.

**Not supportable yet:** these references are hand-written, not learned.  The
GRU baseline does not exist.  Nothing may claim the connectome beats "기존
인공지능" until it does.

**Metric defects to fix before publishing any of this:** `detour` against a
4-connected grid path reads below 1.0 (P blind 0.727) because the grid cannot
go diagonally; and the planner ceiling loses 2 maps to its own crude
lookahead follower, not to the task.

## The connectome flies with its own eyes — 2026-09-19, `experiments/cx_vision_loop.py`

66,601 neurons, 1,230,464 edges, rho 922 -> 0.9.  All 6,199 lamina cells and
all 532 CX families inside.  Scene in through the eyes, cue in at FB5AB, turn
read at the descending neurons.  42.7 ms per control cycle.

**240 held-out episodes**, 10 seeds x 24, gain frozen at -3e6:

| arm | reached | collided | timeout | final dist |
|---|---|---|---|---|
| P + oracle obstacles (ground truth) | 0.938 | 0.000 | 0.062 | 1.88 m |
| **connectome, with vision** | **0.875** | 0.121 | **0.004** | **1.33 m** |
| P control, blind | 0.792 | 0.192 | 0.017 | 1.93 m |
| straight, never turns | 0.125 | 0.233 | 0.642 | 9.90 m |

Paired by seed against the blind P reference:

| | diff | better in | Wilcoxon |
|---|---|---|---|
| reached | +0.083 | **10/10** | p = 0.0020 |
| collided | -0.071 | 9/10 | p = 0.0039 |
| final distance | -0.61 m | 9/10 | p = 0.0098 |

Dropping seed 2, the one the gain was picked on: 9/9 on reach, p = 0.0039.

Against the ORACLE, which is handed ground-truth obstacle positions: the
connectome reaches less often (0.875 vs 0.938, p = 0.051) and collides more
(0.121 vs 0.000), but its paths are SHORTER (1.33 m vs 1.88 m, 9/10,
p = 0.0195) and it times out almost never (0.004 vs 0.062, 10/10, p = 0.0020).
It flies less safely and more efficiently than a hand-written avoider.  Path
efficiency is the deliverable, so that trade is the result, not a footnote.

**What had to be fixed to get here**, all recorded in `provenance/`:
`retinotopy_was_destroyed_by_reset_index` (the root cause — one line),
`the_scene_never_implemented_its_own_premise` (bright target, dark sky,
vertical beacon), `option_B_vision_restored_what_it_found` (lamina must
transmit contrast; the readout address must be the descending neurons).

## The connectome steers in closed loop — 2026-09-19, `experiments/cx_subcircuit.py`

532 neurons (FB5AB, PFNa/p/m, hDeltaC, FC2B/C, FB5N, PFL2/3), 2,697 edges at
>=2 synapses, spectral radius rescaled 60.3 -> 0.9.  Cue in at FB5AB, turn
read as PFL right-minus-left.

Validated on **10 held-out seeds, 240 episodes**, every setting frozen at what
seeds 0-1 chose (`experiments/cx_validate.py`).  Start 12.31 m, 2 obstacles:

| arm | reached | collided | final dist |
|---|---|---|---|
| connectome | 0.796 +- 0.069 | **0.100** | 1.93 m |
| connectome + heading | 0.804 +- 0.071 | 0.083 | 1.86 m |
| P control reference, kp 2 | 0.792 +- 0.071 | 0.192 | 1.93 m |
| straight, never turns | 0.125 +- 0.056 | 0.233 | 9.90 m |

Paired by seed, connectome against the P reference:

| | diff | better in | Wilcoxon |
|---|---|---|---|
| reached | +0.004 | 5/10 | p = 0.65 |
| distance | -0.01 m | 6/10 | p = 0.77 |
| **collisions** | **-0.092** | **8/10** | **p = 0.031** |

**What replicates:** the circuit steers.  0.796 against 0.125 for never
turning is not ambiguous.  And it collides about half as often as the P
reference, consistently, with no obstacle sensing at all -- it turns less
violently, and that is the whole mechanism.

**RETRACTED:** the first report of this said the connectome "beats the
reference on all three".  That was one seed, 24 episodes.  On 10 held-out
seeds reach and distance are TIES (p 0.65 and 0.77).  Only the collision
difference survives.

**What made it work, and it was not what was predicted.**  The readout carries
a DC term that scales with total cue drive and is twice the whole
bearing-dependent range (3.38e-4 against 1.77e-4).  Apparent target size
changes 5x over the approach, so closed loop the circuit reported how BRIGHT
the target was instead of where it was.  That, not the neuron model and not
the graph cut, is what destroyed the three earlier attempts whose open-loop
sweeps had looked monotonic.  Pinning the magnitude so only the bilateral
ratio varies fixes it.

**Refuted in the same run:** injecting heading at the 396 PFNs, the thing this
experiment was built to test, moves reach 0.67 -> 0.71 at n=24 — inside the
noise.  The readout moves 8x less for heading than for bearing.  Caveat:
heading goes in as a bilateral imbalance, not a bump across columns, because
the annotation has no column index.

**Two rules this produced, both now in `provenance/`:**
- An open-loop bearing sweep must be taken AT SEVERAL DISTANCES, or it cannot
  see this class of failure.
- Every closed-loop table carries a P reference and a never-turn baseline.  At
  8 obstacles the P reference itself only reaches 0.38, so a connectome reach
  of 0.00 there meant nothing.

## The comparison arm — the fly beats a P controller, with noise on

Master doc sections 15-16 define condition **D: generic controller baseline**,
*"fly-derived circuit가 일반 controller보다 특별한 robust behavior를 보이는지
비교"*.  It is implemented: `stage1_body_sweep.py --controller {fly,p,data}`.

Measured, 180 paired bodies, **noise off**:

    fly 0.859   vs   P 0.793
    59 cells fail for BOTH (body-attributable)
    15 fly-only worse      57 P-only worse

The fly circuit wins.  Condition D had never run with noise on, which was a
real hole: noise-on is exactly what retracted this project's other headline
("body limits rescue the core").

**Run 2026-09-17, and it HELD.**  36,288 trials, `--controller p
--noise-seeds 8 --plugin rate_clip --tag noise_pcontrol`, paired against the
`noise` fly sweep:

    noise ON      fly 0.825   vs   P 0.772
    cells         fly better 73 / P better 8 / equal 99
    tau= 1  0.913 -> 0.845     tau= 8  0.861 -> 0.841
    tau= 2  0.912 -> 0.847     tau=16  0.759 -> 0.702
    tau= 4  0.909 -> 0.857     tau=32  0.595 -> 0.539

Both controllers drop under noise, the gap narrows (0.066 -> 0.053), but the
direction does not flip and **the fly wins at every level of tau** — not in one
corner of the grid.  73 cells to 8 is not a marginal lead.

Still true: this is a P controller, not "기존 인공지능".  A trained baseline is
what the current work is building toward.

And "기존 인공지능" needs more than a P controller.  Master doc section 15 calls
P the *first* baseline.  Section 4.6 names the paper this study must separate
itself from — **FLYNN (arXiv 2026)**, a *learned* connectome-topology RNN for
robot navigation — so a trained baseline is required, not optional.

## Is it computable on this laptop?  Yes — measured 2026-09-17

`experiments/feasibility_bench.py`, median per call:

    core: Westeinde reduced            0.59 ms
    core: connectome forward           7.06 ms
    core: connectome fwd+bwd          30.44 ms per step
    sensing: analytic scene            0.28 ms
    sensing: plain camera (any res)   12.0  ms per camera
    sensing: flygym compound eye      33.2  ms
    body: abstract yaw plant           0.018 ms
    body: 3D point-mass flight         0.012 ms

At the core's own 10 Hz, 10 s episodes = 100 steps:

    3 cameras + connectome training    6.6 s/episode  -> 10,000 ep = 18 h
    analytic sensing + connectome      3.0 s/episode  -> 10,000 ep = 8.5 h
    analytic sensing + reduced core    0.09 s/episode -> 10,000 ep = 15 min

**Camera render cost is flat in resolution** — 12 ms at 32x32 and at 256x256
alike, because it is per-call overhead (scene update, GL context, readback),
not pixels.  So resolution is free; camera COUNT is what costs.  No CUDA on
this machine (`torch 2.14.0+cpu`), so GPU batch rendering is closed here.

### The fix that made training feasible

Connectome training was **1,911 ms per step** — forward 6.7 ms, backward
1,840 ms, a 285x ratio for a weight matrix that does not even require grad.
Torch's sparse-CSR autograd is the cause.  `_SpMM` in `core/flywire_rate.py`
keeps the transpose in CSR and does backward as one more CSR matvec:

    W  @ r  (CSR)     5.46 ms
    WT @ g  (CSR)     5.14 ms
    custom fwd+bwd   10.93 ms      <- 168x faster
    max |grad diff|   0.000e+00    <- bit-identical

22 days became 8.5 hours.  111 tests still pass.

## Still open, needing the user

1. ~~What the pheromone camera sees~~ — ANSWERED 2026-09-17: it is not a
   chemical sensor at all.  See the sensing section: a virtual attraction cue
   derived from visual target centroids.  Target = what the cue points at.
2. ~~Camera geometry~~ — ANSWERED: two visual cameras, for depth.
3. Which core carries the comparison — Westeinde reduced (master doc priority
   1, verified to 3.4e-14, 15 min per training run) or a steering subnetwork
   cut anatomically out of the connectome.  Building the environment and the
   path-efficiency metrics first is the ordering that does not depend on this.

## The question the connectome work was answering

Can a fly connectome, run as a computable circuit with only its **anatomical**
input and output designated, produce navigation?  Nothing between input and
output may be chosen: no hand-picked "steering circuit", no assigned function.

This is a real sub-question and the work below is real, but it is **priority 9**
and it serves the goal above rather than replacing it.

## Two stages, and which is live


**Stage 0/1 — the reduced model. DONE, closed, written up.**
Westeinde et al. 2024's 5-population steering circuit, reproduced bit-identically
(PFL activity 0.0 difference, steering 3.4e-14), then driven into 252 bodies.
Conclusions in `docs/stage1_conclusions.md`.  Do not reopen; it is the baseline.

**Stage 2 — the whole connectome. LIVE.**
Everything below is this.

## Dataset: MaleCNS, not FAFB

FAFB (`data/flywire/`) is brain-only.  It has no motor neurons, which is why an
earlier attempt had to invent a steering command out of 1,303 mixed descending
neurons.  **MaleCNS v1.0 (`data/malecns/`) contains the nerve cord** and is the
live dataset.

    166,700 neurons, 6.2M edges at >= 5 synapses, builds in ~4 s
    909 motor neurons, named by muscle ("Ti flexor MN") and leg (T1/T2/T3)
    13,161 vnc_intrinsic -- the CPG, as data rather than hand-written
    assignedOlHex1/2 -- the eye lattice, built into the annotation
    `group` -- explicit left-right pairs, which FAFB lacked

Public bucket, no login:
`storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`

FAFB code still exists and still passes its tests.  It is kept as the
reproduction record, not as a live path.

## What runs today

    scene -> 6,199 lamina cells -> 166,700 neurons -> leg motor neurons, 26 s

    src/core/malecns.py          loader, + mirror-average and per-stage balance
    src/core/flywire_rate.py     differentiable rate model, 141 trainable params
    src/core/null_wiring.py      degree-preserving and random-sparse controls
    src/sensors/flywire_eye.py   load_malecns_eye(): hex -> azimuth/elevation
    src/decoder/steering.py      read_motor(): per leg pair, per side
    experiments/flywire_train_pilot.py   training loop (still FAFB-wired)

`pytest tests/ -q` — 110 passing.

## Settled, with the number that settled it

- **Laterality.** Mirror-average where `group` pairs cells (75-88% of motor,
  descending, VNC), then rescale the residue per anatomical stage.  Every stage
  goes to R/L 1.0000 from 0.91-1.26.  Thresholding weak edges does NOT work
  (R/L flat at 1.05-1.10 from threshold 1 to 50).
- **Eye scale.** 5.0 deg per column, checked against the field of view
  (147-162 deg) and independently confirmed by flygym's 157 deg per eye.
- **Weight scale.** w_scale 0.005, not 0.01: at 0.01 the central brain's
  heading variation collapses to 0.9% against 44.8% at 0.005.  Calibrating on
  whole-network mean activity hides this, because the optic lobe dominates it.
- **Sign.** ACh/DA/OA/5-HT excitatory, GABA/Glu inhibitory, per presynaptic
  NEURON not per edge (the per-edge averages flip the sign on 9.6% of edges).
  Same rule Shiu et al. use.
- **Neuron parameters.** Shiu et al. 2024 `model.py`, read from their code.

## Open, and honest about it

- **`turn` does not cross zero.**  It correlates with sin(bar azimuth) at
  -0.67, but offset/range is 0.43.  The wiring is balanced to 1.0000 and
  matching the injected columns between eyes barely moves it (0.48 -> 0.43).
  So it is neither the wiring nor the input count.  Unexplained.
- **No heading bump.**  On FAFB, EPG never formed one.  Not yet re-checked on
  MaleCNS.
- **Training finds degenerate solutions.**  The loss descends only after the
  parameter scales are separated (bias shares the activity scale ~0.01 while
  log_tau/log_gain are exponents; gradient norms differ 13x).  It then reduces
  scale and offset rather than shape, and silences EPG.
- The task may not REQUIRE a bump: an instantaneous stimulus-response map has
  nothing to remember.  A closed loop with a persistent goal would.

## RETRACTED APPROACH — joint-level motor drive (kept as a record)

Everything in this section is a record of a WRONG architecture.  The result
that replaced it is the next section.  Read this only for the measurements,
which stand; the framing does not.

**The error.**  I read 909 motor neurons into 24 joint angles and drove
flygym's position actuators directly, then concluded the fly could not walk
because (a) a rate model converging to a fixed point has no rhythm and (b)
there was no proprioceptive feedback.  (a) is true of that architecture and
(b) was simply false: flygym's own controller reads thorax height, all six
tarsus heights and the ground-contact forces every physics step.

**What I should have done first.**  NeuroMechFly v2's descending interface is
TWO NUMBERS, `[delta_L, delta_R]`.  I had that fact recorded and built past it
instead of checking what the installed package and its repository actually
ship.  Rung 2 of the ladder -- look before you write -- applied and I skipped
it.

## The flygym loop, first version — joint-level, superseded

`flygym 2.1.0` + `mujoco 3.9.0` are installed.  It is NeuroMechFly v2, and the
two interfaces already match:

    get_ommatidia_readouts()  ->  our lamina injection     721/eye, 157 deg FOV
    set_actuator_inputs()     <-  our motor readout        48 leg segments

Mapping 1 — vision — DONE.  `src/sensors/flygym_bridge.py`.  flygym's
ommatidia centroids form a clean hex lattice (spacing CV 0.001, 5.7
neighbours), so the join needs no fisheye inversion: normalise both lattices
to the same field of view and take the nearest direction.  All 23,720 columns
join, angular error median 2.04 deg against a 5 deg interommatidial angle,
and 706 of 721 ommatidia are read.  Our lattice is denser, so ~27 columns
share an ommatidium -- duplicating a reading rather than inventing one.

Mapping 2 — motor — DONE.  `decoder.steering.joint_commands`.  Derived, not
assigned: Drosophila leg muscles are antagonist pairs and the type names say
so.  `Ti extensor` against `Ti flexor`/`Acc. ti flexor` is the tibia;
`Tr extensor` against `Tr flexor`/`Acc. tr flexor`/`Fe reductor` the femur;
`Ta levator` against `Ta depressor` the tarsus; promotor against remotor and
anterior against posterior rotator the coxa.  254 leg motor neurons become 24
commands (6 legs x 4 joints), keyed the way flygym names its segments
(`lf_tibia`, `rh_coxa`).  Left/right counts come out near-equal per joint,
which is a symmetry check as well.

Mapping 3 — timescales — DONE, and it was the easy one.  MuJoCo's timestep is
1e-4 s (`mujoco_globals.yaml`, measured, not the 600 Hz an earlier secondhand
note claimed) and the rate model's is 5 ms, so one brain step is exactly 50
physics steps, held constant across them.  `src/sim/flygym_loop.py`.

Two things flygym does NOT give you by default, both of which cost a debugging
round: a bare `NeuroMechFly` has **no joints, no actuators and no eye
cameras**.  `build_fly()` adds them -- `add_joints(Skeleton(ROLL_PITCH_YAW,
LEGS_ACTIVE_ONLY), NEUTRAL)`, `add_actuators(..., "position", NEUTRAL,
kp=30)`, `add_vision()` -- giving 42 position actuators (7 dof x 6 legs).

The actuators are POSITION actuators, so the input is a target ANGLE.  The
motor readout therefore enters as a **deviation from the neutral pose**, which
is also the honest reading: motor drive moves a joint away from rest, it does
not name an absolute angle.  Each command drives its joint's pitch dof; dof
names are `parent-child-axis` (`lf_trochanterfemur-lf_tibia-pitch`) and our
keys are the child, so the join is exact.  24 of 24 commands reach an
actuator.

    scene -> eye -> 166,700 neurons -> 24 joint commands -> 42 actuators
    20 brain steps (100 ms biological) in ~150 s = 1,590 s per biological s

### And here is the result that matters

`experiments/flygym_loop_control.py`, the paired comparison: identical body,
identical spawn, only `actuator_gain` differs.

    blocked    gain  0.0   moved 0.3808 mm
    connected  gain  0.3   moved 0.3807 mm   separation from blocked 0.00007 mm
    amplified  gain 30.0   moved 0.3770 mm   separation from blocked 0.00581 mm

The readout **does** reach the body -- 100x the gain gives ~83x the separation,
so the path is real and linear.  But 0.380 of the 0.381 mm is **gravity**.  The
brain contributes 0.006 mm at 100x gain.  The loop is closed and the fly is,
for practical purposes, not being driven.

Do not read "moved 0.381 mm" as locomotion.  This is the sixth time a single
pooled number would have said the opposite of the truth; the control is what
says it.

### Why the commands are 1e-3 — measured, `experiments/motor_drive_probe.py`

Three candidates were distinguishable, so they were measured rather than
argued: (A) global scale, (B) agonist/antagonist cancellation, (C) the motor
neurons are not driven.  Rate p99 along the anatomical chain:

    ol_intrinsic       2.10e-1     89,403 neurons, 62% above zero
    visual_projection  3.60e-2     <- 6x drop leaving the optic lobe
    cb_intrinsic       1.71e-2     <- 2x further
    descending_neuron  2.19e-2     no further loss; 1,314 cells, the only
                                   brain -> nerve cord path
    vnc_intrinsic      5.77e-3
    vnc_motor          1.77e-2     motor neurons are MORE active than the
                                   VNC interneurons around them

So not C: the signal reaches the legs, and only 18 of 909 motor neurons
receive nothing.  Not A either: the optic lobe saturates (max 0.9998).  The
loss is one ~10x attenuation crossing out of the optic lobe, then nothing.

**The real finding is in the input, not the rates.**  At every stage the
excitatory and inhibitory input arriving nearly cancel, and the residue is
always negative:

    ol_intrinsic       exc +3.61e-3   inh -5.66e-3   net -3.97e-4
    cb_intrinsic       exc +6.72e-5   inh -6.67e-5   net -1.48e-7
    vnc_motor          exc +8.77e-4   inh -8.67e-4   net -8.33e-6

At `cb_intrinsic` the median neuron's net input is **450x smaller than its own
excitatory input**.  Consequently the median rate is exactly 0.0 at every
stage past the optic lobe: ~60% of each stage is hard-clamped at the
rectification floor of `tanh(relu(v))`.

### The hypothesis this raises, which is bigger than the loop

A network resting at its rectification floor cannot represent a SIGNED
quantity — only the strongest input gets through, and always with the same
sign.  That is the same shape as two problems already open above:

- `turn` correlates with bar azimuth at -0.67 but **never crosses zero**
  (offset/range 0.43), and neither rebalancing the wiring nor matching the
  input columns moved it.
- agonist-minus-antagonist comes out one-sided and ~1e-3.

Both follow if the operating point is below threshold.  Real flies have tonic
excitation and neuromodulation setting a non-zero baseline; this model has
none — `bias` sits at its initialisation and nothing supplies a resting drive.

This is a hypothesis, not a result.  The test is a tonic-drive sweep: add a
constant baseline to every neuron, and ask whether signed output appears.

### The tonic sweep — hypothesis REFUTED

`experiments/tonic_sweep.py`, 12 headings x 200 ms, `results/tonic_sweep.csv`.
Offset/range below 0.5 would mean the turn signal crosses zero.

    tonic    cb active  vnc active   motor off/rng   corr w/ sin(heading)
    0          0.548      0.401          1.73            -0.615
    0.001      0.637      0.687         55.5             -0.667
    0.003      0.725      0.810         70.1             -0.695
    0.01       0.821      0.903        119.9             -0.584
    0.03       0.892      0.945        172.6             -0.160
    0.1        0.935      0.957        278.2             -0.688

The mechanism works exactly as predicted -- tonic drive lifts the network off
the floor, 40% active to 96%.  **The consequence does not follow.**  Signed
output never appears; offset/range gets monotonically *worse*, by a factor of
160, and no level crosses zero.  Lifting everything equally raises the common
mode far faster than the heading-dependent difference.

So the floor is real but it is not what makes `turn` one-sided.  The offset
survives balanced wiring (R/L 1.0000), matched input columns, and now a 100x
range of operating points.  Three separate candidate causes eliminated.

Near-miss worth recording: a 4-heading, 50 ms smoke run of the same script
gave offset/range 0.223 at tonic 0 -- i.e. it appeared to cross zero and
appeared to refute the standing result.  With 12 headings and 200 ms the same
configuration gives 1.73.  Range over four points is not a range.  This is the
seventh time an under-split number pointed the opposite way.

## The habit this project runs on

Five times a pooled average has hidden the real behaviour: Stage 1's marginal
means (three separate retractions), pooling 1,303 descending neurons into one
scalar, calibrating w_scale on whole-network activity, averaging over central
cell types, and argmax over one-spike EPG histograms.

**Split every pooled number before believing it.**  Six claims have been
retracted in this project, all of them caught by re-measuring rather than by
reasoning.  Retractions are recorded in provenance, not deleted.

Related: `desktop-fly` uses the same two datasets but keeps 1,045 neurons and
hand-assigns their function (DNa01/02 steering, DNg11 grooming).  That is the
shortcut this project exists to avoid.
