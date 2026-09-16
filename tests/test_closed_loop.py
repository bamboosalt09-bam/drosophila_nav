"""Stage 0 part 2 tests: decoder, body, sensor, plugin and the scheduler.

Covers handoff document section 44 items 7-12 that apply without a constrained
body yet, and pins the two structural rules the study depends on:

    * actual motion is what feeds back, never the command (section 3.4)
    * the plugin is goal-blind and cannot navigate on its own (sections 3.5, 16)

Run:  python -m pytest -q
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from body.ideal_yaw import IdealYawBody
from core.calibration import calibrate, load
from core.westeinde2024 import CoreParams, WesteindeSteeringCore
from decoder.steering import SteeringDecoder
from environment.heading_task import HeadingTask, trials_from_errors
from plugins.passthrough import PassthroughPlugin
from sensors.ideal_heading import IdealHeadingSensor
from sim.closed_loop import resultant_length, run_closed_loop
from sim.interfaces import BodyModel, ConstraintPlugin, SensorModel
from sim.noise import (SourceNoiseSpec, deg_per_step_to_rad_per_s,
                       source_command_noise_deg)
from sim.source_closed_loop import run_source_loop, steering_lookup_table
from utils.angles import wrap, wrap_deg

REPO = Path(__file__).resolve().parents[1]
NORM_JSON = REPO / "configs" / "norm_constants.json"


@pytest.fixture(scope="module")
def norm():
    return load(NORM_JSON) if NORM_JSON.exists() else calibrate(CoreParams(),
                                                                verbose=False)


@pytest.fixture(scope="module")
def core(norm):
    return WesteindeSteeringCore(CoreParams(), norm)


@pytest.fixture(scope="module")
def decoder():
    return SteeringDecoder()


def _run(core, decoder, task, plugin=None, body=None, sensor=None, noise=None):
    return run_closed_loop(core, decoder,
                           plugin or PassthroughPlugin(),
                           body or IdealYawBody(),
                           sensor or IdealHeadingSensor(),
                           task, command_noise=noise)


# =========================================================================
# decoder
# =========================================================================
def test_decoder_matches_the_source_parameterisation(decoder):
    assert decoder.kappa_deg_per_step == 200.0
    assert decoder.T_core_s == pytest.approx(0.1)


def test_decoder_rate_and_increment_are_the_same_statement(decoder):
    """yaw_rate * T_core must equal the source's heading increment."""
    for s in (-1.0, -0.3, 0.0, 0.25, 1.0):
        inc_deg = decoder.delta_heading_deg(s)
        from_rate = np.rad2deg(decoder.yaw_rate(s) * decoder.T_core_s)
        assert from_rate == pytest.approx(inc_deg, rel=1e-12)


def test_decoder_is_linear_and_odd(decoder):
    assert decoder.yaw_rate(0.0) == 0.0
    assert decoder.yaw_rate(-0.4) == pytest.approx(-decoder.yaw_rate(0.4))


# =========================================================================
# body and sensor
# =========================================================================
def test_ideal_body_integrates_and_wraps():
    b = IdealYawBody()
    b.reset(psi=np.deg2rad(170.0))
    b.step(np.deg2rad(100.0), 0.2)          # +20 deg -> 190 deg -> wraps
    assert np.rad2deg(b.psi) == pytest.approx(-170.0, abs=1e-9)
    assert b.r == pytest.approx(np.deg2rad(100.0))


def test_ideal_body_reset_clears_state():
    b = IdealYawBody()
    b.reset(psi=1.0, r=5.0)
    b.step(3.0, 0.1)
    b.reset()
    assert b.psi == 0.0 and b.r == 0.0


def test_sensor_reports_the_body_not_the_command():
    b = IdealYawBody()
    b.reset(psi=np.deg2rad(42.0))
    assert IdealHeadingSensor().observe(b) == pytest.approx(np.deg2rad(42.0))


def test_components_satisfy_their_protocols():
    assert isinstance(PassthroughPlugin(), ConstraintPlugin)
    assert isinstance(IdealYawBody(), BodyModel)
    assert isinstance(IdealHeadingSensor(), SensorModel)


# =========================================================================
# the two structural rules
# =========================================================================
def test_plugin_signature_cannot_receive_a_goal():
    """Goal-blindness is enforced by the signature, not by convention."""
    params = list(inspect.signature(ConstraintPlugin.command).parameters)
    assert params == ["self", "r_brain", "r_actual"]
    impl = list(inspect.signature(PassthroughPlugin.command).parameters)
    assert impl == ["self", "r_brain", "r_actual"]
    forbidden = ("goal", "psi_g", "error", "target", "path", "heading")
    assert not any(any(f in p for f in forbidden) for p in impl)


def test_plugin_cannot_navigate_without_the_core(decoder, norm):
    """Handoff doc section 44 item 12 / section 16 'core removed' ablation.

    With the core output forced to zero the plugin sees only a zero command
    and the body's own rate.  Heading must not drift toward the goal, because
    nothing downstream of the core knows where the goal is.
    """
    class ZeroCore:
        def steering(self, heading, goal):
            return 0.0

    task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(120.0),
                       duration_s=20.0)
    res = _run(ZeroCore(), decoder, task)
    assert np.allclose(res.psi, res.psi[0], atol=1e-12)
    assert abs(np.rad2deg(res.error[-1])) == pytest.approx(120.0, abs=1e-9)


def test_actual_motion_is_what_feeds_back(core, decoder):
    """A body that refuses to move must keep the core's input constant.

    If the loop ever fed the command back instead of the body state, the
    observed heading would move even though the body did not.
    """
    class FrozenBody:
        name = "frozen"

        def __init__(self):
            self._psi = 0.0

        @property
        def psi(self):
            return self._psi

        @property
        def r(self):
            return 0.0

        def reset(self, psi=0.0, r=0.0):
            self._psi = float(psi)

        def step(self, r_cmd, duration):
            pass  # ignores the command entirely

    task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(75.0),
                       duration_s=5.0)
    res = _run(core, decoder, task, body=FrozenBody())
    assert np.allclose(res.heading_obs, res.heading_obs[0], atol=1e-15)
    # the brain kept asking for a turn, and the body never delivered one
    assert np.all(np.abs(res.r_brain) > 0.0)
    assert np.allclose(res.r, 0.0)
    assert np.all(res.brain_body_mismatch > 0.0)


def test_sensor_observes_the_body_every_cycle(core, decoder):
    """With an ideal sensor and an ideal body, observation == previous state."""
    task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(50.0),
                       duration_s=3.0)
    res = _run(core, decoder, task)
    assert np.allclose(res.heading_obs, res.psi[:-1], atol=1e-15)


# =========================================================================
# scheduler behaviour
# =========================================================================
def test_loop_converges_to_the_goal_from_small_errors(core, decoder):
    for e0 in (-60.0, -30.0, -10.0, 10.0, 30.0, 60.0):
        task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(e0),
                           duration_s=5.0)
        res = _run(core, decoder, task)
        assert abs(np.rad2deg(res.error[-1])) < 1.0, e0


def test_loop_respects_a_non_zero_goal(core, decoder):
    goal = np.deg2rad(115.0)
    task = HeadingTask(goal=goal, initial_heading=goal + np.deg2rad(40.0),
                       duration_s=5.0)
    res = _run(core, decoder, task)
    assert abs(np.rad2deg(res.error[-1])) < 1.0


def test_anti_goal_steering_is_numerically_zero(core, decoder):
    """Exactly 180 deg cancels by symmetry (handoff doc section 48)."""
    assert abs(core.evaluate_deg(180.0, 0.0).steering) < 1e-12


def test_anti_goal_equilibrium_is_unstable_and_amplifies_rounding(core, decoder):
    """The anti-goal point is an UNSTABLE equilibrium, not a resting state.

    Started exactly anti-goal the loop does not sit still: the residual left
    by floating-point asymmetry (order 1e-15) is amplified by a factor of
    about 4.7 per cycle until the trajectory escapes and converges to the
    goal.  That multiplier is 1 + k * d(steering)/de at 180 deg and is a real
    property of the model; the SEED, however, is rounding noise, so the escape
    direction is not portable across platforms or BLAS versions.

    Consequence for the study: the 180 deg condition must be driven by an
    explicit symmetry-breaking perturbation, never by whatever the floating
    point happens to do (see test below).
    """
    task = HeadingTask(goal=0.0, initial_heading=np.pi, duration_s=10.0)
    res = _run(core, decoder, task)
    mag = np.abs(res.steering)
    assert mag[0] < 1e-12, "steering at exactly 180 deg should be ~0"
    growth = mag[6:12] / mag[5:11]
    assert np.allclose(growth, growth[0], rtol=0.05), "growth is not geometric"
    assert 3.0 < float(growth.mean()) < 7.0, float(growth.mean())
    # having escaped, it ends up at the goal rather than anywhere else
    assert abs(np.rad2deg(res.error[-1])) < 1.0


@pytest.mark.parametrize("perturb_deg", [-1.0, -0.1, 0.1, 1.0])
def test_explicit_perturbation_sets_the_escape_direction(core, decoder,
                                                         perturb_deg):
    """With a deliberate perturbation the anti-goal case is reproducible.

    Starting just inside 180 deg, the loop must turn the short way round: the
    first command has the sign that reduces |error|, and the trajectory leaves
    the anti-goal neighbourhood.

    Where it ENDS is deliberately not asserted.  With the source gain k = 200
    the loop has two attractors -- the goal, and a period-2 cycle near
    +-100 deg -- and their basins interleave, so a start at 179 deg reaches the
    cycle while 178 deg reaches the goal.  That is a property of the source
    model measured in experiments/stage0_ideal_loop.py, not something this
    test should pretend away.
    """
    e0 = 180.0 - abs(perturb_deg) if perturb_deg > 0 else -180.0 + abs(perturb_deg)
    task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(e0), duration_s=10.0)
    res = _run(core, decoder, task)
    assert np.sign(res.steering[0]) == -np.sign(e0)
    assert np.abs(np.rad2deg(res.error)).min() < 170.0, "never left the anti-goal region"


def test_loop_is_reproducible_with_a_fixed_seed(core, decoder):
    task = HeadingTask(goal=0.0, initial_heading=np.pi, duration_s=10.0)
    outs = []
    for _ in range(2):
        rng = np.random.default_rng(12345)
        nd = source_command_noise_deg(task.n_cycles, rng, SourceNoiseSpec())
        outs.append(_run(core, decoder, task,
                         noise=deg_per_step_to_rad_per_s(nd, task.T_core_s)).psi)
    assert np.array_equal(outs[0], outs[1])


def test_different_seeds_give_different_trajectories(core, decoder):
    task = HeadingTask(goal=0.0, initial_heading=np.pi, duration_s=10.0)
    psis = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        nd = source_command_noise_deg(task.n_cycles, rng, SourceNoiseSpec())
        psis.append(_run(core, decoder, task,
                         noise=deg_per_step_to_rad_per_s(nd, task.T_core_s)).psi)
    assert not np.allclose(psis[0], psis[1])


def test_command_noise_length_is_checked(core, decoder):
    task = HeadingTask(duration_s=1.0)
    with pytest.raises(ValueError, match="expected"):
        _run(core, decoder, task, noise=np.zeros(task.n_cycles + 3))


# =========================================================================
# agreement with the source's own loop
# =========================================================================
def test_our_scheduler_matches_the_source_loop(core, decoder):
    """Condition A run through the general scheduler IS the source loop.

    Compared away from the anti-goal separatrix: within ~1 degree the two
    agree, the residual being the source's 1-degree heading quantisation.
    """
    hd_grid = np.arange(-180.0, 181.0, 1.0)
    table = steering_lookup_table(core, hd_grid, 0.0)
    for e0 in (-150.0, -90.0, -30.0, 30.0, 90.0, 150.0):
        src = run_source_loop(table, hd_grid, n_steps=100,
                              k=decoder.kappa_deg_per_step,
                              initial_hd_deg=e0)
        task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(e0),
                           duration_s=99 * decoder.T_core_s)
        res = _run(core, decoder, task)
        dev = np.abs(wrap_deg(np.rad2deg(res.psi) - src.hd_deg))
        assert dev.max() < 2.0, "e0=%s deviated by %.2f deg" % (e0, dev.max())


def test_pfl_scalar_controls_goal_holding(core, decoder, norm):
    """Reproduces the source's Figure 5 trend: higher S holds the goal better."""
    task = HeadingTask(goal=0.0, initial_heading=np.pi, duration_s=40.0)
    rng = np.random.default_rng(7)
    nd = source_command_noise_deg(task.n_cycles, rng, SourceNoiseSpec())
    noise = deg_per_step_to_rad_per_s(nd, task.T_core_s)
    rhos = []
    for S in (0.2, 0.6, 1.0):
        c = WesteindeSteeringCore(CoreParams(pfl_scalar_S=S), norm)
        rhos.append(resultant_length(_run(c, decoder, task, noise=noise).psi))
    assert rhos[0] < rhos[1] < rhos[2]
    assert rhos[-1] > 0.8


# =========================================================================
# noise and task helpers
# =========================================================================
def test_source_noise_spec_matches_the_notebook():
    s = SourceNoiseSpec()
    assert (s.noise_level_deg_per_step, s.fpass_hz, s.fs_hz, s.order) == \
        (10.0, 2.0, 10.0, 5)


def test_noise_conversion_round_trips():
    deg = np.array([1.0, -2.5, 0.0])
    rate = deg_per_step_to_rad_per_s(deg, 0.1)
    assert np.allclose(np.rad2deg(rate * 0.1), deg)


def test_trials_cover_the_requested_errors():
    tasks = trials_from_errors([-90.0, 45.0], goal=np.deg2rad(30.0))
    errs = sorted(round(float(np.rad2deg(t.initial_error)), 6) for t in tasks)
    assert errs == [-90.0, 45.0]


def test_calibration_and_test_initial_conditions_do_not_overlap():
    from environment.heading_task import (CALIBRATION_ERRORS_DEG,
                                          TEST_ERRORS_DEG)
    assert not set(CALIBRATION_ERRORS_DEG) & set(TEST_ERRORS_DEG)


def test_seeded_noise_is_paired_across_cells():
    """The sweep's noise draw depends on the seed ONLY.

    If it depended on anything about the body condition, a cell-to-cell
    difference in the noise-on sweep could be a difference in the draw rather
    than a difference in the body, and every paired comparison would be void.
    """
    from sim.noise import command_noise_rad_per_s

    a = command_noise_rad_per_s(150, 0.1, 3)
    b = command_noise_rad_per_s(150, 0.1, 3)
    c = command_noise_rad_per_s(150, 0.1, 4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(a) == 150
    # same recipe as the source path it replaced
    expect = deg_per_step_to_rad_per_s(
        source_command_noise_deg(150, np.random.default_rng(3),
                                 SourceNoiseSpec()), 0.1)
    assert np.array_equal(a, expect)


def test_noise_removes_the_ideal_body_failure_at_90_deg(core, decoder):
    """Finding F1 is noise-free-only, and the sweep's baseline must see that.

    Noise-free, the ideal body limit-cycles at e0 = +-90 deg; that failure is
    what BASELINE_FAILURE / RESCUED_BY_BODY are built on.  With the source
    noise on it disappears, so a noise-on sweep must re-run the baseline per
    seed instead of reusing the noise-free verdict.
    """
    from eval import metrics as mx
    from sim.noise import command_noise_rad_per_s

    crit = mx.SuccessCriterion()
    task = HeadingTask(initial_heading=np.deg2rad(90.0), duration_s=15.0)

    quiet = run_closed_loop(core, decoder, PassthroughPlugin(), IdealYawBody(),
                            IdealHeadingSensor(), task)
    assert not mx.compute(quiet, crit).success

    noisy = [run_closed_loop(core, decoder, PassthroughPlugin(), IdealYawBody(),
                             IdealHeadingSensor(), task,
                             command_noise=command_noise_rad_per_s(
                                 task.n_cycles, task.T_core_s, s))
             for s in range(4)]
    assert all(mx.compute(r, crit).success for r in noisy)


def test_flywire_eye_lattice_and_geometry():
    """The measured eye lattice, its angular scale and the two eyes' geometry.

    Runs the module's own verification: the (p, q) axes really are 120 deg
    apart (a hexagonal lattice, not Zhao's 90 deg display approximation), the
    interommatidial scale reproduces the ~150 deg monocular field, and the two
    eyes point outward with a narrow frontal binocular region.
    """
    import sensors.flywire_eye as eye

    if not Path("data/flywire/column_assignment.csv.gz").exists():
        pytest.skip("FlyWire column assignment not downloaded")
    eye.verify()


def test_flywire_brain_builds_and_propagates():
    """Sensory drive reaches the descending neurons through the connectome."""
    import core.flywire_brain as fb

    if not Path("data/flywire/edges_783.npz").exists():
        pytest.skip("FlyWire edge table not built")
    fb.demo()
