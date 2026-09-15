"""Stage 1 tests: the constrained yaw plant and the rate-clip plugin.

Handoff document section 44, items 7 (rate clip respected), 8 (acceleration
cap respected) and 9 (actual motion used as feedback, here with a body that
genuinely lags behind the command).

Run:  python -m pytest -q
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from body.ideal_yaw import IdealYawBody
from body.yaw_plant import YawPlant, YawPlantParams, from_neural_scale
from core.calibration import calibrate, load
from core.westeinde2024 import CoreParams, WesteindeSteeringCore
from decoder.steering import SteeringDecoder
from environment.heading_task import HeadingTask
from plugins.passthrough import PassthroughPlugin
from plugins.rate_clip import RateClipPlugin
from sensors.ideal_heading import IdealHeadingSensor
from sim.closed_loop import run_closed_loop
from sim.interfaces import BodyModel, ConstraintPlugin

REPO = Path(__file__).resolve().parents[1]
NORM_JSON = REPO / "configs" / "norm_constants.json"
T_CORE = 0.1


@pytest.fixture(scope="module")
def core():
    norm = load(NORM_JSON) if NORM_JSON.exists() else calibrate(CoreParams(),
                                                                verbose=False)
    return WesteindeSteeringCore(CoreParams(), norm)


@pytest.fixture(scope="module")
def decoder():
    return SteeringDecoder()


# =========================================================================
# plant construction and guards
# =========================================================================
def test_plant_satisfies_the_body_protocol():
    assert isinstance(YawPlant(), BodyModel)


def test_plant_rejects_a_timestep_too_coarse_for_the_lag():
    """Explicit Euler on a first-order lag needs dt_body well below tau_r."""
    with pytest.raises(ValueError, match="too small for dt_body"):
        YawPlant(YawPlantParams(tau_r=0.001, dt_body=0.0025))


@pytest.mark.parametrize("bad", [
    {"dt_body": 0.0}, {"r_max": 0.0}, {"alpha_max": -1.0}, {"tau_r": -0.1},
])
def test_plant_rejects_invalid_parameters(bad):
    with pytest.raises(ValueError):
        YawPlant(YawPlantParams(**bad))


def test_unconstrained_plant_matches_the_ideal_body(core, decoder):
    """With no limits and no lag the plant must reduce to the ideal body."""
    task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(40.0),
                       duration_s=3.0)
    a = run_closed_loop(core, decoder, PassthroughPlugin(), IdealYawBody(),
                        IdealHeadingSensor(), task)
    b = run_closed_loop(core, decoder, PassthroughPlugin(), YawPlant(),
                        IdealHeadingSensor(), task)
    assert np.allclose(a.psi, b.psi, atol=1e-9)


# =========================================================================
# item 7: rate limit
# =========================================================================
def test_plant_never_exceeds_r_max():
    r_max = np.deg2rad(300.0)
    plant = YawPlant(YawPlantParams(r_max=r_max))
    plant.reset()
    for _ in range(50):
        plant.step(np.deg2rad(5000.0), T_CORE)
        assert abs(plant.r) <= r_max + 1e-12


def test_rate_limit_applies_in_both_directions():
    r_max = np.deg2rad(120.0)
    plant = YawPlant(YawPlantParams(r_max=r_max))
    plant.reset()
    plant.step(np.deg2rad(-9000.0), T_CORE)
    assert plant.r == pytest.approx(-r_max, rel=1e-9)


def test_plugin_clips_the_command_to_r_max():
    p = RateClipPlugin(r_max=np.deg2rad(200.0))
    assert p.command(np.deg2rad(900.0), 0.0) == pytest.approx(np.deg2rad(200.0))
    assert p.command(np.deg2rad(-900.0), 0.0) == pytest.approx(np.deg2rad(-200.0))
    assert p.command(np.deg2rad(50.0), 0.0) == pytest.approx(np.deg2rad(50.0))


def test_plugin_reports_how_often_it_intervened():
    p = RateClipPlugin(r_max=1.0)
    p.reset()
    for v in (0.5, 2.0, -3.0, 0.1):
        p.command(v, 0.0)
    assert p.intervention_fraction == pytest.approx(0.5)


def test_plugin_is_goal_blind_and_satisfies_the_protocol():
    import inspect
    assert isinstance(RateClipPlugin(), ConstraintPlugin)
    params = list(inspect.signature(RateClipPlugin.command).parameters)
    assert params == ["self", "r_brain", "r_actual"]


# =========================================================================
# item 8: acceleration cap
# =========================================================================
def test_plant_never_exceeds_alpha_max():
    alpha = np.deg2rad(2000.0)          # rad/s^2
    dt = 0.0025
    plant = YawPlant(YawPlantParams(alpha_max=alpha, dt_body=dt))
    plant.reset()
    prev = plant.r
    for _ in range(40):
        plant.step(np.deg2rad(9000.0), dt)   # one sub-step per call
        assert abs(plant.r - prev) <= alpha * dt + 1e-12
        prev = plant.r


def test_acceleration_cap_sets_the_time_to_reach_a_rate():
    """With only an acceleration cap the rate ramps linearly."""
    alpha = np.deg2rad(1000.0)
    plant = YawPlant(YawPlantParams(alpha_max=alpha))
    plant.reset()
    plant.step(np.deg2rad(10000.0), 0.1)     # 0.1 s of full acceleration
    assert plant.r == pytest.approx(alpha * 0.1, rel=1e-6)


def test_saturation_fractions_are_tracked():
    # alpha_max must be large enough to actually REACH r_max within the step,
    # or the rate limit never binds: 5000 deg/s^2 hits 100 deg/s in 0.02 s.
    plant = YawPlant(YawPlantParams(r_max=np.deg2rad(100.0),
                                    alpha_max=np.deg2rad(5000.0)))
    plant.reset()
    plant.step(np.deg2rad(5000.0), T_CORE)
    assert plant.accel_saturation_fraction > 0.0
    assert plant.rate_saturation_fraction > 0.0
    plant.reset()
    assert plant.rate_saturation_fraction == 0.0


def test_acceleration_cap_can_make_the_rate_limit_unreachable():
    """With a weak enough cap, r_max never binds within one cycle.

    Worth pinning: the two limits interact, so a body can be 'rate limited' on
    paper and never once touch that limit in practice.
    """
    plant = YawPlant(YawPlantParams(r_max=np.deg2rad(100.0),
                                    alpha_max=np.deg2rad(500.0)))
    plant.reset()
    plant.step(np.deg2rad(5000.0), T_CORE)      # reaches only 50 deg/s
    assert plant.rate_saturation_fraction == 0.0
    assert plant.r == pytest.approx(np.deg2rad(50.0), rel=1e-6)


# =========================================================================
# response lag
# =========================================================================
def test_first_order_lag_approaches_the_command_exponentially():
    tau = 0.05
    plant = YawPlant(YawPlantParams(tau_r=tau, dt_body=0.0005))
    plant.reset()
    target = np.deg2rad(100.0)
    plant.step(target, tau)              # one time constant
    assert plant.r == pytest.approx(target * (1 - math.exp(-1.0)), rel=0.02)


def test_larger_tau_is_always_slower():
    target = np.deg2rad(200.0)
    reached = []
    for tau in (0.01, 0.05, 0.2, 0.4):
        plant = YawPlant(YawPlantParams(tau_r=tau, dt_body=0.0005))
        plant.reset()
        plant.step(target, 0.1)
        reached.append(plant.r)
    assert all(a > b for a, b in zip(reached, reached[1:]))


def test_lag_and_acceleration_cap_are_different_mechanisms():
    """Handoff doc section 11: they must not be presented as the same knob.

    A small command is limited by the lag but never touches the cap; a large
    command hits the cap.  If they were the same thing this would not hold.
    """
    small, large = np.deg2rad(20.0), np.deg2rad(4000.0)
    params = YawPlantParams(tau_r=0.1, alpha_max=np.deg2rad(2000.0),
                            dt_body=0.0005)
    p_small = YawPlant(params)
    p_small.reset()
    p_small.step(small, 0.1)
    assert p_small.accel_saturation_fraction == 0.0

    p_large = YawPlant(params)
    p_large.reset()
    p_large.step(large, 0.1)
    assert p_large.accel_saturation_fraction > 0.5


# =========================================================================
# scaling helper
# =========================================================================
def test_from_neural_scale_builds_dimensionless_parameters():
    r99 = np.deg2rad(2000.0)
    p = from_neural_scale(r99, r_max_ratio=0.5, alpha_max_ratio=2.0,
                          tau_ratio=1.0, T_core=0.1)
    assert p.r_max == pytest.approx(0.5 * r99)
    assert p.alpha_max == pytest.approx(2.0 * r99 / 0.1)
    assert p.tau_r == pytest.approx(0.1)


def test_from_neural_scale_defaults_to_unconstrained():
    p = from_neural_scale(1.0)
    assert math.isinf(p.r_max) and math.isinf(p.alpha_max) and p.tau_r == 0.0


# =========================================================================
# item 9: a lagging body really does feed back its own motion
# =========================================================================
def test_constrained_body_lags_the_brain_and_the_lag_is_observable(core, decoder):
    """The core must see where the body IS, not what it asked for."""
    scale = np.deg2rad(2000.0)
    params = from_neural_scale(scale, r_max_ratio=0.25, alpha_max_ratio=0.5,
                               tau_ratio=2.0, T_core=T_CORE)
    task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(120.0),
                       duration_s=5.0)
    res = run_closed_loop(core, decoder,
                          RateClipPlugin(r_max=params.r_max),
                          YawPlant(params), IdealHeadingSensor(), task)

    # the body never does what the brain asked
    assert np.max(res.brain_body_mismatch) > 0.0
    # the observed heading is the body's, cycle by cycle
    assert np.allclose(res.heading_obs, res.psi[:-1], atol=1e-12)
    # and the rate limit held throughout
    assert np.all(np.abs(res.r) <= params.r_max + 1e-12)
    assert np.all(np.abs(res.r_cmd) <= params.r_max + 1e-12)


def test_constrained_body_is_slower_than_the_ideal_one(core, decoder):
    """A rate-limited body must take longer to cover the same angle."""
    task = HeadingTask(goal=0.0, initial_heading=np.deg2rad(30.0),
                       duration_s=5.0)
    ideal = run_closed_loop(core, decoder, PassthroughPlugin(), IdealYawBody(),
                            IdealHeadingSensor(), task)
    params = from_neural_scale(np.deg2rad(2000.0), r_max_ratio=0.05,
                               tau_ratio=1.0, T_core=T_CORE)
    slow = run_closed_loop(core, decoder, RateClipPlugin(r_max=params.r_max),
                           YawPlant(params), IdealHeadingSensor(), task)

    def first_below(res, tol_deg=5.0):
        idx = np.where(np.abs(np.rad2deg(res.error)) < tol_deg)[0]
        return int(idx[0]) if idx.size else len(res.psi)

    assert first_below(slow) > first_below(ideal)
