"""Stage 0 structural tests for the reduced steering core.

Covers handoff document section 44, items 1-6 and 11 (the items about the core
alone).  Numerical agreement with the official notebook lives in
test_source_agreement.py.

Run:  python -m pytest -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.calibration import calibrate, load
from core.westeinde2024 import (CoreParams, NormConstants, WesteindeSteeringCore,
                                elu1, linear_rescale)
from utils.angles import deg2rad, wrap, wrap_deg

REPO = Path(__file__).resolve().parents[1]
NORM_JSON = REPO / "configs" / "norm_constants.json"


@pytest.fixture(scope="module")
def norm() -> NormConstants:
    """Frozen constants: the committed file if present, else recompute."""
    if NORM_JSON.exists():
        return load(NORM_JSON)
    return calibrate(CoreParams(), verbose=False)


@pytest.fixture(scope="module")
def core(norm):
    return WesteindeSteeringCore(CoreParams(), norm)


# -- 1 & 2: angle wrap and degree/radian consistency -----------------------
def test_wrap_maps_into_half_open_interval():
    vals = np.deg2rad(np.array([-540.0, -180.0, -0.0, 179.9, 180.0, 360.0, 721.0]))
    w = wrap(vals)
    assert np.all(w >= -np.pi - 1e-12)
    assert np.all(w < np.pi)


def test_wrap_deg_matches_wrap_rad():
    degs = np.array([-359.0, -181.0, -45.0, 0.0, 45.0, 181.0, 359.0])
    assert np.allclose(wrap_deg(degs), np.rad2deg(wrap(np.deg2rad(degs))), atol=1e-9)


def test_radian_api_matches_degree_api(core):
    for hd in (-150.0, -33.0, 0.0, 47.5, 120.0):
        a = core.evaluate_deg(hd, 0.0).steering
        b = core.steering(deg2rad(hd), 0.0)
        assert a == pytest.approx(b, abs=1e-12)


def test_core_takes_radians_not_degrees(core):
    """A degree value fed to the radian API must NOT give the same answer."""
    assert not np.isclose(core.steering(deg2rad(60.0), 0.0),
                          core.steering(60.0, 0.0), atol=1e-6)


# -- 3 & 5: left/right symmetry and opposite signs -------------------------
@pytest.mark.parametrize("err_deg", [10.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 170.0])
def test_left_right_symmetry(core, err_deg):
    p = core.evaluate_deg(err_deg, 0.0)
    m = core.evaluate_deg(-err_deg, 0.0)
    assert p.sum_pfl3r == pytest.approx(m.sum_pfl3l, rel=1e-12)
    assert p.sum_pfl3l == pytest.approx(m.sum_pfl3r, rel=1e-12)
    assert p.sum_pfl2 == pytest.approx(m.sum_pfl2, rel=1e-12)
    assert p.dna02r == pytest.approx(m.dna02l, rel=1e-12)
    assert p.steering == pytest.approx(-m.steering, abs=1e-12)


# -- 4: zero error -> zero steering ---------------------------------------
def test_zero_error_gives_zero_steering(core):
    assert core.evaluate_deg(0.0, 0.0).steering == pytest.approx(0.0, abs=1e-12)


def test_only_heading_error_matters(core):
    """Rotating heading and goal together must not change the output."""
    base = core.evaluate_deg(40.0, 0.0).steering
    for shift in (30.0, 123.0, -250.0):
        rotated = core.evaluate_deg(40.0 + shift, shift).steering
        assert rotated == pytest.approx(base, abs=1e-9)


# -- sign convention: the loop must be NEGATIVE feedback ------------------
@pytest.mark.parametrize("err_deg", [5.0, 30.0, 90.0, 150.0, 179.0])
def test_positive_error_gives_negative_steering(core, err_deg):
    """e = wrap(heading - goal) > 0 must command a turn back toward the goal.

    The source closed loop is hd[t] = hd[t-1] + k*steering with k > 0, so a
    positive error has to produce a negative command.  This pins the R/L phase
    convention: an earlier version of this core had it mirrored, which would
    have made the loop diverge.
    """
    st = core.evaluate_deg(err_deg, 0.0)
    assert st.steering < 0.0
    assert st.dna02r < st.dna02l


def test_steering_is_restoring_over_the_whole_range(core):
    hds = np.arange(1.0, 180.0, 1.0)
    out = core.sweep_deg(hds, 0.0)
    assert np.all(out["steering"] < 0.0)


# -- 6: finite everywhere -------------------------------------------------
def test_output_finite_everywhere(core):
    out = core.sweep_deg(np.arange(-180.0, 180.0, 1.0), 0.0)
    for key, arr in out.items():
        assert np.all(np.isfinite(arr)), "non-finite values in " + key


# -- structure of the reduced model ---------------------------------------
def test_pfl2_is_anti_goal(core):
    """PFL2 bump amplitude must grow with |heading error|, peaking anti-goal."""
    hds = np.arange(-180.0, 180.5, 0.5)
    out = core.sweep_deg(hds, 0.0)
    amp = out["pfl2_bump_amp"]
    assert amp[np.argmin(np.abs(hds))] == pytest.approx(0.0, abs=1e-9)
    assert abs(abs(hds[int(np.argmax(amp))]) - 180.0) < 1.0
    # monotone in |e| on the positive side
    pos = hds >= 0
    assert np.all(np.diff(amp[pos]) > -1e-12)


def test_180_is_an_equilibrium_not_a_failure(core):
    """Exactly anti-goal cancels by symmetry (handoff doc section 48)."""
    assert core.evaluate_deg(180.0, 0.0).steering == pytest.approx(0.0, abs=1e-12)
    # unstable: a small perturbation drives away from it
    assert core.evaluate_deg(179.0, 0.0).steering < 0.0
    assert core.evaluate_deg(-179.0, 0.0).steering > 0.0


# -- PFL2 now has a functional role (was open issue O1) -------------------
def test_pfl2_dominates_the_steering_magnitude(core, norm):
    """Silencing PFL2 must collapse the steering command.

    Before the source code was consulted this core applied a plain ELU with no
    rescaling; every DN input stayed positive, the ELU never left its identity
    regime, and the bilateral PFL2 drive cancelled EXACTLY in DNa02R - DNa02L.
    The source's rescale-to-[-1,1] is what puts the DN stage on the nonlinear
    branch.  This test would fail if that rescale were ever dropped.

    The lesion reuses the SAME frozen constants: silencing a population is not
    a reason to recalibrate the core.
    """
    lesioned = WesteindeSteeringCore(CoreParams(pfl2_silenced=True), norm)
    for err_deg in (30.0, 60.0, 90.0, 120.0, 150.0):
        intact = abs(core.evaluate_deg(err_deg, 0.0).steering)
        without = abs(lesioned.evaluate_deg(err_deg, 0.0).steering)
        assert intact > 10.0 * without, (
            "PFL2 lesion barely changed steering at %.0f deg "
            "(%.6f vs %.6f) -- the DN nonlinearity is not engaging"
            % (err_deg, intact, without))


def test_dn_stage_is_not_a_constant_gain(core):
    """The DN cascade must not collapse to steering = const * dPFL3."""
    ratios = []
    for err_deg in (30.0, 90.0, 150.0):
        st = core.evaluate_deg(err_deg, 0.0)
        ratios.append(st.steering / (st.sum_pfl3r - st.sum_pfl3l))
    assert max(ratios) / min(ratios) > 1.5, (
        "steering / dPFL3 is nearly constant (%r): the DN stage is linear"
        % ratios)


# -- frozen constants -----------------------------------------------------
def test_core_refuses_to_run_without_frozen_constants():
    bare = WesteindeSteeringCore(CoreParams())
    with pytest.raises(RuntimeError, match="frozen normalisation"):
        bare.evaluate_deg(45.0, 0.0)


def test_committed_constants_match_a_fresh_calibration():
    """configs/norm_constants.json must still describe the official grid."""
    if not NORM_JSON.exists():
        pytest.skip("configs/norm_constants.json not generated yet")
        return
    committed = load(NORM_JSON)
    fresh = calibrate(CoreParams(), verbose=False)
    assert committed.steering_max == pytest.approx(fresh.steering_max, rel=1e-12)
    for stage in ("pfl2", "pfl3r", "pfl3l", "dna03r", "dna03l",
                  "dna02r", "dna02l"):
        a = getattr(committed, stage).as_tuple()
        b = getattr(fresh, stage).as_tuple()
        assert a == pytest.approx(b, rel=1e-12), stage


def test_left_right_constants_are_symmetric(norm):
    """Mirror-image stages must share limits, or symmetry breaks silently."""
    assert norm.pfl3r.as_tuple() == pytest.approx(norm.pfl3l.as_tuple(), rel=1e-12)
    assert norm.dna03r.as_tuple() == pytest.approx(norm.dna03l.as_tuple(), rel=1e-12)
    assert norm.dna02r.as_tuple() == pytest.approx(norm.dna02l.as_tuple(), rel=1e-12)


# -- 11: determinism ------------------------------------------------------
def test_core_is_deterministic_and_stateless(core):
    a = core.evaluate_deg(73.0, 0.0).steering
    for _ in range(5):
        core.evaluate_deg(-120.0, 0.0)
    assert core.evaluate_deg(73.0, 0.0).steering == a


# -- primitives -----------------------------------------------------------
def test_linear_rescale_clamps_like_np_interp():
    """Clamping outside the anchors is source behaviour and must be kept."""
    assert linear_rescale(-5.0, -1.0, 1.0, 0.0, 1.0) == pytest.approx(0.0)
    assert linear_rescale(5.0, -1.0, 1.0, 0.0, 1.0) == pytest.approx(1.0)
    assert linear_rescale(0.0, -1.0, 1.0, 0.0, 1.0) == pytest.approx(0.5)


def test_linear_rescale_rejects_degenerate_range():
    with pytest.raises(ValueError):
        linear_rescale(0.0, 1.0, 1.0, 0.0, 1.0)


def test_elu1_matches_the_source_formula():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    lo, hi = -2.0, 2.0
    scaled = np.interp(x, (lo, hi), (-1.0, 1.0))
    expected_raw = np.where(scaled >= 0, scaled, np.exp(scaled) - 1.0)
    post_lo, post_hi = expected_raw.min(), expected_raw.max()
    expected = np.interp(expected_raw, (post_lo, post_hi), (0.0, 1.0))
    assert np.allclose(elu1(x, lo, hi, post_lo, post_hi), expected, atol=1e-15)


def test_data_connectivity_does_not_steer_to_the_commanded_goal():
    """Finding D1: the hemibrain-wired core has a goal-dependent steering bias.

    The abstract core's stable heading equals the commanded goal exactly, at
    every goal.  Wired with hemibrain synapse counts instead, the same circuit
    settles up to ~36 deg away, and near goal = +-90 deg it has no stable
    heading at all.  Verified against the authors' own arrays, not just ours
    (reference/check_connectivity_data.py), so this is a property of the
    source's connectivity_option = 'data', not of this implementation.

    Pinned because the whole comparison of idealised against measured
    connectivity rests on it.
    """
    from core.westeinde2024_data import WesteindeDataCore
    from core.westeinde2024_data import calibrate as calibrate_data

    core = WesteindeDataCore(norm=calibrate_data(verbose=False))
    hd = np.arange(-180.0, 180.0, 0.5)

    def stable_zero(goal_deg):
        s = np.array([core.steering_deg(float(h), goal_deg) for h in hd])
        cross = [hd[i] - s[i] * (hd[i + 1] - hd[i]) / (s[i + 1] - s[i])
                 for i in range(len(hd) - 1) if s[i] > 0 >= s[i + 1]]
        if not cross:
            return None
        return min(cross, key=lambda z: abs((z - goal_deg + 180) % 360 - 180))

    def bias(goal_deg):
        z = stable_zero(goal_deg)
        return None if z is None else (z - goal_deg + 180) % 360 - 180

    # goal = 0 is exact: the assumed left-right mirror symmetry makes it so
    assert abs(bias(0.0)) < 0.5

    # away from that symmetry axis the bias is real and odd-symmetric
    b30, b60 = bias(30.0), bias(60.0)
    assert 5.0 < b30 < 10.0, b30
    assert 28.0 < b60 < 35.0, b60
    assert abs(bias(-30.0) + b30) < 0.5
    assert abs(bias(-60.0) + b60) < 0.5

    # and near +-90 deg there is no stable heading near the goal at all
    assert stable_zero(90.0) is None
    assert stable_zero(-90.0) is None
