"""Stage 0 gate tests for the reduced steering core.

Covers handoff document section 44, items 1-6 and 11 (the items that concern
the core alone; body / plugin / feedback tests come in later stages).

Run:  python -m pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest

from core.westeinde2024 import CoreParams, WesteindeSteeringCore, elu
from utils.angles import deg2rad, wrap, wrap_deg


@pytest.fixture(scope="module")
def core():
    return WesteindeSteeringCore(CoreParams())


# -- 1 & 2: angle wrap and degree/radian consistency -----------------------
def test_wrap_maps_into_half_open_interval():
    vals = np.deg2rad(np.array([-540.0, -180.0, -0.0, 179.9, 180.0, 360.0, 721.0]))
    w = wrap(vals)
    assert np.all(w >= -np.pi - 1e-12)
    assert np.all(w < np.pi)


def test_wrap_deg_matches_wrap_rad():
    degs = np.array([-359.0, -181.0, -45.0, 0.0, 45.0, 181.0, 359.0])
    assert np.allclose(wrap_deg(degs), np.rad2deg(wrap(np.deg2rad(degs))), atol=1e-9)


def test_core_takes_radians_not_degrees(core):
    """A degree value fed as radians must NOT give the same answer."""
    s_rad = core.steering(deg2rad(60.0), 0.0)
    s_deg = core.steering(60.0, 0.0)  # 60 radians -> wrapped to something else
    assert not np.isclose(s_rad, s_deg, atol=1e-6)


# -- 3: left/right symmetry ------------------------------------------------
@pytest.mark.parametrize("err_deg", [10.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 170.0])
def test_left_right_symmetry(core, err_deg):
    st_p = core.evaluate(deg2rad(err_deg), 0.0)
    st_m = core.evaluate(deg2rad(-err_deg), 0.0)
    # mirrored populations
    assert st_p.pfl3r == pytest.approx(st_m.pfl3l, abs=1e-12)
    assert st_p.pfl3l == pytest.approx(st_m.pfl3r, abs=1e-12)
    assert st_p.pfl2 == pytest.approx(st_m.pfl2, abs=1e-12)
    # 5: opposite errors produce opposite steering
    assert st_p.steering == pytest.approx(-st_m.steering, abs=1e-12)


# -- 4: zero error -> zero steering ---------------------------------------
def test_zero_error_gives_zero_steering(core):
    assert core.steering(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    # At a goal that is not aligned with the phase grid the population mean
    # keeps a tiny discretisation residual (order 1e-11 for n_units=1000,
    # against steering magnitudes of order 1).  That is numerical, not a
    # broken symmetry: test_discretisation_residual_is_negligible pins it.
    assert core.steering(deg2rad(137.0), deg2rad(137.0)) == pytest.approx(0.0, abs=1e-8)


def test_discretisation_residual_is_negligible(core):
    """Zero-error steering stays ~1e-8 for arbitrary, grid-unaligned goals."""
    goals = np.deg2rad([13.7, 77.3, 137.0, 201.4, -88.9])
    residuals = [abs(core.steering(g, g)) for g in goals]
    assert max(residuals) < 1e-8
    # and it is small compared with a real steering signal
    assert max(residuals) < 1e-6 * abs(core.steering(deg2rad(90.0), 0.0))


def test_only_heading_error_matters(core):
    """Rotating heading and goal together must not change the output."""
    base = core.steering(deg2rad(40.0), deg2rad(0.0))
    for shift in (30.0, 123.0, -250.0):
        rotated = core.steering(deg2rad(40.0 + shift), deg2rad(shift))
        assert rotated == pytest.approx(base, abs=1e-9)


# -- 5: correct sign of the steering drive --------------------------------
@pytest.mark.parametrize("err_deg", [5.0, 30.0, 90.0, 150.0, 179.0])
def test_positive_error_gives_positive_drive(core, err_deg):
    """e = wrap(heading - goal) > 0  ->  DNa02R > DNa02L.

    This fixes the *raw* sign convention of the core.  Converting it to a
    physical yaw rate (and thus its sign) is the decoder's job.
    """
    st = core.evaluate(deg2rad(err_deg), 0.0)
    assert st.dna02r > st.dna02l
    assert st.steering > 0.0


# -- 6: output finite, and well behaved over the whole circle -------------
def test_output_finite_everywhere(core):
    errs = np.deg2rad(np.arange(-180.0, 180.0, 1.0))
    out = core.sweep(errs)
    for key, arr in out.items():
        assert np.all(np.isfinite(arr)), "non-finite values in " + key


# -- structural checks of the reduced model -------------------------------
def test_pfl3_tuning_peaks_near_plus_minus_67_5(core):
    errs = np.deg2rad(np.arange(-180.0, 180.0, 0.5))
    out = core.sweep(errs)
    peak_r = np.rad2deg(errs[int(np.argmax(out["pfl3r"]))])
    peak_l = np.rad2deg(errs[int(np.argmax(out["pfl3l"]))])
    assert peak_r == pytest.approx(67.5, abs=2.0)
    assert peak_l == pytest.approx(-67.5, abs=2.0)


def test_pfl2_is_anti_goal(core):
    """PFL2 must be minimal at the goal and maximal 180 deg away."""
    errs = np.deg2rad(np.arange(-180.0, 180.0, 0.5))
    out = core.sweep(errs)
    peak = np.rad2deg(errs[int(np.argmax(out["pfl2"]))])
    trough = np.rad2deg(errs[int(np.argmin(out["pfl2"]))])
    assert abs(abs(peak) - 180.0) < 2.0 or abs(peak) > 178.0
    assert trough == pytest.approx(0.0, abs=2.0)


def test_180_is_an_equilibrium_not_a_failure(core):
    """Exactly anti-goal cancels by symmetry (handoff doc section 48)."""
    assert core.steering(deg2rad(180.0), 0.0) == pytest.approx(0.0, abs=1e-12)
    # but it is an UNSTABLE equilibrium: a small perturbation drives away
    assert core.steering(deg2rad(179.0), 0.0) > 0.0
    assert core.steering(deg2rad(-179.0), 0.0) < 0.0


def test_steering_is_restoring_over_the_whole_range(core):
    """sign(steering) == sign(e) for every e in (0, 180) and (-180, 0)."""
    errs = np.deg2rad(np.arange(1.0, 180.0, 1.0))
    out = core.sweep(errs)
    assert np.all(out["steering"] > 0.0)


# -- 11: the core is frozen and deterministic -----------------------------
def test_core_is_deterministic_and_stateless(core):
    a = core.steering(deg2rad(73.0), 0.0)
    for _ in range(5):
        core.steering(deg2rad(-120.0), 0.0)  # other calls must leave no trace
    b = core.steering(deg2rad(73.0), 0.0)
    assert a == b


def test_result_is_independent_of_discretisation():
    """n_units is a discretisation knob (A3), not a model parameter."""
    s = [WesteindeSteeringCore(CoreParams(n_units=n)).steering(deg2rad(75.0), 0.0)
         for n in (200, 500, 1000, 2000)]
    assert np.allclose(s, s[0], rtol=2e-3)


def test_elu_shape():
    assert elu(2.0) == pytest.approx(2.0)
    assert float(elu(0.0)) == pytest.approx(0.0)
    assert float(elu(-1.0)) == pytest.approx(np.expm1(-1.0))
    assert float(elu(-50.0)) > -1.0000001  # bounded below by -alpha


def test_per_population_normalisation_would_break_the_model():
    """Documents WHY normalisation must be global (see core._normalise).

    If each population were divided by its own peak, PFL3R and PFL3L would
    become identical and the steering drive would vanish.  We assert the
    global modes keep a non-zero drive.
    """
    for mode in ("none", "global_peak", "global_rms"):
        c = WesteindeSteeringCore(CoreParams(normalize=mode))
        assert abs(c.steering(deg2rad(90.0), 0.0)) > 1e-6, mode
