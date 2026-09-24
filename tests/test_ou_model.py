"""
Unit-Tests fuer das OU-Modell. Kein grosses Test-Framework-Setup noetig,
laeuft mit pytest:

    pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ou_model import OUModel, OUParams, _year_fraction, estimate_ou_params
from simulation import simulate_ou_paths


def _make_synthetic_ou_series(alpha=5.0, mu=0.0, sigma=1.0, n=1000, dt=1 / 365, seed=42):
    """Erzeugt eine synthetische OU-Reihe mit bekannten Parametern und
    regelmaessigem taeglichem Abstand, als Grundwahrheit fuer die
    Parameterschaetzung."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = mu
    sqrt_dt = np.sqrt(dt)
    for t in range(1, n):
        x[t] = x[t - 1] + alpha * (mu - x[t - 1]) * dt + sigma * sqrt_dt * rng.normal()
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(x), pd.Series(dates)


class TestYearFraction:
    def test_regular_daily_steps(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        dt = _year_fraction(dates)
        assert np.allclose(dt, 1 / 365)

    def test_weekend_gap(self):
        # Freitag -> Montag = 3 Kalendertage
        dates = pd.to_datetime(["2024-01-05", "2024-01-08"])  # Fr -> Mo
        dt = _year_fraction(dates)
        assert np.isclose(dt[0], 3 / 365)

    def test_non_monotonic_dates_raise(self):
        dates = pd.to_datetime(["2024-01-05", "2024-01-01"])
        with pytest.raises(ValueError):
            _year_fraction(dates)


class TestParameterEstimation:
    def test_recovers_known_parameters_approximately(self):
        true_alpha, true_mu, true_sigma = 5.0, 2.0, 1.5
        x, dates = _make_synthetic_ou_series(alpha=true_alpha, mu=true_mu, sigma=true_sigma, n=2000)
        params = estimate_ou_params(x, dates)

        assert params.alpha == pytest.approx(true_alpha, rel=0.3)
        assert params.mu == pytest.approx(true_mu, abs=0.5)
        assert params.sigma == pytest.approx(true_sigma, rel=0.3)

    def test_mismatched_lengths_raise(self):
        x = pd.Series([1.0, 2.0, 3.0])
        dates = pd.to_datetime(["2020-01-01", "2020-01-02"])  # zu kurz
        with pytest.raises(ValueError):
            estimate_ou_params(x, dates)


class TestOUParams:
    def test_half_life_positive_and_decreasing_in_alpha(self):
        slow = OUParams(alpha=1.0, mu=0.0, sigma=1.0)
        fast = OUParams(alpha=10.0, mu=0.0, sigma=1.0)
        assert slow.half_life_days > 0
        assert fast.half_life_days < slow.half_life_days


class TestOUModel:
    def test_fit_before_evaluate_required(self):
        model = OUModel()
        x = pd.Series([1.0, 2.0, 3.0])
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        with pytest.raises(RuntimeError):
            model.evaluate(x, dates)

    def test_evaluate_returns_diagnostics_even_when_assumptions_fail(self):
        x, dates = _make_synthetic_ou_series(n=300, seed=1)
        model = OUModel().fit(x, dates)
        result = model.evaluate(x, dates)
        for key in (
            "rmse",
            "mae",
            "jarque_bera_pvalue",
            "ljung_box_pvalue",
            "model_assumptions_valid",
        ):
            assert key in result


class TestSimulation:
    def test_output_shape(self):
        params = OUParams(alpha=5.0, mu=0.0, sigma=1.0)
        paths = simulate_ou_paths(params, X0=0.0, n_steps=50, n_simulations=20, dt_years=1 / 365, seed=0)
        assert paths.shape == (51, 20)

    def test_first_row_equals_initial_value(self):
        params = OUParams(alpha=5.0, mu=0.0, sigma=1.0)
        paths = simulate_ou_paths(params, X0=3.5, n_steps=10, n_simulations=5, dt_years=1 / 365, seed=0)
        assert np.allclose(paths[0, :], 3.5)

    def test_reproducible_with_seed(self):
        params = OUParams(alpha=5.0, mu=0.0, sigma=1.0)
        paths_a = simulate_ou_paths(params, X0=0.0, n_steps=100, n_simulations=10, dt_years=1 / 365, seed=7)
        paths_b = simulate_ou_paths(params, X0=0.0, n_steps=100, n_simulations=10, dt_years=1 / 365, seed=7)
        assert np.array_equal(paths_a, paths_b)

    def test_long_run_mean_reverts_toward_mu(self):
        params = OUParams(alpha=20.0, mu=5.0, sigma=0.5)
        paths = simulate_ou_paths(params, X0=0.0, n_steps=2000, n_simulations=500, dt_years=1 / 365, seed=3)
        late_mean = paths[-1, :].mean()
        assert late_mean == pytest.approx(params.mu, abs=0.5)
