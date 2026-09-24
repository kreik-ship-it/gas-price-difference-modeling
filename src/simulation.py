"""
Simulation des kalibrierten OU-Prozesses, Monte-Carlo-Optionsbewertung
und Transportkapazitaets-Bewertung als bidirektionaler Optionsstrip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ou_model import OUParams


def simulate_ou_paths(
    params: OUParams, X0: float, n_steps: int, n_simulations: int, dt_years: float, seed: int | None = None
) -> np.ndarray:
    """Euler-Maruyama-Simulation des OU-Prozesses mit konstantem dt (fuer
    die Simulation ausreichend, da hier mit einem einheitlichen Zeitraster
    gearbeitet wird -- die irregulaere-dt-Behandlung betrifft nur die
    PARAMETERSCHAETZUNG in ou_model.py, nicht die Simulation selbst).

    Returns
    -------
    np.ndarray der Form (n_steps + 1, n_simulations); Zeile 0 = X0.
    """
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_steps + 1, n_simulations))
    paths[0, :] = X0
    sqrt_dt = np.sqrt(dt_years)

    for t in range(1, n_steps + 1):
        z = rng.normal(size=n_simulations)
        paths[t, :] = (
            paths[t - 1, :]
            + params.alpha * (params.mu - paths[t - 1, :]) * dt_years
            + params.sigma * sqrt_dt * z
        )

    return paths


@dataclass
class OptionPriceResult:
    price: float
    standard_error: float
    confidence_interval_95: tuple[float, float]
    n_simulations: int


def price_call_option(
    params: OUParams,
    X0: float,
    K: float,
    r: float,
    T_years: float = 1.0,
    n_steps: int = 252,
    n_simulations: int = 10_000,
    seed: int | None = 42,
) -> OptionPriceResult:
    """Bachelier-Style-Call auf den Preisunterschied X_T:
    Payoff = max(X_T - K, 0), Monte-Carlo-Bewertung ueber einen
    simulierten OU-Pfad mit kalibrierten OUParams."""
    dt = T_years / n_steps
    paths = simulate_ou_paths(params, X0, n_steps, n_simulations, dt_years=dt, seed=seed)
    X_T = paths[-1, :]
    payoffs = np.maximum(X_T - K, 0)

    discount = np.exp(-r * T_years)
    price = discount * float(np.mean(payoffs))

    payoff_std = float(np.std(payoffs, ddof=1))
    se = discount * payoff_std / np.sqrt(n_simulations)
    ci = (price - 1.96 * se, price + 1.96 * se)

    return OptionPriceResult(price=price, standard_error=se, confidence_interval_95=ci, n_simulations=n_simulations)


def validate_against_analytical_moments(params: OUParams, X0: float, T_years: float, simulated_X_T: np.ndarray) -> dict:
    """Vergleicht simulierte End-Momente mit den analytischen OU-Momenten
    -- eine Simulation, die die theoretischen Momente nicht reproduziert,
    ist ein Implementierungsfehler, kein Modellierungsdetail."""
    expected_XT = params.mu + (X0 - params.mu) * np.exp(-params.alpha * T_years)
    variance_XT = (params.sigma**2) / (2 * params.alpha) * (1 - np.exp(-2 * params.alpha * T_years))
    std_XT = np.sqrt(variance_XT)

    return {
        "theoretical_mean": float(expected_XT),
        "simulated_mean": float(np.mean(simulated_X_T)),
        "theoretical_std": float(std_XT),
        "simulated_std": float(np.std(simulated_X_T)),
    }


# ----------------------------------------------------------------
# Transportkapazitaets-Bewertung (bidirektional, als Options-Strip)
# ----------------------------------------------------------------


def empirical_capacity_value(price_differences: pd.Series, K: float) -> dict:
    """Empirischer Referenzwert auf Grundlage der tatsaechlich beobachteten
    Preisunterschiede: kein finanzmathematischer Optionswert, sondern das
    Ertragspotenzial, das sich bei direkter Nutzung der historisch
    beobachteten Preisunterschiede ergeben haette.

    Fuer jede Richtung wird der durchschnittliche taegliche Payoff
    max(S_t - K, 0) ueber die Beobachtungsperiode gebildet und auf 365
    Liefertage hochgerechnet -- OHNE Diskontierung (bewusste
    Vereinfachung: der Wert dient als historische Referenzgroesse, nicht
    als realisierter Gewinn oder arbitragefreier Marktpreis).
    """
    x = price_differences.to_numpy()
    payoff_ab = np.maximum(x - K, 0)
    payoff_ba = np.maximum(-x - K, 0)

    value_ab = 365.0 * float(np.mean(payoff_ab))
    value_ba = 365.0 * float(np.mean(payoff_ba))

    return {
        "annual_ab": value_ab,
        "annual_ba": value_ba,
        "annual_total": value_ab + value_ba,
        "n_days": len(x),
    }


def ou_simulated_capacity_value(
    params: OUParams, X0: float, K: float, r: float, n_simulations: int = 1000, T_days: int = 365, seed: int | None = 42
) -> dict:
    """Modellbasierter Kapazitaetswert per Monte-Carlo-Simulation: M
    zukuenftige Preispfade werden ueber T Liefertage aus dem kalibrierten
    OU-Prozess simuliert, der taegliche Payoff beider Richtungen wird
    einzeln mit exp(-r * t/365) diskontiert, der Kapazitaetswert ergibt
    sich als Stichprobenmittel ueber alle simulierten Pfade (vgl.
    Monte-Carlo-Schaetzer in der zugrundeliegenden Bewertungsmethodik).
    Anders als der empirische Referenzwert WIRD hier diskontiert, da es
    sich um eine echte Bewertung zukuenftiger, unsicherer Zahlungen
    handelt -- die beiden Werte sind daher Referenzgroesse und Modellwert,
    keine methodisch identischen Groessen."""
    dt = 1.0 / 365
    paths = simulate_ou_paths(params, X0, n_steps=T_days, n_simulations=n_simulations, dt_years=dt, seed=seed)
    # paths[0] ist X0; Tage 1..T_days sind paths[1:]
    daily_paths = paths[1:, :]  # shape (T_days, n_simulations)

    days = np.arange(1, T_days + 1)
    discount_factors = np.exp(-r * days / 365)

    payoff_ab = np.maximum(daily_paths - K, 0)
    payoff_ba = np.maximum(-daily_paths - K, 0)

    discounted_ab = payoff_ab * discount_factors[:, None]
    discounted_ba = payoff_ba * discount_factors[:, None]

    annual_ab = float(discounted_ab.sum(axis=0).mean())
    annual_ba = float(discounted_ba.sum(axis=0).mean())

    return {
        "annual_ab": annual_ab,
        "annual_ba": annual_ba,
        "annual_total": annual_ab + annual_ba,
        "n_simulations": n_simulations,
    }
