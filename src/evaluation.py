"""
Zusammenfassende Ergebnistabellen ueber alle fuenf Preisunterschiede
hinweg: der finale Modellvergleich (Kalibrierung Regime 2, Bewertung auf
dem Testfenster), der Robustheitscheck (Regime 1 vs. Regime 2 als
Kalibrierungsbasis, beide gegen dasselbe Testfenster) sowie die
Bewertung der Transportkapazitaet (empirischer Referenzwert vs.
OU-modellbasierter Wert je Untersuchungsperiode).
"""

from __future__ import annotations

import pandas as pd

from backtesting import run_backtest
from data import PRICE_DIFFERENCE_DEFINITIONS, split_calibration_test
from ou_model import OUModel
from simulation import empirical_capacity_value, ou_simulated_capacity_value


def summarize_backtest(price_differences: pd.DataFrame) -> pd.DataFrame:
    """Fuehrt den Backtest fuer alle fuenf Preisunterschiede durch,
    kalibriert auf Regime 2, bewertet auf dem Testfenster. Eine Zeile pro
    Preisunterschied."""
    rows = []
    for name in PRICE_DIFFERENCE_DEFINITIONS:
        sub = price_differences[["Date", name]].dropna().reset_index(drop=True)
        split = split_calibration_test(sub)
        result = run_backtest(
            split.regime_2[name], split.regime_2["Date"], split.test[name], split.test["Date"]
        )
        rows.append(
            {
                "Preisunterschied": name,
                "alpha": result["params"].alpha,
                "mu": result["params"].mu,
                "sigma": result["params"].sigma,
                "half_life_days": result["params"].half_life_days,
                "RMSE": result["rmse"],
                "MAE": result["mae"],
                "JB_pvalue": result["jarque_bera_pvalue"],
                "LjungBox_pvalue": result["ljung_box_pvalue"],
                "Annahmen_gueltig": result["model_assumptions_valid"],
            }
        )
    return pd.DataFrame(rows).set_index("Preisunterschied")


def compare_regime_calibrations(price_differences: pd.DataFrame) -> pd.DataFrame:
    """Kalibriert jeden Preisunterschied SEPARAT auf Regime 1 und auf
    Regime 2 und bewertet beide Kalibrierungen gegen dasselbe unberuehrte
    Testfenster -- Robustheitscheck, ob die juengere (Regime 2) oder die
    aeltere (Regime 1) Kalibrierung auf dem Testfenster besser
    abschneidet."""
    rows = []
    for name in PRICE_DIFFERENCE_DEFINITIONS:
        sub = price_differences[["Date", name]].dropna().reset_index(drop=True)
        split = split_calibration_test(sub)

        r1 = run_backtest(split.regime_1[name], split.regime_1["Date"], split.test[name], split.test["Date"])
        r2 = run_backtest(split.regime_2[name], split.regime_2["Date"], split.test[name], split.test["Date"])

        rows.append(
            {
                "Preisunterschied": name,
                "RMSE_Regime1": r1["rmse"],
                "RMSE_Regime2": r2["rmse"],
                "MAE_Regime1": r1["mae"],
                "MAE_Regime2": r2["mae"],
            }
        )
    return pd.DataFrame(rows).set_index("Preisunterschied")


def capacity_valuation_table(
    price_differences: pd.DataFrame,
    K: float = 0.6,
    r: float = 0.03,
    n_simulations: int = 1000,
    T_days: int = 365,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Bewertung der Transportkapazitaet je Preisunterschied und
    Untersuchungsperiode: empirischer Referenzwert (undiskontiert, siehe
    `empirical_capacity_value`) vs. OU-modellbasierter Monte-Carlo-Wert
    (diskontiert, siehe `ou_simulated_capacity_value`).

    Annahmen (wie in der zugrundeliegenden Bewertungsmethodik): 1 MWh/Tag
    Kapazitaet, 365 Liefertage, einheitliche Transportkosten K=0,6 EUR/MWh
    in beide Richtungen, jaehrlicher Zinssatz r=3%, M=1.000 simulierte
    Pfade je Bewertung. Die Bewertung erfolgt unter dem realen
    (physischen) Mass, nicht risikoneutral -- die Werte sind Schaetzungen
    des erwarteten Ertragspotenzials, keine arbitragefreien Marktpreise.

    Untersuchungsperioden je Preisunterschied:
    - Periode 1: Regime 1 (Kalibrierung vor dem Angriffskrieg)
    - Periode 2: Regime 2 (Kalibrierung nach Kriegsbeginn)
    - Gesamtperiode: die gesamte verfuegbare Historie des Preisunterschieds

    Fuer jede Periode wird das OU-Modell separat auf genau dieser Periode
    kalibriert; die Simulation startet vom letzten beobachteten Wert der
    jeweiligen Periode.
    """
    rows = []
    for name in PRICE_DIFFERENCE_DEFINITIONS:
        sub = price_differences[["Date", name]].dropna().reset_index(drop=True)
        split = split_calibration_test(sub)

        periods = {
            "Periode 1 (Regime 1)": split.regime_1,
            "Periode 2 (Regime 2)": split.regime_2,
            "Gesamtperiode": sub,
        }

        for period_label, df_period in periods.items():
            series = df_period[name]
            dates = df_period["Date"]

            empirical = empirical_capacity_value(series, K=K)

            model = OUModel().fit(series, dates)
            X0 = float(series.iloc[-1])
            ou_value = ou_simulated_capacity_value(
                model.params, X0=X0, K=K, r=r,
                n_simulations=n_simulations, T_days=T_days, seed=seed,
            )

            rows.append(
                {
                    "Preisunterschied": name,
                    "Periode": period_label,
                    "Empirisch_EUR": round(empirical["annual_total"], 2),
                    "OU_Modell_EUR": round(ou_value["annual_total"], 2),
                }
            )

    return pd.DataFrame(rows)
