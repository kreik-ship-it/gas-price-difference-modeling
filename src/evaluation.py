"""
Zusammenfassende Ergebnistabellen ueber alle fuenf Preisunterschiede
hinweg: der finale Modellvergleich (Kalibrierung Regime 2, Bewertung auf
dem Testfenster) und der Robustheitscheck (Regime 1 vs. Regime 2 als
Kalibrierungsbasis, beide gegen dasselbe Testfenster).
"""

from __future__ import annotations

import pandas as pd

from backtesting import run_backtest
from data import PRICE_DIFFERENCE_DEFINITIONS, split_calibration_test


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
