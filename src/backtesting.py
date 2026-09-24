"""
Backtesting-Orchestrierung: Kalibrierung des OU-Modells auf einem
Zeitraum, Bewertung auf einem unberuehrten Testfenster.
"""

from __future__ import annotations

import pandas as pd

from ou_model import OUModel


def run_backtest(
    train: pd.Series, train_dates: pd.Series, test: pd.Series, test_dates: pd.Series
) -> dict:
    """Fit des OU-Modells auf `train`, Bewertung auf `test`. Gibt
    Ein-Schritt-Vorhersagefehler (RMSE/MAE) UND die vollstaendige
    Residualdiagnostik (Jarque-Bera, Ljung-Box) zurueck -- eine
    gescheiterte Validierung ist damit strukturell sichtbar, nicht
    optional uebergehbar."""
    model = OUModel().fit(train, train_dates)
    result = model.evaluate(test, test_dates)
    return {"params": model.params, **result}
