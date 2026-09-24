"""
Ornstein-Uhlenbeck-Modell mit irregulaerem Zeitschritt.

Die Rohdaten enthalten (quasi) taegliche Preise Montag-Samstag, wobei der
Samstagspreis de facto das gesamte Wochenende vertritt (Sonntag wird
nicht separat notiert). Die Beobachtungsabstaende sind damit nicht
durchgaengig gleich gross -- mal ein Kalendertag, mal zwei (z. B.
Freitag->Samstag vs. Samstag->Montag). Eine einzelne feste dt-Konvention
(etwa 1/252 fuer Handelstage oder 1/365 fuer Kalendertage) wuerde diese
Unregelmaessigkeit verschleiern und die Parameter systematisch verzerren.

Dieses Modul schaetzt daher mit einem PRO SCHRITT variablen dt_i,
berechnet aus den tatsaechlichen Datumsabstaenden, ueber die exakte
Gauss'sche Transitionsdichte des OU-Prozesses (Maximum Likelihood) -- keine
Naeherung durch eine gewoehnliche AR(1)-OLS-Regression, die ein
konstantes dt voraussetzt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


@dataclass
class OUParams:
    """Annualisierte OU-Parameter (kontinuierliche Zeit).

    dX_t = alpha * (mu - X_t) dt + sigma dW_t
    """

    alpha: float  # Mean-Reversion-Geschwindigkeit (pro Jahr)
    mu: float  # langfristiger Mittelwert
    sigma: float  # Volatilitaet (annualisiert)

    @property
    def half_life_days(self) -> float:
        """Halbwertszeit in Kalendertagen: ln(2) / alpha, mit alpha in
        1/Jahr, umgerechnet auf Tage (Jahr = 365 Tage als Referenzgroesse
        fuer die Umrechnung, unabhaengig vom Schaetz-dt)."""
        return (np.log(2) / self.alpha) * 365.0


def _year_fraction(dates: pd.Series) -> np.ndarray:
    """Berechnet dt_i (in Jahresbruchteilen, Basis 365 Tage) zwischen
    aufeinanderfolgenden Beobachtungen aus den tatsaechlichen Datumswerten.
    Ein Freitag-zu-Montag-Schritt mit dazwischenliegendem Wochenendpreis
    ergibt dt = 3/365, ein normaler Wochentagsschritt dt = 1/365 usw."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    deltas_days = dates.diff().dt.days.to_numpy()[1:]
    if np.any(deltas_days <= 0):
        raise ValueError(
            "Datumsreihe muss streng aufsteigend und eindeutig sein "
            "(gefunden: dt <= 0 an mindestens einer Stelle)."
        )
    return deltas_days / 365.0


def _ljung_box_pvalue(resid: np.ndarray, lags: int = 10) -> float:
    """Ljung-Box-Test auf Autokorrelation, manuell implementiert (kein
    statsmodels-Import, um das Modul ohne diese Zusatzabhaengigkeit
    lauffaehig zu halten). Standardformel:

        Q = n(n+2) * sum_{k=1..h} rho_k^2 / (n-k)  ~ chi2(h) unter H0.
    """
    n = len(resid)
    resid = resid - np.mean(resid)
    var = np.sum(resid**2)
    acf = []
    for k in range(1, lags + 1):
        acf_k = np.sum(resid[:-k] * resid[k:]) / var
        acf.append(acf_k)
    acf = np.array(acf)
    q_stat = n * (n + 2) * np.sum((acf**2) / (n - np.arange(1, lags + 1)))
    p_value = 1 - stats.chi2.cdf(q_stat, df=lags)
    return float(p_value)


def _neg_log_likelihood(params: np.ndarray, X: np.ndarray, dt: np.ndarray) -> float:
    """Negative Log-Likelihood der OU-Transitionsdichte mit PRO SCHRITT
    variablem dt (statt eines skalaren dt). Fuer dt_i konstant reduziert
    sich das exakt auf die klassische AR(1)-Likelihood."""
    alpha, mu, sigma = params
    if alpha <= 0 or sigma <= 0:
        return np.inf

    x_t = X[:-1]
    x_next = X[1:]

    phi = np.exp(-alpha * dt)
    cond_mean = mu + (x_t - mu) * phi
    cond_var = (sigma**2) * (1 - phi**2) / (2 * alpha)
    cond_var = np.maximum(cond_var, 1e-12)

    resid = x_next - cond_mean
    ll = -0.5 * np.log(2 * np.pi * cond_var) - 0.5 * (resid**2) / cond_var
    return -np.sum(ll)


def estimate_ou_params(X: pd.Series, dates: pd.Series) -> OUParams:
    """MLE-Schaetzung mit irregulaerem dt.

    Startwerte fuer den Optimierer stammen aus einer einfachen OLS-
    Naeherung unter Annahme konstanten mittleren dt -- nur als Startpunkt,
    das eigentliche Ergebnis kommt aus der exakten MLE mit variablem dt.
    """
    x = X.to_numpy(dtype=float)
    dt = _year_fraction(dates)
    if len(dt) != len(x) - 1:
        raise ValueError("Laenge von dates und X passt nicht zusammen.")

    mean_dt = float(np.mean(dt))
    y = np.diff(x)
    x_lag = x[:-1]
    slope, intercept = np.polyfit(x_lag, y, 1)
    alpha_0 = max(-slope / mean_dt, 1e-6)
    mu_0 = intercept / (alpha_0 * mean_dt) if alpha_0 > 0 else float(np.mean(x))
    resid_0 = y - (intercept + slope * x_lag)
    sigma_0 = max(float(np.std(resid_0)) / np.sqrt(mean_dt), 1e-6)

    result = minimize(
        _neg_log_likelihood,
        x0=np.array([alpha_0, mu_0, sigma_0]),
        args=(x, dt),
        method="L-BFGS-B",
        bounds=[(1e-6, None), (None, None), (1e-6, None)],
    )
    if not result.success:
        raise RuntimeError(f"OU-MLE-Optimierung nicht konvergiert: {result.message}")

    alpha, mu, sigma = result.x
    return OUParams(alpha=float(alpha), mu=float(mu), sigma=float(sigma))


def validate_residuals(X: pd.Series, dates: pd.Series, params: OUParams) -> dict:
    """Diagnostik der Modellannahmen (Normalitaet, Unabhaengigkeit der
    Residuen). Das Ergebnis wird IMMER zurueckgegeben und soll im Aufrufer
    nicht stillschweigend uebergangen werden -- eine gescheiterte
    Validierung soll strukturell sichtbar bleiben, nicht optional."""
    x = X.to_numpy(dtype=float)
    dt = _year_fraction(dates)
    x_t, x_next = x[:-1], x[1:]

    phi = np.exp(-params.alpha * dt)
    cond_mean = params.mu + (x_t - params.mu) * phi
    cond_std = params.sigma * np.sqrt((1 - phi**2) / (2 * params.alpha))
    standardized_resid = (x_next - cond_mean) / np.maximum(cond_std, 1e-12)

    jb_stat, jb_p = stats.jarque_bera(standardized_resid)
    lb_pvalue = _ljung_box_pvalue(standardized_resid, lags=10)

    normal_ok = bool(jb_p > 0.05)
    independent_ok = bool(lb_pvalue > 0.05)

    return {
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "ljung_box_pvalue": float(lb_pvalue),
        "residual_std": float(np.std(standardized_resid)),
        "normality_assumption_holds": normal_ok,
        "independence_assumption_holds": independent_ok,
        "model_assumptions_valid": normal_ok and independent_ok,
    }


class OUModel:
    """fit/predict/evaluate-Interface fuer das OU-Modell.

    evaluate() gibt die Diagnostik-Ergebnisse IMMER als Teil der Rueckgabe
    zurueck -- eine gescheiterte Validierung ist damit strukturell nicht
    uebersehbar."""

    def __init__(self) -> None:
        self.params: OUParams | None = None

    def fit(self, X: pd.Series, dates: pd.Series) -> "OUModel":
        self.params = estimate_ou_params(X, dates)
        return self

    def predict_conditional_mean(self, x_t: float, dt_years: float) -> float:
        if self.params is None:
            raise RuntimeError("Modell wurde noch nicht gefittet.")
        phi = np.exp(-self.params.alpha * dt_years)
        return self.params.mu + (x_t - self.params.mu) * phi

    def evaluate(self, X: pd.Series, dates: pd.Series) -> dict:
        """Bewertet das gefittete Modell auf (moeglicherweise neuen)
        Daten: Ein-Schritt-Vorhersagefehler (RMSE) plus die vollstaendige
        Residualdiagnostik aus validate_residuals()."""
        if self.params is None:
            raise RuntimeError("Modell wurde noch nicht gefittet.")

        x = X.to_numpy(dtype=float)
        dt = _year_fraction(dates)
        x_t, x_next = x[:-1], x[1:]
        phi = np.exp(-self.params.alpha * dt)
        pred = self.params.mu + (x_t - self.params.mu) * phi
        rmse = float(np.sqrt(np.mean((x_next - pred) ** 2)))
        mae = float(np.mean(np.abs(x_next - pred)))

        diagnostics = validate_residuals(X, dates, self.params)
        return {"rmse": rmse, "mae": mae, **diagnostics}
