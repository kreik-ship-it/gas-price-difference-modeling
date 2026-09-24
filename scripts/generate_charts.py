"""
Erzeugt die fuenf Kernabbildungen fuer README/Ergebnisdarstellung:
Preisunterschieds-Verlauf, Kalibrierungs-/Test-Split, OU-Kalibrierung
(Mean-Reversion-Level), Out-of-Sample-Prognose und Backtest-Ergebnisse
ueber alle fuenf Preisunterschiede. Fokus-Preisunterschied fuer die
Detail-Abbildungen (3 und 4): TTF_THE.

    python scripts/generate_charts.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import (  # noqa: E402
    load_raw_prices, compute_price_differences,
    REGIME_1_START, REGIME_1_END, REGIME_2_START, REGIME_2_END,
    TEST_START, TEST_END, split_calibration_test,
)
from ou_model import OUModel  # noqa: E402
from evaluation import summarize_backtest  # noqa: E402

FIGURES = Path(__file__).parent.parent / "results" / "figures"
TABLES = Path(__file__).parent.parent / "results" / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

COLOR_REGIME1 = "#4C72B0"
COLOR_REGIME2 = "#DD8452"
COLOR_TEST = "#55A868"
COLOR_ACTUAL = "#333333"
COLOR_FIT = "#C44E52"

FOCUS = "TTF_THE"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def chart_1_price_difference_history(diffs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    s = diffs.dropna(subset=[FOCUS])
    ax.plot(s["Date"], s[FOCUS], color=COLOR_ACTUAL, linewidth=0.8, label=f"{FOCUS.replace('TTF_', 'TTF–')} Preisunterschied")
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")

    spans = [
        (REGIME_1_START, REGIME_1_END, COLOR_REGIME1, "Regime 1 (Kalibrierung)"),
        (REGIME_2_START, REGIME_2_END, COLOR_REGIME2, "Regime 2 (Kalibrierung)"),
        (TEST_START, TEST_END, COLOR_TEST, "Test"),
    ]
    for start, end, color, _ in spans:
        ax.axvspan(start, end, color=color, alpha=0.12, linewidth=0)

    ax.set_title("Entwicklung des TTF–THE-Preisunterschieds, 2017–2024")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Preisunterschied (€/MWh)")

    handles = [plt.Line2D([0], [0], color=COLOR_ACTUAL, linewidth=1.2, label="Preisunterschied")]
    handles += [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.25, label=l) for _, _, c, l in spans]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "01_price_difference_history.png", dpi=150)
    plt.close(fig)


def chart_2_split_timeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 2.6))
    segments = [
        (REGIME_1_START, REGIME_1_END, COLOR_REGIME1, "Regime 1\n(Kalibrierung, vor 2022)"),
        (REGIME_2_START, REGIME_2_END, COLOR_REGIME2, "Regime 2\n(Kalibrierung, ab 2022)"),
        (TEST_START, TEST_END, COLOR_TEST, "Test\n(unberührt)"),
    ]
    for start, end, color, label in segments:
        width_days = (end - start).days
        ax.barh(0, width_days, left=start, height=0.6, color=color, edgecolor="white")
        mid = start + (end - start) / 2
        ax.text(mid, 0, label, ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlim(REGIME_1_START, TEST_END)
    ax.set_title("Kalibrierungs-/Test-Split")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "02_split_timeline.png", dpi=150)
    plt.close(fig)


def chart_3_ou_calibration(diffs: pd.DataFrame) -> "OUModel":
    sub = diffs[["Date", FOCUS]].dropna().reset_index(drop=True)
    split = split_calibration_test(sub)

    model = OUModel().fit(split.regime_2[FOCUS], split.regime_2["Date"])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(split.regime_2["Date"], split.regime_2[FOCUS], color=COLOR_ACTUAL, linewidth=0.9, label="Beobachtet (Regime 2)")
    ax.axhline(model.params.mu, color=COLOR_FIT, linewidth=1.5, linestyle="--",
               label=f"Geschätztes Mean-Reversion-Level μ = {model.params.mu:.3f}")
    ax.set_title(
        f"OU-Kalibrierung auf Regime 2 ({FOCUS.replace('TTF_', 'TTF–')}): "
        f"α={model.params.alpha:.2f}/Jahr, Halbwertszeit≈{model.params.half_life_days:.0f} Tage"
    )
    ax.set_xlabel("Datum")
    ax.set_ylabel("Preisunterschied (€/MWh)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_ou_calibration.png", dpi=150)
    plt.close(fig)
    return model


def chart_4_oos_forecast(diffs: pd.DataFrame, model: "OUModel") -> None:
    sub = diffs[["Date", FOCUS]].dropna().reset_index(drop=True)
    split = split_calibration_test(sub)
    test = split.test

    from ou_model import _year_fraction

    dt = _year_fraction(test["Date"])
    x = test[FOCUS].to_numpy(dtype=float)
    x_t = x[:-1]
    phi = np.exp(-model.params.alpha * dt)
    pred_mean = model.params.mu + (x_t - model.params.mu) * phi
    pred_std = model.params.sigma * np.sqrt((1 - phi**2) / (2 * model.params.alpha))

    dates_pred = test["Date"].to_numpy()[1:]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(test["Date"], x, color=COLOR_ACTUAL, linewidth=1.0, label="Beobachtet (Test)")
    ax.plot(dates_pred, pred_mean, color=COLOR_FIT, linewidth=1.2, label="Ein-Schritt-Prognose (bedingter Erwartungswert)")
    ax.fill_between(
        dates_pred, pred_mean - 1.96 * pred_std, pred_mean + 1.96 * pred_std,
        color=COLOR_FIT, alpha=0.15, label="95%-Prognoseintervall",
    )
    ax.set_title(f"Out-of-Sample-Prognose auf dem Testfenster ({FOCUS.replace('TTF_', 'TTF–')})")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Preisunterschied (€/MWh)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_oos_forecast.png", dpi=150)
    plt.close(fig)


def chart_5_backtest_results(diffs: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_backtest(diffs)
    summary.to_csv(TABLES / "backtest_results.csv")

    names = [n.replace("TTF_", "TTF–") for n in summary.index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, summary["RMSE"], color=COLOR_REGIME2)
    ax.set_ylabel("RMSE auf Testfenster (09/2023–04/2024)")
    ax.set_title("OU-Modell: Backtest-Ergebnisse je Preisunterschied")
    fig.tight_layout()
    fig.savefig(FIGURES / "05_backtest_results.png", dpi=150)
    plt.close(fig)
    return summary


def main() -> None:
    prices = load_raw_prices("data/Historie_EGSI.xlsx")
    diffs = compute_price_differences(prices)

    print("Erzeuge 01_price_difference_history.png ...")
    chart_1_price_difference_history(diffs)

    print("Erzeuge 02_split_timeline.png ...")
    chart_2_split_timeline()

    print("Erzeuge 03_ou_calibration.png ...")
    model = chart_3_ou_calibration(diffs)

    print("Erzeuge 04_oos_forecast.png ...")
    chart_4_oos_forecast(diffs, model)

    print("Erzeuge 05_backtest_results.png ...")
    summary = chart_5_backtest_results(diffs)
    print("\n== Backtest-Ergebnisse (auch in results/tables/backtest_results.csv) ==")
    print(summary.round(3))

    print("\nFertig -- siehe results/figures/ und results/tables/")


if __name__ == "__main__":
    main()
