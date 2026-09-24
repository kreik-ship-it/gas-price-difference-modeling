"""
Explorative Datenanalyse: deskriptive Statistik, Fehlwerte-Report,
Korrelationsmatrix, Verteilungen, Autokorrelation der Preisunterschiede.
Ergebnis: PNGs unter results/figures/.

    python scripts/generate_eda.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import load_raw_prices, compute_price_differences, missing_value_report, PRICE_DIFFERENCE_DEFINITIONS  # noqa: E402

FIGURES = Path(__file__).parent.parent / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

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


def descriptive_stats(diffs: pd.DataFrame) -> pd.DataFrame:
    cols = list(PRICE_DIFFERENCE_DEFINITIONS)
    stats = pd.DataFrame({
        "Mean": diffs[cols].mean(),
        "Median": diffs[cols].median(),
        "Std": diffs[cols].std(),
        "Min": diffs[cols].min(),
        "Max": diffs[cols].max(),
        "Skewness": diffs[cols].skew(),
        "Kurtosis": diffs[cols].kurtosis(),
        "N": diffs[cols].count(),
    })
    return stats.round(3)


def chart_distributions(diffs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharey=False)
    for ax, name, color in zip(axes, PRICE_DIFFERENCE_DEFINITIONS, PALETTE):
        s = diffs[name].dropna()
        ax.hist(s, bins=40, color=color, alpha=0.85)
        ax.axvline(s.mean(), color="#222222", linewidth=1, linestyle="--")
        ax.set_title(name.replace("TTF_", "TTF–"), fontsize=10)
        ax.set_xlabel("€/MWh", fontsize=9)
    axes[0].set_ylabel("Haeufigkeit")
    fig.suptitle("Verteilung der fünf Preisunterschiede (2017–2024)", y=1.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "price_difference_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_correlation(diffs: pd.DataFrame) -> pd.DataFrame:
    cols = list(PRICE_DIFFERENCE_DEFINITIONS)
    corr = diffs[cols].corr()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    labels = [c.replace("TTF_", "TTF–") for c in cols]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr.iloc[i, j]) > 0.6 else "#222222", fontsize=9)
    ax.set_title("Korrelation der Preisunterschiede untereinander")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "price_difference_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return corr.round(3)


def chart_acf(diffs: pd.DataFrame, name: str = "TTF_THE", nlags: int = 40) -> None:
    s = diffs[name].dropna().to_numpy()
    s = s - s.mean()
    var = np.sum(s**2)
    acf = [1.0] + [np.sum(s[:-k] * s[k:]) / var for k in range(1, nlags + 1)]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(range(nlags + 1), acf, color=PALETTE[0], width=0.6)
    ci = 1.96 / np.sqrt(len(s))
    ax.axhline(ci, color="#999999", linestyle="--", linewidth=0.8)
    ax.axhline(-ci, color="#999999", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Lag (Tage)")
    ax.set_ylabel("Autokorrelation")
    ax.set_title(f"Autokorrelationsfunktion: {name.replace('TTF_', 'TTF–')}")
    fig.tight_layout()
    fig.savefig(FIGURES / f"acf_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    prices = load_raw_prices("data/Historie_EGSI.xlsx")
    diffs = compute_price_differences(prices)

    print("== Fehlwerte-Report (Rohpreise) ==")
    print(missing_value_report(prices))

    print("\n== Deskriptive Statistik (Preisunterschiede) ==")
    stats = descriptive_stats(diffs)
    print(stats)

    print("\nErzeuge price_difference_distributions.png ...")
    chart_distributions(diffs)

    print("Erzeuge price_difference_correlation.png ...")
    corr = chart_correlation(diffs)
    print("\n== Korrelationsmatrix ==")
    print(corr)

    print("Erzeuge acf_TTF_THE.png ...")
    chart_acf(diffs, "TTF_THE")

    print("\nFertig -- siehe results/figures/")


if __name__ == "__main__":
    main()
