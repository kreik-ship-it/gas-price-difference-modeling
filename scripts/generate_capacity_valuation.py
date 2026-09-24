"""
Bewertung der Transportkapazitaet: empirischer Referenzwert (undiskontiert)
vs. OU-modellbasierter Monte-Carlo-Wert (diskontiert), je Preisunterschied
und Untersuchungsperiode. Annahmen: 1 MWh/Tag Kapazitaet, 365 Liefertage,
Transportkosten K=0,6 EUR/MWh in beide Richtungen, Zinssatz r=3% p.a.,
M=1.000 simulierte Pfade.

    python scripts/generate_capacity_valuation.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import load_raw_prices, compute_price_differences  # noqa: E402
from evaluation import capacity_valuation_table  # noqa: E402

FIGURES = Path(__file__).parent.parent / "results" / "figures"
TABLES = Path(__file__).parent.parent / "results" / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

COLOR_EMPIRICAL = "#4C72B0"
COLOR_OU = "#C44E52"

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


def chart_capacity_comparison(table: pd.DataFrame) -> None:
    gesamt = table[table["Periode"] == "Gesamtperiode"].set_index("Preisunterschied")
    names = [n.replace("TTF_", "TTF–") for n in gesamt.index]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, gesamt["Empirisch_EUR"], width, label="Empirisch (Referenz, undiskontiert)", color=COLOR_EMPIRICAL)
    ax.bar(x + width / 2, gesamt["OU_Modell_EUR"], width, label="OU-Modell (Monte-Carlo, diskontiert)", color=COLOR_OU)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Wert der Transportkapazität (€, 1 MWh/Tag, 365 Liefertage)")
    ax.set_title("Transportkapazitäts-Bewertung: Gesamtperiode")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURES / "06_capacity_valuation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    prices = load_raw_prices("data/Historie_EGSI.xlsx")
    diffs = compute_price_differences(prices)

    print("Berechne Transportkapazitäts-Bewertung (K=0.6 EUR/MWh, r=3%, M=1000 Pfade) ...")
    table = capacity_valuation_table(diffs)
    table.to_csv(TABLES / "capacity_valuation.csv", index=False)

    print("\n== Transportkapazitäts-Bewertung ==")
    print(table.to_string(index=False))

    print("\nErzeuge 06_capacity_valuation.png ...")
    chart_capacity_comparison(table)

    print("\nFertig -- siehe results/tables/capacity_valuation.csv und results/figures/06_capacity_valuation.png")


if __name__ == "__main__":
    main()
