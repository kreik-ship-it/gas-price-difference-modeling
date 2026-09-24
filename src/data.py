"""
Datenladen, Preisunterschiede und Zeitraum-Split fuer das Ornstein-
Uhlenbeck-Modellierungsprojekt.

Rohdatenquelle: Historie_EGSI.xlsx, Blatt "Pivot", Spalten A:I, skiprows=1.
Oeffentlich zugaengliche Marktpreise; die Datei liegt unter data/ im Repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Die sechs ueber die gesamte Historie hinweg durchgaengigen Preissaeulen.
# THE_combined ersetzt die getrennten THE/NCG-Notierungen (deutscher Markt,
# Umstellung von NCG/GASPOOL auf THE 2021), PEG_combined ersetzt PEG/PEGN
# (franzoesischer Markt, Zusammenlegung der Marktgebiete 2018).
PRICE_COLUMNS = ["TTF", "THE_combined", "PEG_combined", "ZTP", "VTP AUT", "CZ"]

# Preisunterschiede zwischen dem TTF-Referenzmarkt und den uebrigen Huebs.
PRICE_DIFFERENCE_DEFINITIONS = {
    "TTF_THE": ("TTF", "THE_combined"),
    "TTF_PEG": ("TTF", "PEG_combined"),
    "TTF_ZTP": ("TTF", "ZTP"),
    "TTF_VTP": ("TTF", "VTP AUT"),
    "TTF_CZ": ("TTF", "CZ"),
}


def load_raw_prices(xlsx_path: str | Path) -> pd.DataFrame:
    """Laedt die Rohpreisreihen und fusioniert die strukturell gebrochenen
    Marktgebiets-Paare (THE/NCG, PEG/PEGN) zu durchgaengigen Serien.

    Parameters
    ----------
    xlsx_path:
        Pfad zur Historie_EGSI.xlsx.

    Returns
    -------
    DataFrame mit Spalte "Date" (normalisiert auf Mitternacht) und den
    sechs Preisspalten aus PRICE_COLUMNS, chronologisch sortiert.
    """
    df = pd.read_excel(xlsx_path, sheet_name="Pivot", usecols="A:I", skiprows=1)
    df = df.rename(columns={"Zeilenbeschriftungen": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    df["THE_combined"] = df["THE"].combine_first(df["NCG"])
    df["PEG_combined"] = df["PEG"].combine_first(df["PEGN"])
    df = df.drop(columns=["NCG", "THE", "PEGN", "PEG"])

    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date"] + PRICE_COLUMNS]


def compute_price_differences(prices: pd.DataFrame) -> pd.DataFrame:
    """Berechnet die fuenf TTF-Preisunterschiede aus den Rohpreisen.

    Reine Differenzbildung (TTF minus jeweiliger Hub), keine weitere
    Transformation. Vorzeichenkonvention: ein positiver Wert bedeutet
    TTF > Ziel-Hub.
    """
    diffs = pd.DataFrame(index=prices.index)
    diffs["Date"] = prices["Date"]
    for name, (a, b) in PRICE_DIFFERENCE_DEFINITIONS.items():
        diffs[name] = prices[a] - prices[b]
    return diffs


def missing_value_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Fehlende-Werte-Report je Preissaeule."""
    rows = []
    for col in PRICE_COLUMNS:
        series = prices[col]
        first_idx = series.first_valid_index()
        last_idx = series.last_valid_index()
        rows.append(
            {
                "column": col,
                "first_date": prices.loc[first_idx, "Date"] if first_idx is not None else pd.NaT,
                "last_date": prices.loc[last_idx, "Date"] if last_idx is not None else pd.NaT,
                "observations": int(series.count()),
                "missing": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean()) * 100, 2),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def load_clean_data(xlsx_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bequemlichkeitsfunktion: laedt, bereinigt und liefert Preise +
    Preisunterschiede in einem Aufruf."""
    prices = load_raw_prices(xlsx_path)
    diffs = compute_price_differences(prices)
    return prices, diffs


# ----------------------------------------------------------------
# Zeitraum-Split: Kalibrierung nach Marktregime + unberuehrter Test
# ----------------------------------------------------------------
#
# Die Preisunterschiede haben sich infolge des russischen Angriffskriegs
# gegen die Ukraine (Februar 2022) strukturell veraendert. Eine einzelne
# durchgaengige Kalibrierung wuerde Vorkriegs- und Nachkriegsdynamik
# vermischen, daher zwei getrennte Kalibrierungszeitraeume statt eines
# einzelnen Trainingsfensters. Das OU-Modell hat keine Hyperparameter, die
# auf einem separaten Validierungsfenster gewaehlt werden muessten -- der
# Split bleibt entsprechend auf Kalibrierung + Test beschraenkt.

REGIME_1_START = pd.Timestamp("2017-12-30")
REGIME_1_END = pd.Timestamp("2021-12-31")

REGIME_2_START = pd.Timestamp("2022-01-01")
REGIME_2_END = pd.Timestamp("2023-08-31")

TEST_START = pd.Timestamp("2023-09-01")
TEST_END = pd.Timestamp("2024-04-18")


@dataclass(frozen=True)
class SplitResult:
    regime_1: pd.DataFrame  # Kalibrierung "vor dem Angriffskrieg"
    regime_2: pd.DataFrame  # Kalibrierung "nach Kriegsbeginn"
    test: pd.DataFrame  # unberuehrter Holdout


def split_calibration_test(df: pd.DataFrame, date_col: str = "Date") -> SplitResult:
    """Teilt einen Preis- oder Preisunterschieds-DataFrame in Regime-1-
    Kalibrierung, Regime-2-Kalibrierung und Testfenster auf.

    Wirft einen Fehler, wenn `date_col` keine sortierte, eindeutige
    Datumsspalte ist -- ein stiller falscher Split waere schlimmer als ein
    lauter Fehler hier.
    """
    dates = pd.to_datetime(df[date_col])
    if not dates.is_monotonic_increasing:
        raise ValueError(
            f"'{date_col}' ist nicht chronologisch sortiert; vor dem Split "
            "sortieren (df.sort_values(date_col))."
        )

    regime_1 = df[(dates >= REGIME_1_START) & (dates <= REGIME_1_END)]
    regime_2 = df[(dates >= REGIME_2_START) & (dates <= REGIME_2_END)]
    test = df[(dates >= TEST_START) & (dates <= TEST_END)]

    return SplitResult(
        regime_1=regime_1.reset_index(drop=True),
        regime_2=regime_2.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
