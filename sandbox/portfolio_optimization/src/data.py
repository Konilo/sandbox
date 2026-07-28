"""Data acquisition for the portfolio covariance study.

Each proxy sleeve is turned into a monthly *simple* return series in EUR. Raw
source files are committed under ``data/`` so the pipeline is deterministic and
offline once the snapshot exists; the ``download_*`` helpers exist to refresh
those snapshots reproducibly.

Sources
-------
Gold: LBMA PM gold fix, ``https://prices.lbma.org.uk/json/gold_pm.json``
      (snapshot fetched 2026-07-25). Each record is ``{"d": date,
      "v": [USD, GBP, EUR]}``; we take the EUR column directly, i.e. LBMA's own
      EUR fixing, so no separate FX conversion is applied.

Equity: MSCI World Standard (Large+Mid Cap), Net, EUR, from the MSCI Index Data
      Search (``app2.msci.com/products/index-data-search``, indexId 990100,
      priceLevel NETR, currency EUR; snapshot 2026-07-25). Already month-end
      monthly levels; the trailing MSCI legal disclaimer rows are dropped on load.

Bonds: FTSE World Government Bond - Developed Markets (Hedged EUR), from Curvo's
      compiled backtest dataset (``curvo.eu/backtest/data/...json``; snapshot
      2026-07-25). Monthly month-end index levels back to 1985. This is Curvo's
      compilation, not primary FTSE Russell data.

Trend: SG CTA Index, the live net-of-fee managed-futures benchmark that the
      vehicle DBMFE replicates, from SG Prime Services (snapshot 2026-07-25).
      Daily USD levels (the ``VAMI`` column, base 1000 at 2000-01), representing
      the net daily return of a pool of the largest CTAs -- live funds, not a
      back-tested replication. DBMFE (LU2951555403) is a USD-base, *unhedged* EUR
      share class, so the EUR investor bears EUR/USD: we convert the USD level to
      EUR by dividing by EURUSD (USD per EUR), using Curvo's ``eur-usd.json``
      (snapshot 2026-07-25), then take month-end returns.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

LBMA_GOLD_PM_URL = "https://prices.lbma.org.uk/json/gold_pm.json"
GOLD_RAW = DATA_DIR / "lbma_gold_pm_2026-07-25.json"
MSCI_WORLD_RAW = DATA_DIR / "msci_world_net_eur_2026-07-25.xls"

CURVO_WGBI_URL = (
    "https://curvo.eu/backtest/data/"
    "ftse-world-government-bond-developed-markets-hedged-eur.json"
)
WGBI_RAW = DATA_DIR / "curvo_ftse_wgbi_dev_hedged_eur_2026-07-25.json"

SG_CTA_RAW = DATA_DIR / "sg_cta_index_2026-07-25.xls"
CURVO_EURUSD_URL = "https://curvo.eu/backtest/data/eur-usd.json"
EURUSD_RAW = DATA_DIR / "curvo_eur_usd_2026-07-25.json"


def download_lbma_gold(dest: Path = GOLD_RAW) -> Path:
    """Download the LBMA PM gold-fix JSON to ``dest`` (refresh the snapshot)."""
    with urllib.request.urlopen(LBMA_GOLD_PM_URL, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def load_gold_eur_levels(raw_path: Path = GOLD_RAW) -> pd.Series:
    """Daily LBMA PM gold fix in EUR, as a sorted level series.

    The EUR fixing (``v[2]``) is null before 1999; those rows are dropped.
    """
    records = json.loads(raw_path.read_text())
    dates: list[pd.Timestamp] = []
    prices: list[float] = []
    for r in records:
        price_eur = r["v"][2]  # [USD, GBP, EUR]
        if price_eur is None:
            continue
        dates.append(pd.Timestamp(r["d"]))
        prices.append(price_eur)
    return pd.Series(prices, index=pd.DatetimeIndex(dates), name="gold_eur").sort_index()


def monthly_returns_from_levels(levels: pd.Series) -> pd.Series:
    """Month-end simple returns from a daily level series.

    Month-end is the last published fix in each calendar month (``"ME"``).
    ``fill_method=None`` avoids silently forward-filling across any gap.
    """
    monthly = levels.resample("ME").last()
    returns = monthly.pct_change(fill_method=None).dropna()
    return returns.rename(f"{levels.name}_ret")


def gold_eur_monthly_returns(raw_path: Path = GOLD_RAW) -> pd.Series:
    """Monthly simple EUR returns for the gold sleeve (LBMA PM fix)."""
    return monthly_returns_from_levels(load_gold_eur_levels(raw_path))


def load_msci_world_eur_levels(raw_path: Path = MSCI_WORLD_RAW) -> pd.Series:
    """MSCI World Net EUR month-end levels from the MSCI export.

    The sheet carries metadata rows on top and a legal disclaimer at the bottom;
    we keep only rows whose first column parses as a ``"%b %d, %Y"`` date and
    whose second column is numeric.
    """
    raw = pd.read_excel(raw_path, sheet_name=0, header=None, engine="calamine")
    dates = pd.to_datetime(raw[0], format="%b %d, %Y", errors="coerce")
    values = pd.to_numeric(raw[1], errors="coerce")
    mask = dates.notna() & values.notna()
    return pd.Series(
        values[mask].to_numpy(),
        index=pd.DatetimeIndex(dates[mask]),
        name="msci_world_eur",
    ).sort_index()


def msci_world_eur_monthly_returns(raw_path: Path = MSCI_WORLD_RAW) -> pd.Series:
    """Monthly simple EUR returns for the equity sleeve (MSCI World Net)."""
    return monthly_returns_from_levels(load_msci_world_eur_levels(raw_path))


def download_curvo_wgbi(dest: Path = WGBI_RAW) -> Path:
    """Download the Curvo WGBI-hedged-EUR levels JSON to ``dest``."""
    with urllib.request.urlopen(CURVO_WGBI_URL, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def load_wgbi_eur_levels(raw_path: Path = WGBI_RAW) -> pd.Series:
    """FTSE WGBI Developed (Hedged EUR) month-end levels from the Curvo JSON.

    The file is a list of ``{"date": "YYYY-MM-DD", "value": level}`` records.
    """
    records = json.loads(raw_path.read_text())
    index = pd.DatetimeIndex([pd.Timestamp(r["date"]) for r in records])
    values = [r["value"] for r in records]
    return pd.Series(values, index=index, name="wgbi_eur_hedged").sort_index()


def wgbi_eur_hedged_monthly_returns(raw_path: Path = WGBI_RAW) -> pd.Series:
    """Monthly simple EUR returns for the bond sleeve (WGBI Developed, EUR-hedged)."""
    return monthly_returns_from_levels(load_wgbi_eur_levels(raw_path))


def download_curvo_eurusd(dest: Path = EURUSD_RAW) -> Path:
    """Download the Curvo EURUSD (USD per EUR) levels JSON to ``dest``."""
    with urllib.request.urlopen(CURVO_EURUSD_URL, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def load_eurusd(raw_path: Path = EURUSD_RAW) -> pd.Series:
    """EURUSD (USD per EUR), month-end monthly, from the Curvo JSON."""
    records = json.loads(raw_path.read_text())
    index = pd.DatetimeIndex([pd.Timestamp(r["date"]) for r in records])
    values = [r["value"] for r in records]
    return pd.Series(values, index=index, name="eurusd").sort_index()


def load_sg_cta_usd_levels(raw_path: Path = SG_CTA_RAW) -> pd.Series:
    """SG CTA Index daily USD levels (net-of-fee), from the VAMI column.

    The sheet has a title row, a spaced-out header row (``Dstamp | ROR | VAMI |
    MTD | QTD | YTD``), then data. We read it header-less and keep rows whose
    first column parses as a date and whose third column (VAMI, the index level)
    is numeric.
    """
    raw = pd.read_excel(raw_path, header=None, engine="calamine")
    # The date cells parse as datetimes; the title/header rows are strings. Mark
    # the column mixed so those non-dates coerce to NaT without a per-row warning.
    dates = pd.to_datetime(raw[0], format="mixed", errors="coerce")
    values = pd.to_numeric(raw[2], errors="coerce")  # VAMI level
    mask = dates.notna() & values.notna()
    return pd.Series(
        values[mask].to_numpy(),
        index=pd.DatetimeIndex(dates[mask]),
        name="trend_usd",
    ).sort_index()


def load_trend_eur_levels(
    cta_path: Path = SG_CTA_RAW, eurusd_path: Path = EURUSD_RAW
) -> pd.Series:
    """Month-end trend levels in EUR: USD SG CTA Index level divided by EURUSD.

    The USD index is resampled to month-end, then converted at the month-end
    EURUSD (USD per EUR); division aligns on the shared month-end index, so the
    result ends at the earlier of the two series (EURUSD, currently 2026-06).
    """
    usd_me = load_sg_cta_usd_levels(cta_path).resample("ME").last()
    eurusd_me = load_eurusd(eurusd_path)
    return (usd_me / eurusd_me).dropna().rename("trend_eur")


def trend_eur_monthly_returns(
    cta_path: Path = SG_CTA_RAW, eurusd_path: Path = EURUSD_RAW
) -> pd.Series:
    """Monthly simple EUR returns for the trend sleeve (DBMFE, USD unhedged)."""
    return monthly_returns_from_levels(load_trend_eur_levels(cta_path, eurusd_path))


def monthly_returns_matrix() -> pd.DataFrame:
    """The four sleeve return series aligned on their common month-end window."""
    series = {
        "equity": msci_world_eur_monthly_returns(),
        "bonds": wgbi_eur_hedged_monthly_returns(),
        "trend": trend_eur_monthly_returns(),
        "gold": gold_eur_monthly_returns(),
    }
    return pd.concat(series, axis=1, join="inner").sort_index()
