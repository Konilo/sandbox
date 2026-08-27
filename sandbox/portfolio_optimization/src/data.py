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

Equity: MSCI ACWI Standard (Large+Mid Cap), Net, EUR, from the MSCI Index Data
      Search (``app2.msci.com/products/index-data-search``, indexId 892400,
      priceLevel NETR, currency EUR; daily snapshot 2026-08-04). Daily index
      levels, resampled to month-end for the monthly study (and used raw for the
      daily stress test); the trailing MSCI legal disclaimer rows are dropped on load.

Bonds: US Treasury 7-10y, EUR-hedged. The vehicle is a EUR-hedged UCITS ETF on the
      ICE US Treasury 7-10y index (e.g. iShares CEMF, IE000K1VI152; or Amundi 7USH
      on the Bloomberg 7-10y index). The EUR-hedged total return is the local USD
      total return plus the one-month covered-interest-parity carry,
      ``(i_EUR - i_USD) / 12`` -- the covered-interest-parity equivalent of the
      rolling one-month currency forward that the ICE hedged index itself applies
      (ICE Bond Index Methodologies, pp. 22-23,
      https://www.ice.com/publicdocs/data/Bond_Index_Methodologies.pdf). Proxied
      by IEF (US-listed, same ICE index, long
      history): its dividend-adjusted close is the USD total-return level (Yahoo
      daily, snapshot 2026-08-08, from 2002-07), and month-end USD returns plus the
      carry from FRED short rates (1-month UST DGS1MO, euro overnight
      IRSTCI01EZM156N) give the hedged EUR return. Validated against the live
      hedged ETF 7USH (2023+) on both correlation and level by
      ``hedge_tracking_check``.

Trend: SG CTA Index, the live net-of-fee managed-futures benchmark that the
      vehicle DBMFE replicates, from SG Prime Services (snapshot 2026-07-25).
      Daily USD levels (the ``VAMI`` column, base 1000 at 2000-01), representing
      the net daily return of a pool of the largest CTAs -- live funds, not a
      back-tested replication. DBMFE (LU2951555403) is a USD-base, *unhedged* EUR
      share class, so the EUR investor bears EUR/USD: we convert the USD level to
      EUR by dividing by EURUSD (USD per EUR), using the ECB daily reference rate
      (snapshot 2026-08-04, resampled to month-end), then take month-end returns.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

LBMA_GOLD_PM_URL = "https://prices.lbma.org.uk/json/gold_pm.json"
GOLD_RAW = DATA_DIR / "lbma_gold_pm_2026-07-25.json"
MSCI_ACWI_RAW = DATA_DIR / "msci_acwi_net_eur_daily_2026-08-04.xls"

IEF_YAHOO_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/IEF"
    "?period1=0&period2=9999999999&interval=1d&events=div"
)
IEF_RAW = DATA_DIR / "yahoo_ief_daily_2026-08-08.json"

# Short rates for the EUR hedge carry (covered-interest-parity): the 1-month US
# Treasury rate and the euro-area overnight interbank rate, both from FRED.
FRED_USD_1M_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO"
FRED_USD_1M_RAW = DATA_DIR / "fred_DGS1MO_2026-08-08.csv"
FRED_EUR_ON_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRSTCI01EZM156N"
FRED_EUR_ON_RAW = DATA_DIR / "fred_IRSTCI01EZM156N_2026-08-08.csv"

# Live EUR-hedged UCITS ETF, used only to validate the reconstructed hedge.
# Month-end dividend-adjusted closes from Yahoo, snapshot 2026-08-26.
LIVE_HEDGED_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?range=5y&interval=1mo&events=div%2Csplit"
)
LIVE_HEDGED_RAW = {"7USH": DATA_DIR / "yahoo_7USH_monthly_2026-08-26.json"}

SG_CTA_RAW = DATA_DIR / "sg_cta_index_2026-07-25.xls"
ECB_EURUSD_URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata"
EURUSD_RAW = DATA_DIR / "ecb_eurusd_daily_2026-08-04.json"


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


def load_msci_acwi_eur_levels(raw_path: Path = MSCI_ACWI_RAW) -> pd.Series:
    """MSCI ACWI Net EUR daily levels from the MSCI export.

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
        name="msci_acwi_eur",
    ).sort_index()


def msci_acwi_eur_monthly_returns(raw_path: Path = MSCI_ACWI_RAW) -> pd.Series:
    """Monthly simple EUR returns for the equity sleeve (MSCI ACWI Net)."""
    return monthly_returns_from_levels(load_msci_acwi_eur_levels(raw_path))


def download_ief(dest: Path = IEF_RAW) -> Path:
    """Download IEF daily dividend-adjusted close (US iShares 7-10y Treasury) as JSON.

    IEF is distributing, so the *adjusted* close (dividends reinvested), not the raw
    close, is the USD total-return level.
    """
    import datetime as dt

    req = urllib.request.Request(IEF_YAHOO_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode())["chart"]["result"][0]
    timestamps = result["timestamp"]
    adjclose = result["indicators"]["adjclose"][0]["adjclose"]
    records = [
        {"date": dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"), "value": v}
        for t, v in zip(timestamps, adjclose, strict=True)
        if v is not None
    ]
    dest.write_text(json.dumps(records))
    return dest


def load_ief_usd_levels(raw_path: Path = IEF_RAW) -> pd.Series:
    """IEF daily USD total-return levels (dividend-adjusted close).

    IEF tracks the ICE US Treasury 7-10y index -- the same index the EUR-hedged
    UCITS wrapper the study targets (e.g. iShares CEMF) hedges -- with a longer
    history. The file is a list of ``{"date", "value"}`` records.
    """
    records = json.loads(raw_path.read_text())
    index = pd.DatetimeIndex([pd.Timestamp(r["date"]) for r in records])
    values = [r["value"] for r in records]
    return pd.Series(values, index=index, name="ustsy_usd").sort_index()


def download_fred(url: str, dest: Path) -> Path:
    """Download a FRED series CSV (``date,value`` rows) to ``dest``."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def _fred_rate_me(raw_path: Path) -> pd.Series:
    """Month-end FRED rate as a decimal (percent / 100); missing '.' rows dropped."""
    df = pd.read_csv(raw_path)
    col = df.columns[1]
    df = df[df[col] != "."]
    s = pd.Series(
        pd.to_numeric(df[col]).to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0])),
    ).sort_index()
    s = s.resample("ME").last() / 100.0
    s.index = s.index.to_period("M").to_timestamp("M")
    return s


def eur_minus_usd_carry(
    usd_path: Path = FRED_USD_1M_RAW, eur_path: Path = FRED_EUR_ON_RAW
) -> pd.Series:
    """Month-end EUR-minus-USD short-rate differential (annualized decimal).

    The covered-interest-parity carry of a EUR-hedged USD bond: ``i_EUR - i_USD``,
    negative when USD rates exceed EUR rates (the usual hedging cost). The euro leg
    (FRED IRSTCI01EZM156N) lags ~7 months and is forward-filled on the tail; the
    resulting carry error is ~0.02 %/month, immaterial to the covariance.
    """
    usd = _fred_rate_me(usd_path)
    eur = _fred_rate_me(eur_path)
    return (eur.reindex(usd.index).ffill() - usd).dropna().rename("carry")


def ustsy_hedged_monthly_returns(
    ief_path: Path = IEF_RAW,
    usd_path: Path = FRED_USD_1M_RAW,
    eur_path: Path = FRED_EUR_ON_RAW,
) -> pd.Series:
    """Monthly EUR-hedged US Tsy 7-10y returns: local USD return + CIP carry.

        hedged = local_usd_return + (i_EUR - i_USD) / 12

    ``local_usd_return`` is IEF's month-end total return (USD); the carry is set at
    the start of the month (prior month-end rates); ``hedge_tracking_check``
    validates it against the live hedged ETF 7USH (2023+).
    """
    local = load_ief_usd_levels(ief_path).resample("ME").last().pct_change()
    carry = eur_minus_usd_carry(usd_path, eur_path).reindex(local.index).ffill().shift(1) / 12.0
    return (local + carry).dropna().rename("ustsy_hedged_ret")


def live_hedged_monthly_returns(name: str) -> pd.Series:
    """Month-end returns of a live EUR-hedged UCITS ETF.

    ``{"date", "value"}`` records. Yahoo labels a monthly bar at the start of its
    period, which the UTC offset lands on the prior month-end; the snapshot is
    written already shifted onto the month the bar covers.
    """
    records = json.loads(LIVE_HEDGED_RAW[name].read_text())
    index = pd.DatetimeIndex([pd.Timestamp(r["date"]) for r in records])
    levels = pd.Series([r["value"] for r in records], index=index).sort_index()
    return levels.pct_change().dropna().rename(name)


def hedge_tracking_check() -> pd.DataFrame:
    """Reconstructed hedge vs the live ETF: correlation and annualized level."""
    study = ustsy_hedged_monthly_returns()
    rows = {}
    for name in LIVE_HEDGED_RAW:
        joined = pd.concat(
            {"study": study, "live": live_hedged_monthly_returns(name)}, axis=1
        ).dropna()
        annualized = (1.0 + joined).prod() ** (12.0 / len(joined)) - 1.0
        rows[name] = {
            "months": len(joined),
            "from": f"{joined.index[0]:%Y-%m}",
            "to": f"{joined.index[-1]:%Y-%m}",
            "correlation": joined["study"].corr(joined["live"]),
            "study (%/yr)": annualized["study"],
            "live (%/yr)": annualized["live"],
            "difference (bp/yr)": (annualized["study"] - annualized["live"]) * 1e4,
        }
    return pd.DataFrame(rows).T


def download_ecb_eurusd(dest: Path = EURUSD_RAW) -> Path:
    """Download the ECB daily EURUSD reference rate (USD per EUR) to ``dest`` as JSON."""
    req = urllib.request.Request(ECB_EURUSD_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        lines = resp.read().decode().strip().split("\n")
    header = lines[0].split(",")
    di, vi = header.index("TIME_PERIOD"), header.index("OBS_VALUE")
    records = [
        {"date": cells[di], "value": float(cells[vi])}
        for cells in (line.split(",") for line in lines[1:])
        if cells[vi]
    ]
    dest.write_text(json.dumps(records))
    return dest


def load_eurusd(raw_path: Path = EURUSD_RAW) -> pd.Series:
    """EURUSD (USD per EUR), daily, from the ECB reference-rate JSON snapshot."""
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


def load_trend_eur_levels(cta_path: Path = SG_CTA_RAW, eurusd_path: Path = EURUSD_RAW) -> pd.Series:
    """Month-end trend levels in EUR: USD SG CTA Index level divided by EURUSD.

    Both series are resampled to month-end (the SG CTA USD level and the ECB
    daily EURUSD, USD per EUR); division aligns on the shared month-end index, so
    the result ends at the earlier of the two.
    """
    usd_me = load_sg_cta_usd_levels(cta_path).resample("ME").last()
    eurusd_me = load_eurusd(eurusd_path).resample("ME").last()
    return (usd_me / eurusd_me).dropna().rename("trend_eur")


def trend_eur_monthly_returns(
    cta_path: Path = SG_CTA_RAW, eurusd_path: Path = EURUSD_RAW
) -> pd.Series:
    """Monthly simple EUR returns for the trend sleeve (DBMFE, USD unhedged)."""
    return monthly_returns_from_levels(load_trend_eur_levels(cta_path, eurusd_path))


def monthly_returns_matrix() -> pd.DataFrame:
    """The four sleeve return series aligned on their common month-end window."""
    series = {
        "equity": msci_acwi_eur_monthly_returns(),
        "bonds": ustsy_hedged_monthly_returns(),
        "trend": trend_eur_monthly_returns(),
        "gold": gold_eur_monthly_returns(),
    }
    return pd.concat(series, axis=1, join="inner").sort_index()


# --- Daily series (margin stress test) -------------------------------------
# The margin test runs daily to catch intra-month spikes. Equity/gold/trend are
# daily at source; the bond sleeve is the same hedged US Tsy construction as the
# monthly sleeve (IEF daily plus the CIP carry), so the common daily window
# reaches back to 2002-07 (IEF) -- covering every crisis the monthly sample does.


def load_ustsy_hedged_daily_levels(
    ief_path: Path = IEF_RAW,
    usd_path: Path = FRED_USD_1M_RAW,
    eur_path: Path = FRED_EUR_ON_RAW,
) -> pd.Series:
    """Daily EUR-hedged US Tsy 7-10y total-return level: IEF local + CIP carry.

    Daily analogue of ``ustsy_hedged_monthly_returns``: IEF's daily USD total
    return plus a daily slice of the (i_EUR - i_USD) carry (the monthly FRED
    differential forward-filled, divided by 252), compounded to a level (base ~1).
    """
    local_ret = load_ief_usd_levels(ief_path).pct_change()
    carry_daily = (
        eur_minus_usd_carry(usd_path, eur_path).reindex(local_ret.index, method="ffill") / 252.0
    )
    hedged_ret = (local_ret + carry_daily).dropna()
    return (1.0 + hedged_ret).cumprod().rename("ustsy_hedged")


def load_trend_eur_daily_levels(
    cta_path: Path = SG_CTA_RAW, eurusd_path: Path = EURUSD_RAW
) -> pd.Series:
    """Daily trend levels in EUR: daily SG CTA USD level / daily EURUSD.

    Division aligns on the two series' common daily dates (both business-day, with
    slightly different holiday calendars).
    """
    usd = load_sg_cta_usd_levels(cta_path)
    eurusd = load_eurusd(eurusd_path)
    return (usd / eurusd).dropna().rename("trend_eur")


def daily_returns_from_levels(levels: pd.Series) -> pd.Series:
    """Daily simple returns from a daily level series (no gap forward-filling)."""
    return levels.pct_change(fill_method=None).dropna().rename(f"{levels.name}_ret")


def daily_returns_matrix() -> pd.DataFrame:
    """The four sleeve daily returns aligned on their common trading days.

    Equity/gold/trend reach back to 2000-2001; the hedged US Tsy bond sleeve
    (IEF + carry) starts 2002-07, which sets the common daily window.
    """
    series = {
        "equity": daily_returns_from_levels(load_msci_acwi_eur_levels()),
        "bonds": daily_returns_from_levels(load_ustsy_hedged_daily_levels()),
        "trend": daily_returns_from_levels(load_trend_eur_daily_levels()),
        "gold": daily_returns_from_levels(load_gold_eur_levels()),
    }
    return pd.concat(series, axis=1, join="inner").sort_index()
