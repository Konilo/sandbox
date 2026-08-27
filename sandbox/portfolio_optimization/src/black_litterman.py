"""Black-Litterman engine: equilibrium prior + views -> posterior expected returns.

The posterior mu, net of vehicle fees, feeds ``composition.tangency_weights``.
All figures real and EUR-consistent: the caps only supply dimensionless market
weights, and the prior is struck at ``mu.RF_REAL`` so the posterior is a real
return. See the "Expected
returns" section of the qmd for the choices behind the constants below.
"""

from __future__ import annotations

import pandas as pd
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel

from mu import RF_REAL, TERS, VIEWS_REAL

# Market caps (USD tn, 2026; only the ratios enter). Trend (managed futures) is
# zero-net-supply, so it has no cap: its prior falls out of the reverse-opt as
# beta_MF x premium (~0) and it enters entirely as a view (He & Litterman 1999).
MARKET_CAPS = {
    "equity": 101.47,  # MSCI ACWI free-float market cap (factsheet, 2026-06-30)
    "bonds": 21.66,  # US Treasury notes + bonds outstanding (FiscalData MSPD, 2026-07-31)
    "gold": 15.40,  # WGC "financial gold": bars/coins/ETF + official (2025-12-31)
}

# External forward equity risk premium (arithmetic, over bills, world), used to
# derive delta. Centre of the forward cluster ~5%: Damodaran 4.23%, Kroll 5.0%,
# Fernandez-2026 5.5%; DMS-2026 anchors the magnitude (US realized 6.6% vs 0.5%).
EQUITY_ERP = 0.050

VIEW_ASSETS = ("equity", "bonds", "trend")  # gold has no forward anchor -> no view
VIEW_CONFIDENCES = {"equity": 0.8, "bonds": 0.7, "trend": 0.65}  # Idzorek; per-view reliability
TAU = 0.05


def market_weights(index: pd.Index) -> pd.Series:
    """Dimensionless market-cap weights over ``index`` (assets without a cap -> 0)."""
    caps = pd.Series(MARKET_CAPS).reindex(index).fillna(0.0)
    return caps / caps.sum()


def market_delta(cov: pd.DataFrame) -> float:
    """Reverse-opt risk aversion ``delta = EQUITY_ERP / cov(equity, market)``.

    Anchoring delta to equity makes the market-implied equity excess return equal
    ``EQUITY_ERP``; every other sleeve's prior then follows from its covariance
    with the market, with no separately assumed premium.
    """
    weights = market_weights(cov.index)
    cov_equity_market = float((cov.values @ weights.values)[cov.index.get_loc("equity")])
    return EQUITY_ERP / cov_equity_market


def equilibrium_prior(cov: pd.DataFrame) -> pd.Series:
    """Market-implied (reverse-optimised) prior returns Pi = delta * Sigma * w_mkt.

    Struck at ``RF_REAL`` so Pi is a real *total* return, consistent with the
    views and the leverage step.
    """
    caps = pd.Series(MARKET_CAPS).reindex(cov.index).fillna(0.0)
    return black_litterman.market_implied_prior_returns(
        caps, market_delta(cov), cov, risk_free_rate=RF_REAL
    )


def bl_model(cov: pd.DataFrame) -> BlackLittermanModel:
    """The fitted Black-Litterman model (Idzorek confidences on the three views)."""
    views = {asset: VIEWS_REAL[asset] for asset in VIEW_ASSETS}
    confidences = [VIEW_CONFIDENCES[asset] for asset in views]
    return BlackLittermanModel(
        cov,
        pi=equilibrium_prior(cov),
        absolute_views=views,
        omega="idzorek",
        view_confidences=confidences,
        tau=TAU,
    )


def posterior_returns(cov: pd.DataFrame) -> pd.Series:
    """Black-Litterman posterior real expected returns, gross of vehicle fees."""
    return bl_model(cov).bl_returns().reindex(cov.index).rename("mu")


def vehicle_fees(index: pd.Index) -> pd.Series:
    """Ongoing charge of each sleeve's vehicle, ordered to match ``index``."""
    return pd.Series(TERS).reindex(index).fillna(0.0).rename("ter")


def posterior_net_returns(cov: pd.DataFrame) -> pd.Series:
    """Posterior returns net of vehicle fees -- the MVO input."""
    return (posterior_returns(cov) - vehicle_fees(cov.index)).rename("mu")
