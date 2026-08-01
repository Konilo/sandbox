"""Black-Litterman engine: equilibrium prior + views -> posterior expected returns.

Replaces the old hand-set honest-mu / ERC-blend allocation. The humility now
lives in BL's shrinkage of the views toward the market-implied equilibrium
prior; the risky portfolio is the max-Sharpe (tangency) of the posterior mu on
the study covariance (composition.tangency_weights), then levered in leverage.py.

All figures real, EUR-consistent (the caps only supply dimensionless market
weights; the prior is struck at mu.RF_REAL so the posterior is a real return).

Decisions (each set with Konilo; see the "Expected returns" section):

Market caps (USD tn, 2026; currency is immaterial -- only the ratios enter):
  equity  101.47  MSCI ACWI free-float market cap (factsheet, 2026-06-30)
  bonds    30.54  FTSE WGBI-Developed index *market value* (factsheet, 2026-06-30)
  gold     15.40  WGC "financial gold" (bars/coins/ETF + official), 2025-12-31
  trend     0.00  managed futures is zero-net-supply -> no market cap; its prior
                  falls out of the reverse-opt as beta_MF x premium (~0), and its
                  return enters entirely as a view (He & Litterman 1999).

delta (market risk aversion) is DERIVED, not defaulted: delta = market premium /
market variance. The market premium is an INDEPENDENT external ERP (not Konilo's
own views -- that would be circular), triangulated arithmetic / over-bills /
world: DMS-2026 forward ~4.5-5%, Damodaran implied 4.23%, Kroll 5.0%,
Fernandez-2026 survey 5.5% -> ~5.0%. With the small bond/gold premia below this
gives delta ~= 3.6.

Views (absolute, real) are Konilo's honest from-source estimates (mu.VIEWS_REAL):
equity 4.03% (Elm P-CAEY), bonds 1.20% (WGBI-DM yield), trend 1.80% (SG CTA).
Gold gets NO view (no forward anchor) -> it stays at its equilibrium prior.
Per-view Idzorek confidences reflect each view's reliability vs the prior:
  equity 0.8  P-CAEY forward/valuation-aware; the prior is valuation-blind and
              higher, so trust the view (Cayas corroborates a low equity ~3.5%).
  bonds  0.7  the starting yield is a strong forward predictor; the CAPM prior is
              a weak model for bonds; Cayas 1.3% ~= the view, not the prior.
  trend  0.6  noisier estimate (backward mean), but momentum evidence + Cayas
              (2.7%, more bullish) support keeping the sleeve meaningful.
"""

from __future__ import annotations

import pandas as pd
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel

from mu import RF_REAL, VIEWS_REAL

MARKET_CAPS = {"equity": 101.47, "bonds": 30.54, "gold": 15.40}  # USD tn; trend has none

EQUITY_ERP = 0.050  # external, arithmetic, over bills, world (see docstring)
BOND_PREMIUM = 0.010  # developed-govt term premium over bills (small)
GOLD_PREMIUM = 0.0  # gold carries ~no risk premium over real bills

VIEW_ASSETS = ("equity", "bonds", "trend")  # gold intentionally has no view
VIEW_CONFIDENCES = {"equity": 0.8, "bonds": 0.7, "trend": 0.6}  # Idzorek
TAU = 0.05


def market_weights(index: pd.Index) -> pd.Series:
    """Dimensionless market-cap weights over ``index`` (assets without a cap -> 0)."""
    caps = pd.Series(MARKET_CAPS).reindex(index).fillna(0.0)
    return caps / caps.sum()


def market_delta(cov: pd.DataFrame) -> float:
    """Reverse-opt risk aversion delta = external market premium / market variance.

    The market premium is the cap-weighted blend of the *external* per-asset
    premia (equity ERP dominates), NOT the views -- keeping delta an independent
    statement about the market rather than a mirror of Konilo's own returns.
    """
    weights = market_weights(cov.index)
    premia = pd.Series(
        {"equity": EQUITY_ERP, "bonds": BOND_PREMIUM, "trend": 0.0, "gold": GOLD_PREMIUM}
    ).reindex(cov.index)
    market_premium = float(weights @ premia)
    market_variance = float(weights @ cov.values @ weights)
    return market_premium / market_variance


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
    """Black-Litterman posterior real expected returns (the MVO input)."""
    return bl_model(cov).bl_returns().reindex(cov.index).rename("mu")
