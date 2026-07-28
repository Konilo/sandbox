"""Regime inspection (pass 2).

The full-sample correlations are an average that hides state-dependent
diversification. Here we test the actual thesis -- that trend and gold
decorrelate from equities *when equities fall* -- two ways:

- ``conditional_correlations``: bucket the months by equity state and recompute
  each sleeve's correlation with equity per bucket (uses all data, no dates).
- ``rolling_correlations``: a 36-month rolling correlation with equity over
  time, to see the 2022 stock-bond flip and the crisis dips continuously.
- ``rolling_volatility``: each sleeve's 36-month rolling annualized volatility,
  to see that the single-number vols hide large regime shifts (e.g. the bond
  vol regime change post-2020).
"""

from __future__ import annotations

import pandas as pd
from data import monthly_returns_matrix

ROLLING_WINDOW = 36


def equity_state_masks(returns: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean month-masks for equity-state buckets."""
    equity = returns["equity"]
    worst_quartile = equity <= equity.quantile(0.25)
    return {
        "full sample": pd.Series(True, index=returns.index),
        "equity up": equity >= 0,
        "equity down": equity < 0,
        "equity worst quartile": worst_quartile,
    }


def conditional_correlations(
    returns: pd.DataFrame | None = None, anchor: str = "equity"
) -> pd.DataFrame:
    """Correlation of each other sleeve with ``anchor``, per equity-state bucket.

    Rows are buckets (with their month count ``n``); columns are the other sleeves.
    """
    returns = monthly_returns_matrix() if returns is None else returns
    others = [c for c in returns.columns if c != anchor]
    rows = {}
    for name, mask in equity_state_masks(returns).items():
        sub = returns.loc[mask]
        row = {"# months": int(mask.sum())}
        row.update({other: sub[anchor].corr(sub[other]) for other in others})
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def rolling_correlations(
    returns: pd.DataFrame | None = None,
    anchor: str = "equity",
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Rolling ``window``-month correlation of each other sleeve with ``anchor``."""
    returns = monthly_returns_matrix() if returns is None else returns
    others = [c for c in returns.columns if c != anchor]
    rolling = {other: returns[anchor].rolling(window).corr(returns[other]) for other in others}
    return pd.DataFrame(rolling).dropna()


def rolling_volatility(
    returns: pd.DataFrame | None = None, window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """Each sleeve's rolling ``window``-month annualized volatility."""
    returns = monthly_returns_matrix() if returns is None else returns
    return (returns.rolling(window).std() * 12**0.5).dropna()
