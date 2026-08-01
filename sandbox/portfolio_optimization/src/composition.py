"""Portfolio composition: the unlevered risky-portfolio weights.

With a real risk-free asset (borrow at ~rf via short box spreads) and leverage,
two-fund separation says every investor holds the *same* risky portfolio -- the
max-Sharpe tangency of the Black-Litterman posterior (``black_litterman.py``) --
scaled by risk appetite (that scaling is the leverage step, ``leverage.py``).
Black-Litterman supplies the humility (its posterior is already shrunk toward the
market prior), so the tangency is taken directly, with no further shrinkage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier

from covariance import final_covariance
from mu import RF_REAL, expected_returns


def tangency_weights(
    cov: pd.DataFrame | None = None,
    mu: pd.Series | None = None,
    rf: float = RF_REAL,
) -> pd.Series:
    """Long-only max-Sharpe (tangency) weights at real financing rate ``rf``.

    ``rf`` is the real risk-free/financing anchor (``mu.RF_REAL``); the tangency
    maximises ``(mu - rf) / sigma``, so a positive ``rf`` tilts it away from any
    sleeve whose real return sits below the financing cost.
    """
    cov = final_covariance() if cov is None else cov
    mu = expected_returns(cov) if mu is None else mu
    ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef.max_sharpe(risk_free_rate=rf)
    return pd.Series(ef.clean_weights(), name="tangency").reindex(cov.index)


def portfolio_performance(
    weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float = RF_REAL
) -> tuple[float, float, float]:
    """Real expected return, volatility, and *excess* Sharpe over ``rf``.

    The returned Sharpe is ``(ret - rf) / vol`` -- the quantity the leverage step
    consumes -- so it is consistent with the tangency, which is also struck at
    ``rf``. With ``rf = 0`` this reduces to the plain ``ret / vol``.
    """
    w = weights.reindex(cov.index).values
    ret = float(mu.reindex(cov.index).values @ w)
    vol = float(np.sqrt(w @ cov.values @ w))
    return ret, vol, (ret - rf) / vol
