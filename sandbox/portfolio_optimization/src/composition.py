"""Portfolio composition: the unlevered risky-portfolio weights.

The risky book is the long-only max-Sharpe tangency of the Black-Litterman
posterior (``black_litterman.py``), taken directly with no further shrinkage
(the BL posterior is already shrunk toward the market prior). ``leverage.py``
then scales it to risk appetite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier

from covariance import final_covariance
from leverage import GAMMA
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


def _weight_lattice(units: int, n_assets: int) -> np.ndarray:
    """Every non-negative integer vector of length ``n_assets`` summing to ``units``."""
    if n_assets == 1:
        return np.array([[units]])
    blocks = []
    for first in range(units + 1):
        rest = _weight_lattice(units - first, n_assets - 1)
        blocks.append(np.hstack([np.full((len(rest), 1), first), rest]))
    return np.vstack(blocks)


def optimal_weight_ranges(
    cov: pd.DataFrame,
    mu: pd.Series,
    rf: float = RF_REAL,
    gamma: float = GAMMA,
    tolerance_bp: float = 10.0,
    step: float = 0.01,
) -> pd.DataFrame:
    """Per sleeve, the weight range costing at most ``tolerance_bp`` of levered return.

    A CRRA investor levering to ``k* = S / (gamma * sigma)`` earns
    ``rf + S**2 / gamma``, so a Sharpe shortfall maps to a return shortfall in bp
    and volatility drops out. Every long-only weight vector on the ``step``
    lattice is enumerated and the best Sharpe at each level of each sleeve is
    kept, which makes the result exact on that lattice and optimiser-free.
    """
    units = round(1 / step)
    lattice = _weight_lattice(units, len(cov.index))
    weights = lattice / units
    variance = np.einsum("ij,jk,ik->i", weights, cov.values, weights)
    sharpe = np.where(variance > 0, (weights @ mu.values - rf) / np.sqrt(variance), -np.inf)
    loss_bp = (sharpe.max() ** 2 - sharpe**2) / gamma * 1e4

    rows = {}
    for i, sleeve in enumerate(cov.index):
        within = np.full(units + 1, False)
        for level in range(units + 1):
            at_level = loss_bp[lattice[:, i] == level]
            within[level] = at_level.size > 0 and at_level.min() <= tolerance_bp
        peak = lattice[np.argmax(sharpe), i]
        low = high = peak
        while low > 0 and within[low - 1]:
            low -= 1
        while high < units and within[high + 1]:
            high += 1
        rows[sleeve] = {"low": low * step, "high": high * step}
    return pd.DataFrame(rows).T
