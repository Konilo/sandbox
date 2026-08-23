"""Leverage (pass 5): scale the risky portfolio to the CRRA-optimal risk.

With the box-spread borrowing rate ``BOX_REAL`` (``mu.BOX_REAL``) and a chosen
risky portfolio of *excess* Sharpe ``S = (mu_p - BOX_REAL) / sigma`` and
volatility ``sigma``, a CRRA investor with risk aversion ``gamma`` holds a
fraction

    k* = S / (gamma * sigma)

of wealth in the risky portfolio (borrowing the rest at ``BOX_REAL``). That
implies a target volatility ``sigma* = k* * sigma = S / gamma`` and an expected
real return ``BOX_REAL + S**2 / gamma``. ``k* > 1`` is gross leverage, financed
by short box spreads at ~``BOX_REAL``. The tangency itself is struck at the
risk-free ``RF_REAL`` (composition); the leverage excess is measured over the
higher ``BOX_REAL``.

These are the *unconstrained* optima; a separate, practical cap on achievable
leverage (portfolio margin, box-spread capacity) is applied downstream.
"""

from __future__ import annotations

import pandas as pd

from mu import BOX_REAL

GAMMA = 2.0


def optimal_leverage(sharpe: float, sigma: float, gamma: float = GAMMA) -> float:
    """CRRA-optimal gross exposure ``k* = S / (gamma * sigma)``.

    ``sharpe`` is the *excess* Sharpe over the real financing rate ``RF_REAL``.
    """
    return sharpe / (gamma * sigma)


def target_volatility(sharpe: float, gamma: float = GAMMA) -> float:
    """CRRA-optimal portfolio volatility ``sigma* = S / gamma``."""
    return sharpe / gamma


def levered_expected_return(
    sharpe: float, gamma: float = GAMMA, rf: float = BOX_REAL
) -> float:
    """Expected real return of the levered book, ``rf + S**2 / gamma``.

    ``sharpe`` is the *excess* Sharpe over ``rf``; ``rf`` defaults to the box
    borrowing rate ``BOX_REAL``.
    """
    return rf + sharpe**2 / gamma


def levered_allocation(weights: pd.Series, k: float) -> pd.Series:
    """Risky weights scaled to gross exposure ``k`` (as fractions of NAV).

    The financing leg is ``1 - k`` (negative = borrowed via box spreads) and is
    appended as a ``borrowed cash`` row.
    """
    scaled = weights * k
    scaled["borrowed cash"] = 1.0 - k
    return scaled.rename("% NAV")
