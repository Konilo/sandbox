"""Portfolio composition (pass 4): the unlevered risky-portfolio weights.

With a real risk-free asset (borrow at ~rf via short box spreads) and leverage,
two-fund separation says every investor holds the *same* risky portfolio -- the
max-Sharpe tangency -- scaled by risk appetite (that scaling is the leverage
step, ``leverage.py``). So the composition target is the tangency.

But the tangency sits on a near-flat stretch of the frontier and piles into the
lowest-vol sleeve, so its exact weights are ill-determined and fragile. I temper
it by blending toward an equal-risk-contribution (ERC) anchor:

    w(lambda) = (1 - lambda) * tangency + lambda * ERC

``lambda`` is a swept robustness dial. ERC is computed with a small self-contained
convex solve; ``erc_weights_riskfolio`` reproduces it with riskfolio-lib as an
on-demand cross-check (kept off the default path -- importing riskfolio costs
~70s -- but it validated ``erc_weights`` to <0.001pp).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from scipy.optimize import minimize

from covariance import final_covariance
from mu import RF_REAL, expected_returns

BLEND_LAMBDA = 0.5


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


def erc_weights(cov: pd.DataFrame | None = None) -> pd.Series:
    """Equal-risk-contribution weights via the convex log-barrier formulation.

    Minimise ``0.5 w'Sw - (1/n) sum ln(w_i)`` for ``w > 0``, then normalise to
    sum 1; the stationarity condition forces every risk contribution
    ``w_i (S w)_i`` equal. Validated identical to riskfolio-lib MV risk parity
    to <0.001pp (see ``erc_weights_riskfolio``).
    """
    cov = final_covariance() if cov is None else cov
    S = cov.values
    n = len(S)
    res = minimize(
        lambda w: 0.5 * w @ S @ w - np.log(w).sum() / n,
        np.ones(n) / n,
        jac=lambda w: S @ w - (1.0 / n) / w,
        method="L-BFGS-B",
        bounds=[(1e-8, None)] * n,
        options={"ftol": 1e-16, "gtol": 1e-12},
    )
    return pd.Series(res.x / res.x.sum(), index=cov.index, name="erc")


def erc_weights_riskfolio(
    cov: pd.DataFrame | None = None, returns: pd.DataFrame | None = None
) -> pd.Series:
    """ERC via riskfolio-lib (MV risk parity): on-demand cross-check of ``erc_weights``.

    riskfolio is imported lazily (its import costs ~70s) so it never slows the
    default path. Uses our covariance, not riskfolio's own estimate.
    """
    import riskfolio as rp

    from data import monthly_returns_matrix

    cov = final_covariance() if cov is None else cov
    returns = monthly_returns_matrix() if returns is None else returns
    order = list(cov.index)
    port = rp.Portfolio(returns=returns[order])
    port.assets_stats(method_mu="hist", method_cov="hist")
    port.cov = pd.DataFrame(cov.values, index=order, columns=order)
    weights = port.rp_optimization(model="Classic", rm="MV", rf=0, b=None, hist=True)
    return weights.reindex(order)["weights"].rename("erc_riskfolio")


def blend(tangency: pd.Series, erc: pd.Series, lam: float = BLEND_LAMBDA) -> pd.Series:
    """The composition dial: ``(1 - lam) * tangency + lam * ERC``."""
    return ((1 - lam) * tangency + lam * erc).rename("blend")


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Fractional risk contribution of each sleeve, ``w_i (S w)_i / (w'Sw)``."""
    w = weights.reindex(cov.index).values
    contrib = w * (cov.values @ w)
    return pd.Series(contrib / contrib.sum(), index=cov.index, name="risk_contribution")


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
