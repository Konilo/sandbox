"""Covariance construction for the portfolio study.

The Ledoit-Wolf shrinkage of the sample covariance of the four monthly EUR
sleeve returns, annualized (``frequency=12``), with a ``constant_correlation``
target. ``final_covariance`` then optionally overrides the equity-bonds
correlation (the study sets it to 0; rationale in the qmd). Correlations and
volatilities are derived views on the matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from data import monthly_returns_matrix
from pypfopt import risk_models

FREQUENCY = 12
SHRINKAGE_TARGET = "constant_correlation"


def ledoit_wolf_covariance(
    returns: pd.DataFrame | None = None, target: str = SHRINKAGE_TARGET
) -> tuple[pd.DataFrame, float]:
    """annualized Ledoit-Wolf-shrunk covariance and the shrinkage intensity delta."""
    returns = monthly_returns_matrix() if returns is None else returns
    estimator = risk_models.CovarianceShrinkage(returns, returns_data=True, frequency=FREQUENCY)
    cov = estimator.ledoit_wolf(shrinkage_target=target)
    return cov, estimator.delta


def annualized_vols(cov: pd.DataFrame) -> pd.Series:
    """annualized volatilities (square root of the covariance diagonal)."""
    return pd.Series(np.sqrt(np.diag(cov)), index=cov.index, name="ann_vol")


def correlation(cov: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix implied by a covariance matrix."""
    return risk_models.cov_to_corr(cov)


def apply_correlation_overrides(
    cov: pd.DataFrame, overrides: dict[tuple[str, str], float]
) -> pd.DataFrame:
    """Covariance with given pairwise correlations overridden; volatilities kept.

    ``overrides`` maps ``(asset_a, asset_b)`` to a correlation. The matrix is
    rebuilt as ``diag(vol) @ corr @ diag(vol)`` and checked to stay positive
    semi-definite (an override can otherwise break PSD and the optimiser).
    """
    corr = correlation(cov)
    vols = annualized_vols(cov)
    for (asset_a, asset_b), rho in overrides.items():
        corr.loc[asset_a, asset_b] = rho
        corr.loc[asset_b, asset_a] = rho
    cov_new = corr.mul(vols, axis=0).mul(vols, axis=1)
    min_eigenvalue = np.linalg.eigvalsh(cov_new)[0]
    if min_eigenvalue < 0:
        raise ValueError(
            f"overrides produced a non-PSD matrix (min eigenvalue {min_eigenvalue:.2e})"
        )
    return cov_new


def final_covariance(
    returns: pd.DataFrame | None = None,
    rho_equity_bonds: float | None = None,
) -> pd.DataFrame:
    """Shrunk covariance, optionally overriding the equity-bonds correlation.

    With ``rho_equity_bonds`` ``None`` this returns the Ledoit-Wolf-shrunk
    covariance unchanged; otherwise that one correlation pair is overridden.
    """
    cov, _ = ledoit_wolf_covariance(returns)
    if rho_equity_bonds is not None:
        cov = apply_correlation_overrides(cov, {("equity", "bonds"): rho_equity_bonds})
    return cov
