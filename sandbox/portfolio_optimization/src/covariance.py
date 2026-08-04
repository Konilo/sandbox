"""Covariance construction for the portfolio study.

Pass 1 (historical base): the sample covariance of the four monthly EUR sleeve
returns, and its Ledoit-Wolf shrinkage. All matrices are annualized
(``frequency=12``). Correlations and volatilities are derived views on those.

The shrinkage target is ``constant_correlation`` (every pair shrunk toward the
average sample correlation): for four cross-asset sleeves it is a more
defensible prior than the ``single_factor`` market model (ill-defined for gold
and managed futures) or the ``constant_variance`` scaled identity.

Pass 3 (cautious overrides): the regime inspection shows the equity-bonds
correlation is unstable and currently positive, so ``final_covariance`` overrides
just that pair to a conservative value (default 0.25), leaving the other five
correlations at their shrunk levels. It also overrides the bond volatility: the
full-sample 3.8% leans on the structurally calm ZIRP/QE decades, so it is raised
to a forward-realistic ~5% (the ETF's 3y factsheet std is 4.48%, recent realized
~5%). Both overrides are targeted and meant to be swept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from data import monthly_returns_matrix
from pypfopt import risk_models

FREQUENCY = 12
SHRINKAGE_TARGET = "constant_correlation"
STOCK_BOND_OVERRIDE = 0.25
BOND_VOL_OVERRIDE = 0.043


def sample_covariance(returns: pd.DataFrame | None = None) -> pd.DataFrame:
    """annualized sample covariance of the four sleeve returns."""
    returns = monthly_returns_matrix() if returns is None else returns
    return risk_models.sample_cov(returns, returns_data=True, frequency=FREQUENCY)


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


def apply_volatility_overrides(cov: pd.DataFrame, overrides: dict[str, float]) -> pd.DataFrame:
    """Covariance with given sleeve volatilities overridden; correlations kept.

    ``overrides`` maps a sleeve to an annualized volatility. The matrix is
    rebuilt as ``diag(vol) @ corr @ diag(vol)`` with the new vols; since a
    positive diagonal rescaling of a PSD correlation matrix stays PSD, this
    cannot break PSD, but it is checked for symmetry with the correlation path.
    """
    corr = correlation(cov)
    vols = annualized_vols(cov)
    for asset, sigma in overrides.items():
        vols[asset] = sigma
    cov_new = corr.mul(vols, axis=0).mul(vols, axis=1)
    min_eigenvalue = np.linalg.eigvalsh(cov_new)[0]
    if min_eigenvalue < 0:
        raise ValueError(
            f"overrides produced a non-PSD matrix (min eigenvalue {min_eigenvalue:.2e})"
        )
    return cov_new


def final_covariance(
    returns: pd.DataFrame | None = None,
    rho_equity_bonds: float = STOCK_BOND_OVERRIDE,
    sigma_bonds: float | None = BOND_VOL_OVERRIDE,
) -> pd.DataFrame:
    """Shrunk covariance with the equity-bonds correlation and bond vol overridden.

    Both are swept variables. ``rho_equity_bonds`` overrides just that pair; the
    other five correlations stay at their Ledoit-Wolf-shrunk levels.
    ``sigma_bonds`` raises the bond volatility off its calm-era 3.8% to a
    forward-realistic level; pass ``None`` to keep the historical vol.
    """
    cov, _ = ledoit_wolf_covariance(returns)
    cov = apply_correlation_overrides(cov, {("equity", "bonds"): rho_equity_bonds})
    if sigma_bonds is not None:
        cov = apply_volatility_overrides(cov, {"bonds": sigma_bonds})
    return cov
