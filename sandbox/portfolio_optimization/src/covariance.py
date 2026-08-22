"""Covariance construction for the portfolio study.

Pass 1 (historical base): the sample covariance of the four monthly EUR sleeve
returns, and its Ledoit-Wolf shrinkage. All matrices are annualized
(``frequency=12``). Correlations and volatilities are derived views on those.

The shrinkage target is ``constant_correlation`` (every pair shrunk toward the
average sample correlation): for four cross-asset sleeves it is a more
defensible prior than the ``single_factor`` market model (ill-defined for gold
and managed futures) or the ``constant_variance`` scaled identity.

Pass 3 (no overrides): the shrunk covariance is used as-is. The equity-bonds
correlation the earlier (unhedged) design distrusted is, for the EUR-hedged US
Treasury sleeve, a structural flight-to-quality decorrelation and the very
reason the sleeve was chosen; overriding it away would be inconsistent, so it is
left at its Ledoit-Wolf-shrunk level, which already tempers the raw estimate. Its
one failure mode (inflation shocks, e.g. 2022) is covered by the gold and trend
sleeves, not by distorting this input. ``final_covariance`` still exposes optional
``rho_equity_bonds`` / ``sigma_bonds`` overrides for sweeps, but both default off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from data import monthly_returns_matrix
from pypfopt import risk_models

FREQUENCY = 12
SHRINKAGE_TARGET = "constant_correlation"


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
    rho_equity_bonds: float | None = None,
    sigma_bonds: float | None = None,
) -> pd.DataFrame:
    """Shrunk covariance, optionally with targeted overrides (both off by default).

    With both arguments ``None`` this returns the Ledoit-Wolf-shrunk covariance
    unchanged. ``rho_equity_bonds`` overrides just that correlation pair;
    ``sigma_bonds`` overrides the bond volatility. Kept for sensitivity sweeps.
    """
    cov, _ = ledoit_wolf_covariance(returns)
    if rho_equity_bonds is not None:
        cov = apply_correlation_overrides(cov, {("equity", "bonds"): rho_equity_bonds})
    if sigma_bonds is not None:
        cov = apply_volatility_overrides(cov, {"bonds": sigma_bonds})
    return cov
