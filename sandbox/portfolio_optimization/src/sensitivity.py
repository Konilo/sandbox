"""Sensitivity & reliability sweeps (see draft/SENSITIVITY.md).

Each pipeline stage -- MVO inputs -> tangency (B), blend by lambda (C), lever by
gamma (D) -- has one knob, and each sweep probes the robustness of the stage its
knob lives in. The through-line metric is always the final *levered book*, so
results are comparable across stages.

The baseline is the study's live configuration: the from-source mu
(``mu.VIEWS_REAL``), the final covariance (``covariance.final_covariance`` with
rho_eq-bonds 0.25 and sigma_bonds 5%), lambda 0.5, gamma 2, and the real
financing rate ``mu.RF_REAL``. ``run_cell()`` with no arguments reproduces the
qmd's levered book exactly.
"""

from __future__ import annotations

# Column/index labels use Greek finance notation (mu, sigma, rho, lambda) so the
# rendered tables match the rest of the study; silence the ambiguous-unicode rule.
# ruff: noqa: RUF001

import numpy as np
import pandas as pd

from composition import (
    BLEND_LAMBDA,
    blend,
    erc_weights,
    portfolio_performance,
    tangency_weights,
)
from covariance import (
    BOND_VOL_OVERRIDE,
    STOCK_BOND_OVERRIDE,
    apply_correlation_overrides,
    apply_volatility_overrides,
    final_covariance,
)
from leverage import (
    GAMMA,
    levered_expected_return,
    optimal_leverage,
    target_volatility,
)
from mu import VIEWS_REAL, expected_returns

SLEEVES = ["equity", "bonds", "trend", "gold"]
NAV_COLS = [f"{s} %NAV" for s in SLEEVES]
BOOK_COLS = [*NAV_COLS, "k*", "σ*", "E[r]", "Sharpe"]


def realized_equity_vol() -> float:
    """Equity's annualized volatility in the baseline covariance (the 13.9% level)."""
    return float(np.sqrt(final_covariance().loc["equity", "equity"]))


def build_inputs(
    equity_mu: float = VIEWS_REAL["equity"],
    sigma_equity: float | None = None,
    rho_equity_bonds: float = STOCK_BOND_OVERRIDE,
    sigma_bonds: float = BOND_VOL_OVERRIDE,
    extra_mu: dict[str, float] | None = None,
    extra_vol: dict[str, float] | None = None,
    extra_corr: dict[tuple[str, str], float] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """(mu, cov) for one input cell; every default reproduces the live study.

    ``sigma_equity=None`` keeps the realized equity vol (overriding it to that
    same value is a no-op, so the default cell is exact). ``extra_*`` carry the
    frozen-input perturbations (gold correlations, trend mu/vol) used to confirm
    the held inputs are inert.
    """
    cov = final_covariance(rho_equity_bonds=rho_equity_bonds, sigma_bonds=sigma_bonds)
    vol_overrides = dict(extra_vol or {})
    if sigma_equity is not None:
        vol_overrides["equity"] = sigma_equity
    if vol_overrides:
        cov = apply_volatility_overrides(cov, vol_overrides)
    if extra_corr:
        cov = apply_correlation_overrides(cov, extra_corr)
    mu = expected_returns(cov).copy()
    mu["equity"] = equity_mu
    for asset, value in (extra_mu or {}).items():
        mu[asset] = value
    return mu, cov


def levered_book(
    mu: pd.Series, cov: pd.DataFrame, lam: float = BLEND_LAMBDA, gamma: float = GAMMA
) -> dict[str, float]:
    """The final levered book for a (mu, cov): sleeve %NAV, k*, sigma*, E[r], Sharpe.

    Runs the full composition -> leverage path: tangency (at RF_REAL) blended
    with ERC by ``lam``, scaled to the CRRA-optimal exposure ``k*``. Sleeve %NAV
    is ``k* * weight``; the implied financing leg is ``1 - k*``.
    """
    weights = blend(tangency_weights(cov, mu), erc_weights(cov), lam)
    _, vol, sharpe = portfolio_performance(weights, mu, cov)
    k = optimal_leverage(sharpe, vol, gamma)
    book = {f"{s} %NAV": weights[s] * k for s in SLEEVES}
    book["k*"] = k
    book["σ*"] = target_volatility(sharpe, gamma)
    book["E[r]"] = levered_expected_return(sharpe, gamma)
    book["Sharpe"] = sharpe
    return book


def run_cell(
    equity_mu: float = VIEWS_REAL["equity"],
    sigma_equity: float | None = None,
    rho_equity_bonds: float = STOCK_BOND_OVERRIDE,
    sigma_bonds: float = BOND_VOL_OVERRIDE,
    lam: float = BLEND_LAMBDA,
    gamma: float = GAMMA,
    **input_kwargs: dict,
) -> dict[str, float]:
    """Levered book for one fully-specified input cell (defaults = baseline)."""
    mu, cov = build_inputs(
        equity_mu=equity_mu,
        sigma_equity=sigma_equity,
        rho_equity_bonds=rho_equity_bonds,
        sigma_bonds=sigma_bonds,
        **input_kwargs,
    )
    return levered_book(mu, cov, lam, gamma)


def factorial(
    equity_mu_levels: tuple[float, ...],
    sigma_equity_levels: tuple[float | None, ...],
    rho_levels: tuple[float, ...],
    sigma_bonds_levels: tuple[float, ...] = (BOND_VOL_OVERRIDE,),
    lam: float = BLEND_LAMBDA,
    gamma: float = GAMMA,
) -> pd.DataFrame:
    """Full factorial over the uncertain MVO inputs; one row per cell.

    ``sigma_bonds`` is the fourth axis (its main effect is the largest and it
    interacts with rho on the bond sleeve, so it earns a factorial slot rather
    than a one-at-a-time sweep). ``None`` in ``sigma_equity_levels`` is the
    realized-vol baseline and is displayed as that value.
    """
    realized = realized_equity_vol()
    rows = []
    for equity_mu in equity_mu_levels:
        for sigma_equity in sigma_equity_levels:
            for rho in rho_levels:
                for sigma_bonds in sigma_bonds_levels:
                    book = run_cell(
                        equity_mu=equity_mu,
                        sigma_equity=sigma_equity,
                        rho_equity_bonds=rho,
                        sigma_bonds=sigma_bonds,
                        lam=lam,
                        gamma=gamma,
                    )
                    rows.append(
                        {
                            "equity μ": equity_mu,
                            "equity σ": realized if sigma_equity is None else sigma_equity,
                            "ρ eq-bonds": rho,
                            "σ_bonds": sigma_bonds,
                            **book,
                        }
                    )
    return pd.DataFrame(rows)


def decision_band(df: pd.DataFrame, cols: list[str] = BOOK_COLS) -> pd.DataFrame:
    """Min/baseline/max of each output across the cells of ``df``."""
    baseline = pd.Series(run_cell())
    return pd.DataFrame(
        {"min": df[cols].min(), "baseline": baseline[cols], "max": df[cols].max()}
    )


def main_effects(
    df: pd.DataFrame,
    axes: tuple[str, ...] = ("equity μ", "equity σ", "ρ eq-bonds", "σ_bonds"),
    metric: str = "k*",
) -> pd.DataFrame:
    """Mean ``metric`` at each axis's low/high level and the across-level range.

    Averaging the metric over the other axes isolates one axis's main effect; the
    ``range`` column ranks the axes by how much they move the metric on their own
    -- i.e. where the estimation-trust budget belongs. Sorted by ``range``.
    """
    rows = {}
    for axis in axes:
        means = df.groupby(axis)[metric].mean().sort_index()
        label = f"{axis} ({means.index[0]:.2%} → {means.index[-1]:.2%})"
        rows[label] = {
            f"{metric} low": means.iloc[0],
            f"{metric} high": means.iloc[-1],
            "range": means.max() - means.min(),
        }
    return pd.DataFrame(rows).T.sort_values("range", ascending=False)


def lambda_sweep(
    lambdas: tuple[float, ...], gamma: float = GAMMA
) -> pd.DataFrame:
    """Composition and levered book across the blend dial ``lambda`` (baseline inputs).

    Reports the unlevered blend weights (which is where lambda acts) alongside
    the levered book, so the Sharpe give-up and the weight stabilization are both
    visible.
    """
    mu, cov = build_inputs()
    erc = erc_weights(cov)
    tan = tangency_weights(cov, mu)
    rows = []
    for lam in lambdas:
        weights = blend(tan, erc, lam)
        _, vol, sharpe = portfolio_performance(weights, mu, cov)
        k = optimal_leverage(sharpe, vol, gamma)
        rows.append(
            {
                "λ": lam,
                **{f"{s} weight": weights[s] for s in SLEEVES},
                "Sharpe": sharpe,
                "k*": k,
            }
        )
    return pd.DataFrame(rows)


def weight_stability_by_lambda(
    equity_mu_levels: tuple[float, ...],
    lambdas: tuple[float, ...] = (0.0, 0.5, 1.0),
    sleeve: str = "bonds",
) -> pd.DataFrame:
    """mu-driven range of the *unlevered* blend weight of ``sleeve`` vs lambda.

    This is the shape (composition) counterpart to ``band_by_lambda``'s scale
    (k*) view: it collapses to zero at lambda=1 because ERC weights are mu-free,
    confirming that lambda stabilizes the *weights* against mu even where it does
    not stabilize the levered k*.
    """
    _, cov = build_inputs()
    erc = erc_weights(cov)
    rows = []
    for lam in lambdas:
        weights = []
        for equity_mu in equity_mu_levels:
            mu = expected_returns(cov).copy()
            mu["equity"] = equity_mu
            weights.append(blend(tangency_weights(cov, mu), erc, lam)[sleeve])
        rows.append({"λ": lam, f"{sleeve} weight μ-range": max(weights) - min(weights)})
    return pd.DataFrame(rows)


def band_by_lambda(
    equity_mu_levels: tuple[float, ...],
    sigma_equity_levels: tuple[float | None, ...],
    rho_levels: tuple[float, ...],
    lambdas: tuple[float, ...] = (0.0, 0.5, 1.0),
    metric: str = "k*",
) -> pd.DataFrame:
    """``metric`` band width from the B-grid, split into mu-driven and Sigma-driven, per lambda.

    mu-driven = spread of ``metric`` as equity mu moves over its levels with
    Sigma at baseline; Sigma-driven = spread as (equity sigma, rho) move with mu
    at baseline. Shows lambda collapsing the mu dimension (ERC ignores mu in the
    weights) while the Sigma dimension persists -- and that a residual mu spread
    survives because leverage still scales with the mu-dependent Sharpe.
    """
    mu_base, sig_base, rho_base = (
        VIEWS_REAL["equity"],
        None,
        STOCK_BOND_OVERRIDE,
    )
    rows = []
    for lam in lambdas:
        mu_driven = [
            run_cell(equity_mu=m, sigma_equity=sig_base, rho_equity_bonds=rho_base, lam=lam)[metric]
            for m in equity_mu_levels
        ]
        sigma_driven = [
            run_cell(equity_mu=mu_base, sigma_equity=s, rho_equity_bonds=r, lam=lam)[metric]
            for s in sigma_equity_levels
            for r in rho_levels
        ]
        rows.append(
            {
                "λ": lam,
                "μ-driven range": max(mu_driven) - min(mu_driven),
                "Σ-driven range": max(sigma_driven) - min(sigma_driven),
            }
        )
    return pd.DataFrame(rows)
