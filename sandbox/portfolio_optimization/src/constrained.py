"""Margin-aware leverage: cap the CRRA optimum by an explicit ruin budget, and
test whether a binding cap should shift the composition.

The CRRA leverage ``k* = S / (gamma * sigma)`` (see ``leverage.py``) maximises
mean-variance utility, which prices *drawdown* (through ``gamma``) but not the
absorbing event of a forced liquidation. This module adds that missing
constraint. The ruin-safe leverage ``k_margin`` is the largest ``k`` whose daily
stationary-bootstrap probability of at least one margin call -- at the UCITS
maintenance ratio, quarterly resets -- stays within a ruin budget. A composition
sweep then checks whether, once leverage is capped, a more-equity book than the
tangency raises utility (it does only when the cap binds hard, i.e. at low
``gamma``; see the study).

Everything reuses ``stress.py``: the bootstrap and the margin mechanics are the
same. The one addition is a closed form for the leverage at which a given path
breaches maintenance, which makes the ruin budget and the sweep cheap.

Critical-leverage identity
--------------------------
Within a rebalance segment, ``stress.simulate_path`` gives the margin ratio

    rho_t = 1 - (1 - 1/k) * (factor_t / g_t)

where ``g_t`` is the weighted cumulative sleeve growth since the reset and
``factor_t`` the financing accrual (the segment's ``NAV_start`` cancels in the
ratio). The worst ``rho`` over a whole path is therefore ``1 - (1 - 1/k) * X``
with ``X = max_t(factor_t / g_t)`` over every segment, so ``rho = m`` solves to

    k_crit = 1 / (1 - (1 - m) / X).

``X`` does not depend on ``k``, so one pass over the bootstrap paths yields the
critical leverage of every path; ``P(call at k) = mean(k_crit < k)`` and
``k_margin`` is the ``ruin_budget`` quantile of ``k_crit``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import stress
from mu import RF_REAL


def _resampled_array(
    returns_daily: pd.DataFrame, n_periods: int, block: int, n_paths: int, seed: int
) -> np.ndarray:
    """``n_paths`` stationary-bootstrap paths as a ``(n_paths, n_periods, n_sleeves)`` array.

    Generated exactly as ``stress.bootstrap_grid`` does (same rng, same resampler),
    so the call probabilities here line up with the stress section's.
    """
    rng = np.random.default_rng(seed)
    arr = np.empty((n_paths, n_periods, returns_daily.shape[1]), dtype=np.float32)
    for i in range(n_paths):
        arr[i] = stress.stationary_bootstrap_returns(
            returns_daily, n_periods, block, rng
        ).to_numpy()
    return arr


def _max_factor_over_g(
    arr: np.ndarray, w: np.ndarray, f_period: float, rebalance_period: int
) -> np.ndarray:
    """``X = max_t(factor_t / g_t)`` per path (the worst margin point of each path)."""
    n_periods = arr.shape[1]
    X = np.zeros(arr.shape[0])
    for t in range(0, n_periods, rebalance_period):
        seg = arr[:, t : t + rebalance_period, :]
        g = np.cumprod(1.0 + seg, axis=1) @ w  # (n_paths, seg_len)
        factor = (1.0 + f_period) ** np.arange(1, seg.shape[1] + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            seg_max = np.where(g > 0, factor / g, np.inf).max(axis=1)
        X = np.maximum(X, seg_max)
    return X


def _critical_from_X(X: np.ndarray, maintenance: float) -> np.ndarray:
    """Leverage at which each path's worst margin ratio reaches ``maintenance``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        crit = np.where(
            X > (1.0 - maintenance), 1.0 / (1.0 - (1.0 - maintenance) / X), np.inf
        )
    return np.clip(crit, 1.0, None)


def critical_leverages(
    returns_daily: pd.DataFrame,
    weights: pd.Series,
    *,
    maintenance: float = 0.25,
    horizon_years: int = 30,
    n_paths: int = 2000,
    block: int = 63,
    spread: float = 0.005,
    rebalance_period: int = 63,
    periods_per_year: int = 252,
    seed: int = 12345,
) -> np.ndarray:
    """Per bootstrapped path, the leverage at which margin is first called.

    See the module docstring for the identity. Defaults mirror the study's daily
    stress test (stationary bootstrap, mean block ~63 trading days, quarterly
    resets, financing at €STR + 0.5 %).
    """
    w = weights.reindex(returns_daily.columns).to_numpy()
    n_periods = horizon_years * periods_per_year
    arr = _resampled_array(returns_daily, n_periods, block, n_paths, seed)
    f_period = stress.period_financing(stress.ESTR_NOMINAL + spread, periods_per_year)
    X = _max_factor_over_g(arr, w, f_period, rebalance_period)
    return _critical_from_X(X, maintenance)


def margin_call_probability(critical: np.ndarray, k: float) -> float:
    """Share of paths whose critical leverage is below ``k`` (i.e. that get called)."""
    return float((critical < k).mean())


def ruin_safe_leverage(
    returns_daily: pd.DataFrame, weights: pd.Series, *, ruin_budget: float = 0.05, **kwargs
) -> float:
    """Largest ``k`` with bootstrap ``P(call) <= ruin_budget`` (the budget quantile)."""
    crit = critical_leverages(returns_daily, weights, **kwargs)
    return float(np.percentile(crit, ruin_budget * 100.0))


def historical_breach_leverage(
    returns_daily: pd.DataFrame,
    weights: pd.Series,
    *,
    maintenance: float = 0.25,
    spread: float = 0.005,
    rebalance_period: int = 63,
    periods_per_year: int = 252,
) -> float:
    """The leverage at which the *realised* history's worst margin ratio hits ``maintenance``.

    The "no historical call" rule: the single actual path, not the bootstrap.
    """
    w = weights.reindex(returns_daily.columns).to_numpy()
    arr = returns_daily.to_numpy()[None, :, :].astype(np.float32)
    f_period = stress.period_financing(stress.ESTR_NOMINAL + spread, periods_per_year)
    X = _max_factor_over_g(arr, w, f_period, rebalance_period)[0]
    return float(1.0 / (1.0 - (1.0 - maintenance) / X)) if X > (1.0 - maintenance) else np.inf


def composition_sweep(
    cov: pd.DataFrame,
    mu: pd.Series,
    returns_daily: pd.DataFrame,
    tangency_weights: pd.Series,
    *,
    w_eq_grid: list[float] | None = None,
    gammas: tuple[float, ...] = (1.0, 2.0),
    rf_real: float = RF_REAL,
    ruin_budget: float = 0.05,
    maintenance: float = 0.25,
    horizon_years: int = 30,
    n_paths: int = 2000,
    block: int = 63,
    spread: float = 0.005,
    rebalance_period: int = 63,
    periods_per_year: int = 252,
    seed: int = 12345,
) -> pd.DataFrame:
    """Sweep the equity weight; for each, cap leverage at ``k_margin`` and score utility.

    A one-parameter family: equity takes ``w_eq``, the remainder is split among the
    other sleeves in the tangency's internal proportions. For each ``gamma`` the
    leverage is ``min(k*(gamma), k_margin)`` and both the compound growth and the
    CRRA utility ``J = rf + k*ER - 0.5*gamma*(k*sigma)^2`` are reported. The
    bootstrap is drawn once and reused across compositions.
    """
    order = list(cov.index)
    Cov = cov.reindex(index=order, columns=order).to_numpy()
    Mu = mu.reindex(order).to_numpy()
    div = [a for a in order if a != "equity"]
    divprop = tangency_weights[div] / tangency_weights[div].sum()
    if w_eq_grid is None:
        base = round(float(tangency_weights["equity"]), 3)
        w_eq_grid = [base, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]

    n_periods = horizon_years * periods_per_year
    arr = _resampled_array(returns_daily, n_periods, block, n_paths, seed)
    f_period = stress.period_financing(stress.ESTR_NOMINAL + spread, periods_per_year)
    col = {a: i for i, a in enumerate(returns_daily.columns)}
    idx = [col[a] for a in order]

    rows = []
    for w_eq in w_eq_grid:
        wv = np.array([w_eq if a == "equity" else (1 - w_eq) * divprop[a] for a in order])
        er = float(wv @ Mu) - rf_real
        sigma = float(np.sqrt(wv @ Cov @ wv))
        sharpe = er / sigma
        X = _max_factor_over_g(arr[:, :, idx], wv, f_period, rebalance_period)
        k_margin = float(np.percentile(_critical_from_X(X, maintenance), ruin_budget * 100.0))
        row = {"w_eq": w_eq, "sigma": sigma, "sharpe": sharpe, "k_margin": k_margin}
        for g in gammas:
            k = min(sharpe / (g * sigma), k_margin)
            row[f"k_g{g:g}"] = k
            row[f"Er_g{g:g}"] = rf_real + k * er
            row[f"growth_g{g:g}"] = rf_real + k * er - 0.5 * (k * sigma) ** 2
            row[f"J_g{g:g}"] = rf_real + k * er - 0.5 * g * (k * sigma) ** 2
        rows.append(row)
    return pd.DataFrame(rows)
