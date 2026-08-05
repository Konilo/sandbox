"""Margin-call stress test for the levered book (evaluation, not allocation).

This module changes nothing about the allocation. It takes the chosen tangency
weights and CRRA leverage ``k`` and pushes them through history and through
block-bootstrapped paths, answering the first question any levered-strategy post
gets: what is the drawdown and margin-call risk?

Margin mechanics
----------------
Box-spread financing is a fixed-term loan -- the lender cannot pull or reprice it
mid-crisis -- but the borrowed cash buys marked-to-market ETFs, and IBKR still
liquidates if account equity falls below the maintenance requirement on those
longs. Normalising NAV0 = 1 with leverage ``k`` and maintenance ratio ``m``:

    Gross_t = sum_i P_{i,t}         (marked-to-market longs)
    Loan_t  = Loan_{t-1}(1 + f)     (box accrues the financing rate f)
    NAV_t   = Gross_t - Loan_t
    rho_t   = NAV_t / Gross_t       (margin ratio)  ->  call if rho_t < m

Between rebalances the loan is fixed while the longs drift, so effective leverage
rises after losses (no preventive deleveraging -- stricter than reality). At each
rebalance the book is reset to the target ``k`` and weights.

NOMINAL by construction: a margin call is a nominal accounting event and the
sleeve series are nominal index returns, so the loan accrues at a *nominal*
financing rate (``ESTR_NOMINAL`` + spread), not the study's real ``RF_REAL``.
The margin ratio is invariant to that choice anyway (a common deflator cancels
in NAV/Gross), but nominal keeps returns and financing on the same footing.

TIER 1 (monthly) -- this run is a FLOOR on call risk, not the answer: month-end
marks miss intra-month spikes (a crash that reverses within a month is invisible),
so realised call risk is higher. The daily reconstruction (tier 2) is the credible
version for the post.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Nominal financing base: the same ECB euro short-term rate that anchors the
# study's real financing (RF_REAL), reference 2026-07-24. The box costs ~this at
# its tenor; the sweep adds a spread on top (Cayas assumes +0.5%; +1% is severe).
ESTR_NOMINAL = 0.02186
FINANCING_SPREADS = (0.005, 0.010)

MAINTENANCE_RATIOS = (0.25, 0.30, 0.35)  # 25% ~ typical UCITS; 35% deliberately stressed
REBALANCE_MONTHS = {"quarterly": 3, "annual": 12}

# Named crisis windows (month-end), for the historical worst-episode report. The
# dot-com window opens at the data start, so the book is already levered into the
# fall -- its drawdown is the deepest partly for that start-date reason.
CRISES = {
    "2001-03 dot-com": ("2001-01", "2003-03"),
    "2008 GFC": ("2007-10", "2009-02"),
    "2020 Covid": ("2020-01", "2020-04"),
    "2022 reflation": ("2021-12", "2022-10"),
}

# Daily-frequency analogues (stress-test tier 2). The daily window starts 2008-10
# (DBZB), so 2008 is only partially covered and the dot-com is out of range.
DAILY_PER_YEAR = 252
DAILY_REBALANCE = {"quarterly": 63, "annual": 252}  # trading days between resets
DAILY_CRISES = {
    "2008 GFC": ("2008-10", "2009-06"),
    "2020 Covid": ("2020-02", "2020-04"),
    "2022 reflation": ("2021-12", "2022-10"),
}


def period_financing(f_annual: float, periods_per_year: int = 12) -> float:
    """Per-period compounding-equivalent of an annual financing rate."""
    return (1.0 + f_annual) ** (1.0 / periods_per_year) - 1.0


def max_drawdown(nav: np.ndarray) -> float:
    """Worst peak-to-trough drawdown of a NAV path (<= 0)."""
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


@dataclass
class PathResult:
    """Outcome of one levered-book path."""

    nav: pd.Series  # NAV at each period end (index matches the returns)
    ratio: pd.Series  # margin ratio rho_t
    max_drawdown: float  # worst peak-to-trough NAV drawdown (<= 0)
    min_cushion: float  # min(rho_t - m) over the path (< 0 means a call fired)
    first_breach: pd.Timestamp | int | None  # index label of the first rho_t < m
    ruin: bool  # NAV hit zero


def simulate_path(
    returns: pd.DataFrame,
    weights: pd.Series,
    k: float,
    m: float,
    f_annual: float,
    rebalance_period: int,
    periods_per_year: int = 12,
) -> PathResult:
    """Run the levered book through ``returns`` and track the margin ratio.

    ``returns`` are periodic simple returns per sleeve; ``weights`` the tangency
    weights (target long book, summing to 1); ``k`` gross leverage; ``m`` the
    maintenance ratio. Rebalancing resets the book to ``k`` and ``weights`` every
    ``rebalance_period`` periods (positional). A call is flagged the first time
    ``rho_t < m``; the path continues (binary breach, no forced liquidation).

    Vectorised per rebalance segment: within a segment starting at NAV ``v``, the
    gross book grows as ``v*k*g_t`` (with ``g_t`` the weighted cumulative product
    of sleeve returns) and the loan as ``v*(k-1)*(1+f)**days``, so ``NAV`` and the
    margin ratio follow in closed form and only ``v`` recurses across segments.
    """
    w = weights.reindex(returns.columns).to_numpy()
    R = returns.to_numpy()
    f_period = period_financing(f_annual, periods_per_year)
    n = len(returns)

    navs = np.empty(n)
    ratios = np.empty(n)
    nav_start = 1.0
    ruin = False

    t = 0
    while t < n:
        end = min(t + rebalance_period, n)
        g = np.cumprod(1.0 + R[t:end], axis=0) @ w  # weighted gross growth since reset
        factor = (1.0 + f_period) ** np.arange(1, end - t + 1)
        nav_seg = nav_start * (k * g - (k - 1.0) * factor)
        gross_seg = nav_start * k * g
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_seg = np.where(gross_seg > 0, nav_seg / gross_seg, -np.inf)
        navs[t:end] = nav_seg
        ratios[t:end] = ratio_seg
        if (nav_seg <= 0).any():  # ruin: NAV hit zero mid-segment
            j = int(np.argmax(nav_seg <= 0))
            navs[t + j :] = nav_seg[j]
            ratios[t + j :] = -np.inf
            ruin = True
            break
        nav_start = nav_seg[-1]
        t = end

    nav_s = pd.Series(navs, index=returns.index)
    ratio_s = pd.Series(ratios, index=returns.index)
    first_breach = returns.index[int(np.argmax(ratios < m))] if (ratios < m).any() else None
    dd_path = np.concatenate([[1.0], navs])
    return PathResult(
        nav=nav_s,
        ratio=ratio_s,
        max_drawdown=max_drawdown(dd_path),
        min_cushion=float((ratio_s - m).min()),
        first_breach=first_breach,
        ruin=ruin,
    )


def historical_grid(
    returns: pd.DataFrame,
    weights: pd.Series,
    k: float,
    *,
    maintenance=MAINTENANCE_RATIOS,
    spreads=FINANCING_SPREADS,
    rebalances=REBALANCE_MONTHS,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Run the actual historical path across the (m, spread, rebalance) grid."""
    rows = []
    for reb_name, reb_period in rebalances.items():
        for spread in spreads:
            for m in maintenance:
                res = simulate_path(
                    returns, weights, k, m, ESTR_NOMINAL + spread, reb_period,
                    periods_per_year,
                )
                rows.append(
                    {
                        "rebalance": reb_name,
                        "financing": f"€STR+{spread:.1%}",
                        "m": f"{m:.0%}",
                        "max drawdown": f"{res.max_drawdown:.0%}",
                        "min cushion": f"{res.min_cushion:+.1%}",
                        "call": "yes" if res.first_breach is not None else "no",
                        "first call": (
                            "--"
                            if res.first_breach is None
                            else pd.Timestamp(res.first_breach).strftime("%Y-%m")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def crisis_report(
    returns: pd.DataFrame,
    weights: pd.Series,
    k: float,
    f_annual: float,
    rebalance_period: int,
    *,
    crises=CRISES,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Per-crisis levered drawdown and lowest margin ratio.

    Both are independent of the maintenance ratio ``m`` (which only sets the call
    threshold), so the reader compares the lowest ``rho`` directly against 25 /
    30 / 35 %. Drawdown is peak-to-trough within each window.
    """
    res = simulate_path(returns, weights, k, 0.0, f_annual, rebalance_period,
                        periods_per_year)
    rows = []
    for name, (start, end) in crises.items():
        nav_w = res.nav.loc[start:end].to_numpy()
        rho_w = res.ratio.loc[start:end]
        rows.append(
            {
                "crisis": name,
                "levered drawdown": f"{max_drawdown(nav_w):.0%}",
                "lowest margin ratio": f"{rho_w.min():.1%}",
            }
        )
    return pd.DataFrame(rows)


def block_bootstrap_returns(
    returns: pd.DataFrame, n_periods: int, block: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Concatenate random fixed-length blocks of consecutive rows into a path.

    Preserves serial dependence within each block; breaks it at block boundaries.
    """
    t = len(returns)
    n_blocks = int(np.ceil(n_periods / block))
    starts = rng.integers(0, t - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n_periods]
    return returns.iloc[idx].reset_index(drop=True)


def stationary_bootstrap_returns(
    returns: pd.DataFrame, n_periods: int, mean_block: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Politis & Romano (1994) stationary bootstrap.

    Geometric-length blocks (mean ``mean_block``) of consecutive rows, wrapped
    circularly. Each step continues the current block with probability
    ``1 - 1/mean_block`` or jumps to a fresh random start otherwise -- so there is
    no fixed block boundary and the resampled series is stationary, preserving
    serial dependence more smoothly than fixed blocks.
    """
    t = len(returns)
    restart = rng.random(n_periods) < (1.0 / mean_block)
    restart[0] = True  # the first draw always starts a block
    segment = np.cumsum(restart) - 1  # 0-based block index of each step
    starts = np.flatnonzero(restart)  # step positions where a block begins
    offset = np.arange(n_periods) - starts[segment]  # position within the block
    base = rng.integers(0, t, size=len(starts))  # random start row per block
    idx = (base[segment] + offset) % t  # circular walk from each block's base
    return returns.iloc[idx].reset_index(drop=True)


def bootstrap_grid(
    returns: pd.DataFrame,
    weights: pd.Series,
    k: float,
    *,
    horizon_years: int = 30,
    n_paths: int = 2000,
    block: int = 12,
    bootstrap: str = "block",
    seed: int = 12345,
    maintenance=MAINTENANCE_RATIOS,
    spreads=FINANCING_SPREADS,
    rebalances=REBALANCE_MONTHS,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Bootstrap ``n_paths`` horizon-year paths; tabulate call/drawdown risk.

    ``bootstrap`` is ``"block"`` (fixed-length blocks) or ``"stationary"`` (Politis
    & Romano geometric blocks with mean length ``block``). ``P(call)`` is the share
    of paths with at least one ``rho_t < m``; the fifth percentile of max drawdown
    is the adverse tail. Paths reuse a synthetic index (rebalancing is positional).
    """
    rng = np.random.default_rng(seed)
    n_periods = horizon_years * periods_per_year
    resample = (
        stationary_bootstrap_returns if bootstrap == "stationary" else block_bootstrap_returns
    )
    paths = [resample(returns, n_periods, block, rng) for _ in range(n_paths)]
    rows = []
    for reb_name, reb_period in rebalances.items():
        for spread in spreads:
            # The margin-ratio path is independent of m (m only sets the call
            # threshold), so simulate each path once and read every m off it.
            min_rho = np.empty(n_paths)
            dds = np.empty(n_paths)
            for i, path in enumerate(paths):
                res = simulate_path(
                    path, weights, k, 0.0, ESTR_NOMINAL + spread, reb_period,
                    periods_per_year,
                )
                min_rho[i] = res.ratio.min()
                dds[i] = res.max_drawdown
            for m in maintenance:
                rows.append(
                    {
                        "rebalance": reb_name,
                        "financing": f"€STR+{spread:.1%}",
                        "m": f"{m:.0%}",
                        "P(call) 30y": f"{(min_rho < m).mean():.0%}",
                        "DD (median)": f"{np.median(dds):.0%}",
                        "DD (5th pct)": f"{np.percentile(dds, 5):.0%}",
                    }
                )
    return pd.DataFrame(rows)
