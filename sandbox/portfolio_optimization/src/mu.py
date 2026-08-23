"""Expected returns (mu) and the real rate constants for the study.

From-source real, EUR, ~5-10y views, used as-is (no shrinkage; the Black-Litterman
step supplies the humility). See the "Expected returns" section of the qmd for
each sleeve's derivation.
"""

from __future__ import annotations

import pandas as pd

# From-source real expected returns (see the "Expected returns" section).
VIEWS_REAL = {
    # Elm Wealth's own all-world P-CAEY, 4.03% real as of 2026-06-30 -- the
    # P-CAEY authors' (Haghani & White) published live capital-market
    # assumption, built region by region (US ~2.9%, ex-US ~5.8%, cap-weighted).
    "equity": 0.0403,
    # IEF 4.60% USD YTM - 1.6pp EUR hedge cost (US-EUR short diff) - 2.0% inflation ~= 1.0% real
    "bonds": 0.010,
    # forward: ~0.30 net Sharpe x ~9.5% strategy vol + ~0.19% real cash; between the gross
    # ceiling (CFM/HOP 0.72-0.76) and the live SG-CTA floor (~0.13). Cayas CMA 2.7%.
    "trend": 0.0270,
    "gold": 0.0000,  # long-run real return = inflation (Ilmanen)
}

# Real RISK-FREE cash rate: the collateral yield / prior anchor.
# = overnight €STR 2.186% (ECB, reference 2026-07-24) - 2.0% expected inflation
# ~= 0.19% real. Anchors the BL equilibrium prior, the tangency Sharpe
# (composition), and the trend sleeve's collateral yield -- NOT the leverage,
# which is financed at the box borrowing rate BOX_REAL below.
RF_REAL = 0.0019

# Real BOX-SPREAD borrowing rate that finances the leverage (leverage.py):
# 3-month Euribor 2.524% (euribor-rates.eu, 2026-08-21) + 0.5% box spread
# (Cayas's central assumption, "MSCI World ... Cayas fait (bien) mieux") - 2.0%
# inflation ~= 1.02% real. Distinct from RF_REAL: the box is a ~3-month loan, so
# its base is the 3-month rate and the +0.5% is the box's spread over it; the
# prior / tangency / collateral instead earn the overnight risk-free.
BOX_REAL = 0.0102


def expected_returns(cov: pd.DataFrame | None = None) -> pd.Series:
    """The from-source real mu vector, ordered to match ``cov`` if given."""
    mu = pd.Series(VIEWS_REAL, name="mu")
    return mu.reindex(cov.index) if cov is not None else mu
