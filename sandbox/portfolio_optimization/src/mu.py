"""Expected returns (mu) for the portfolio study.

Reasoned per sleeve -- real, EUR, ~5-10y horizon. Bonds, trend, and gold follow
Ilmanen's per-asset method; equity is Elm Wealth's published all-world P-CAEY
(the method's authors' own live figure). See draft/MU_ESTIMATION.md and the
"Expected returns" section for each sleeve's derivation.

These are honest from-source views, used as-is. An earlier version shrank them
toward an equal-Sharpe (risk-parity) prior, but that prior is volatility-scaled
(prior_i = s_bar * vol_i), so it rewards the highest-vol sleeve with the highest
return -- inflating gold (the least groundable sleeve) from an honest 0% toward
~2%, contradicting the structural view that Sharpe ratios are NOT equal, and
quietly lifting the Sharpe that sets the leverage. Risk-parity humility is a
statement about *weights*, so it now lives only in the composition step (the
tangency -> ERC blend), leaving mu -- and thus the leverage math -- uncorrupted.
"""

from __future__ import annotations

import pandas as pd

# From-source real expected returns (see the "Expected returns" section).
VIEWS_REAL = {
    # Elm Wealth's own all-world P-CAEY, 4.03% real as of 2026-06-30 -- the
    # P-CAEY authors' (Haghani & White) published live capital-market
    # assumption, built region by region (US ~2.9%, ex-US ~5.8%, cap-weighted).
    "equity": 0.0403,
    "bonds": 0.0120,  # WGBI-DM 4.08% local YTW - ~0.9pp EUR hedge - 2.0% inflation ~= 1.2% real
    "trend": 0.0180,  # SG CTA Index live net-of-fee: 3.8% nominal-EUR CAGR - 2.0% inflation
    "gold": 0.0000,  # long-run real return = inflation (Ilmanen)
}

# Real financing / risk-free anchor for the excess returns that drive BOTH the
# tangency (composition) and the CRRA leverage. It is load-bearing: a positive
# real rate thins every sleeve's excess return, pushes the long-only tangency
# out of any sub-rate sleeve, and lowers k* = S / (gamma * sigma).
#
# = EUR nominal financing - expected inflation
# = 2.186% (ECB euro short-term rate, reference 2026-07-24) - 2.0%
# ~= 0.19% real.
# The box-spread financing cost is the EUR risk-free curve at the box tenor;
# overnight EUR STR is its base -- exact if rolling short boxes / a flat curve,
# and it ignores any small box spread over the risk-free (both second-order).
RF_REAL = 0.0019


def expected_returns(cov: pd.DataFrame | None = None) -> pd.Series:
    """The from-source real mu vector, ordered to match ``cov`` if given."""
    mu = pd.Series(VIEWS_REAL, name="mu")
    return mu.reindex(cov.index) if cov is not None else mu
