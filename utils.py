# Minimal helper utilities used across the streamlined Signals stack.
# Removed ADX/ADX_ROC and other unused helpers.

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def cross_sectional_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-wise percentiles (0..1) with NaNs preserved.
    Ties receive average rank behavior (pandas default).
    """
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None,
                            columns=df.columns if df is not None else None,
                            dtype=float)
    return df.rank(axis=1, pct=True)


# ---------- NEW: groupwise percentiles ----------

def percentile_within_groups(row: pd.Series,
                             groups: Optional[pd.Series],
                             min_group_size: int = 1,
                             fallback: Optional[float] = None) -> pd.Series:
    """
    Percentile-rank a single cross section (one row) *within each group*.

    Parameters
    ----------
    row : pd.Series
        Cross section at a single date. Index are tickers. Values are numeric.
    groups : pd.Series or None
        Mapping ticker -> group label (e.g., industry). If None or empty, falls back to global percentile.
        Missing group labels are treated as unique groups for their tickers.
    min_group_size : int
        If a group's total membership is < min_group_size, use a fallback for those tickers.
        Default 1 means "always compute within-group percentiles."
    fallback : Optional[float]
        What to use for tickers in tiny groups:
          - None (default): use the *global* cross-sectional percentile of `row`.
          - float value (e.g., 0.5): use this constant.

    Returns
    -------
    pd.Series of floats in [0,1], aligned to `row.index`. NaNs preserved where `row` is NaN.
    """
    if row is None or row.empty:
        return pd.Series(index=(row.index if row is not None else None), dtype=float)

    # No groups provided -> global percentile
    if groups is None or groups.empty:
        return row.rank(pct=True)

    # Align groups to row and ensure missing labels become unique per ticker
    g = groups.reindex(row.index)
    if g.isna().any():
        g = g.copy()
        g[g.isna()] = g.index[g.isna()]  # each missing gets its own group id

    # Base within-group percentile ranks (NaNs preserved)
    within = row.groupby(g, dropna=False).rank(pct=True)

    # Tiny-group fallback handling
    if int(min_group_size) > 1:
        # size per element's group
        sizes = g.map(g.value_counts(dropna=False))
        tiny_mask = (sizes < int(min_group_size)) & row.notna()

        if tiny_mask.any():
            if fallback is None:
                global_pct = row.rank(pct=True)
                within = within.where(~tiny_mask, global_pct)
            else:
                within = within.where(~tiny_mask, float(fallback))

    return within.astype(float)


def cross_sectional_percentile_within_groups(df: pd.DataFrame,
                                             groups: Optional[pd.Series],
                                             min_group_size: int = 1,
                                             fallback: Optional[float] = None) -> pd.DataFrame:
    """
    Row-wise percentiles within groups for an entire DataFrame.
    Applies `percentile_within_groups` to each date independently.

    Parameters mirror `percentile_within_groups`.
    """
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None,
                            columns=df.columns if df is not None else None,
                            dtype=float)
    return df.apply(lambda row: percentile_within_groups(row, groups, min_group_size, fallback), axis=1)


# ---------- existing helpers (unchanged) ----------

def residual_returns(prices: pd.DataFrame, betas: pd.Series) -> pd.DataFrame:
    """
    Residual daily log returns:
      r_it_resid = r_it - beta_i * r_mkt_t
    where r_mkt_t is the equal-weighted cross-sectional return at t.

    prices : wide DataFrame (dates x tickers), level-adjusted prices
    betas  : Series indexed by tickers (same columns as prices); missing betas default to 1.0
    """
    if prices is None or prices.empty:
        return pd.DataFrame(dtype=float)

    rets = np.log(prices.astype(float)).diff()

    # equal-weight market return each day
    r_mkt = rets.mean(axis=1)

    # align betas to columns, default 1.0 where missing
    b = betas.reindex(rets.columns).fillna(1.0) if isinstance(betas, pd.Series) else pd.Series(1.0, index=rets.columns)

    beta_mat = pd.DataFrame(np.tile(b.values, (len(rets), 1)),
                            index=rets.index, columns=rets.columns)

    resid = rets - beta_mat.mul(r_mkt, axis=0)
    return resid


def residual_pseudo_price(resid: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """
    Pseudo-price from residual log returns (cumexp) with a fixed base.
    """
    if resid is None or resid.empty:
        return pd.DataFrame(dtype=float)
    return np.exp(resid.fillna(0.0).cumsum()) * float(base)


def current_run_length(flag_df: pd.DataFrame) -> pd.DataFrame:
    """
    For a boolean DataFrame, return the length (in days) of the current contiguous
    True-run at each (date, ticker). Zeros where False.
    """
    if flag_df is None or flag_df.empty:
        return pd.DataFrame(dtype=int)

    out = pd.DataFrame(0, index=flag_df.index, columns=flag_df.columns, dtype=int)
    for c in flag_df.columns:
        s = flag_df[c].fillna(False).astype(bool)
        grp = (~s).cumsum()                 # increments when s flips to False -> new run id for True blocks
        run = s.groupby(grp).cumcount() + 1 # counts within each True run
        run[~s] = 0
        out[c] = run.astype(int)
    return out


def demean_within_groups(w: pd.Series, groups: Optional[pd.Series]) -> pd.Series:
    """
    Demean a cross section (row) within provided groups (e.g., industry).
    Any names with missing group labels are demeaned by the global mean.
    """
    if w is None or w.empty:
        return w
    if groups is None or groups.empty:
        return w - w.mean()

    g = groups.reindex(w.index)
    # Compute group means for non-null groups
    by = w.groupby(g, dropna=True).transform("mean")
    # Fill entries whose group label is NaN with the global mean
    by = by.fillna(w.mean())
    return w - by
