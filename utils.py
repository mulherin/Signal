# New helpers added; existing functions untouched.
from typing import Tuple, Optional
import numpy as np
import pandas as pd

def cross_sectional_percentile(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)

def residual_returns(prices: pd.DataFrame, betas: pd.Series) -> pd.DataFrame:
    rets = np.log(prices).diff()
    ew = rets.mean(axis=1)
    beta_mat = pd.DataFrame(np.tile(betas.values, (len(rets), 1)), index=rets.index, columns=rets.columns)
    resid = rets - beta_mat.mul(ew, axis=0)
    return resid

def residual_pseudo_price(resid: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    return np.exp(resid.fillna(0).cumsum()) * base

def wilder_ema(x: pd.Series, n: int) -> pd.Series:
    alpha = 1.0 / max(n, 1)
    return x.ewm(alpha=alpha, adjust=False, min_periods=n).mean()

def adx_close_only(series: pd.Series, n: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0).abs()
    dn = (-delta).clip(lower=0.0).abs()
    tr = delta.abs()
    atr = wilder_ema(tr.fillna(0.0), n)
    plus_di = 100.0 * (wilder_ema(up.fillna(0.0), n) / atr.replace(0, np.nan))
    minus_di = 100.0 * (wilder_ema(dn.fillna(0.0), n) / atr.replace(0, np.nan))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = wilder_ema(dx.fillna(0.0), n)
    return adx

def within_name_trailing_percentile(series: pd.Series, window_days: int, min_days: int) -> pd.Series:
    def _pct(x: np.ndarray) -> float:
        last = x[-1]
        return float(np.mean(x <= last))
    return series.rolling(window=window_days, min_periods=min_days).apply(_pct, raw=True)

def years_to_days(years: float) -> int:
    return int(round(years * 252))

# -------- NEW: cross-sectional zscore row-wise
def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0).replace(0.0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).fillna(0.0)

# -------- NEW: orthogonalize a score to a factor, return 0..1 percentiles
def orthogonalize_to_factor(score: pd.DataFrame, factor: pd.DataFrame) -> pd.DataFrame:
    s_z = cs_zscore(score)
    f_z = cs_zscore(factor)
    out = pd.DataFrame(index=score.index, columns=score.columns, dtype=float)
    for t in score.index:
        srow = s_z.loc[t]; frow = f_z.loc[t]
        m = srow.notna() & frow.notna()
        if m.sum() >= 3:
            X = np.column_stack([np.ones(m.sum()), frow[m].values])
            beta, *_ = np.linalg.lstsq(X, srow[m].values, rcond=None)
            resid = np.full(len(srow), np.nan)
            resid[m] = srow[m].values - X.dot(beta)
            out.loc[t] = resid
        else:
            out.loc[t] = srow.values
    return cross_sectional_percentile(out.fillna(0.0))

# -------- NEW: current run-length of True (per column)
def current_run_length(flag_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(0, index=flag_df.index, columns=flag_df.columns, dtype=int)
    for c in flag_df.columns:
        s = flag_df[c].fillna(False).astype(bool)
        grp = (~s).cumsum()
        run = s.groupby(grp).cumcount() + 1
        run[~s] = 0
        out[c] = run.astype(int)
    return out

# -------- NEW: demean within groups (e.g., industry)
def demean_within_groups(w: pd.Series, groups: Optional[pd.Series]) -> pd.Series:
    if groups is None or groups.empty:
        return w - w.mean()
    aligned = groups.reindex(w.index)
    return w - w.groupby(aligned).transform("mean")
