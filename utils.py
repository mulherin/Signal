# utils.py
# Copy-paste ready

from typing import Tuple
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
    """
    Close-only ADX on a pseudo-price series.
    """
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
    """
    For each date, percentile of current value within trailing window.
    """
    def _pct(x: np.ndarray) -> float:
        last = x[-1]
        # include equals to be consistent
        return float(np.mean(x <= last))
    return series.rolling(window=window_days, min_periods=min_days).apply(_pct, raw=True)

def daily_to_years(days: float) -> int:
    return int(round(days))

def years_to_days(years: float) -> int:
    return int(round(years * 252))

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
