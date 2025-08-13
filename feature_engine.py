# feature_engine.py
from typing import Dict
import numpy as np
import pandas as pd
from config import Config
from utils import (
    cross_sectional_percentile,
    residual_returns,
    residual_pseudo_price,
    adx_close_only,
)

def _align_like(df: pd.DataFrame, idx: pd.Index, cols: pd.Index, fill: float = np.nan) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(fill, index=idx, columns=cols)
    return df.reindex(index=idx, columns=cols)

def build_features(prices: pd.DataFrame,
                   betas: pd.Series,
                   valuation_raw: pd.DataFrame,
                   cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Returns dict with keys:
      - Resid: residual daily log returns
      - Pseudo: residual pseudo-price (cum of Resid)
      - RS: standardized residual-sum (RS_std)
      - RS_pct: cross-sectional percentile of RS_std
      - RS_med_pct: cross-sectional percentile of RS rolling-sum over cfg.RS_MED_LOOKBACK_D (apples-to-apples base for Emergent)
      - Accel_pct: cross-sectional percentile of ΔRS over cfg.ACCEL_LOOKBACK_D
      - ADX: ADX computed from pseudo-price (close-only)
      - ADX_ROC: ADX change over cfg.ADX_ROC_LEN_D
      - Val_pct: aligned valuation percentile (ffill), or 0.5 if absent
    """
    # 1) Residual model & pseudo price
    resid = residual_returns(prices, betas).fillna(0.0)
    pseudo = residual_pseudo_price(resid).clip(lower=1e-12)

    # 2) RS and volatility standardization (for general ranking)
    look = int(cfg.RS_LOOKBACK_D)
    RS_sum = resid.rolling(look).sum()
    vol = resid.rolling(look).std(ddof=0).replace(0.0, np.nan)
    RS_std = RS_sum.divide(vol).fillna(0.0)

    # 3) Cross-sectional percentiles (standardized)
    RS_pct = cross_sectional_percentile(RS_std)

    # 3b) Apples-to-apples "med" horizon on RS rolling-sum (not standardized)
    med_look = int(cfg.RS_MED_LOOKBACK_D)
    RS_med_sum = resid.rolling(med_look).sum()
    RS_med_pct = cross_sectional_percentile(RS_med_sum)

    # 4) Acceleration (pace) on RS rolling-sum over cfg.ACCEL_LOOKBACK_D
    accel_look = int(cfg.ACCEL_LOOKBACK_D)
    accel = RS_sum - RS_sum.shift(accel_look)
    Accel_pct = cross_sectional_percentile(accel)

    # 5) ADX & ROC from pseudo (close-only)
    ADX = pd.DataFrame(index=pseudo.index, columns=pseudo.columns, dtype=float)
    for c in pseudo.columns:
        ADX[c] = adx_close_only(pseudo[c], int(cfg.ADX_LEN_D))
    ADX_ROC = ADX.diff(int(cfg.ADX_ROC_LEN_D))

    # 6) Valuation percentile (aligned & ffilled), or neutral 0.5 if missing
    if valuation_raw is not None and not valuation_raw.empty:
        Val_pct = _align_like(valuation_raw, RS_pct.index, RS_pct.columns).ffill().fillna(0.5)
    else:
        Val_pct = pd.DataFrame(0.5, index=RS_pct.index, columns=RS_pct.columns)

    return {
        "Resid": resid,
        "Pseudo": pseudo,
        "RS": RS_std,           # standardized RS for general ranking/backtests (Trend/Stars)
        "RS_pct": RS_pct,
        "RS_med_pct": RS_med_pct,   # NEW: used by Emergent for apples-to-apples crosses
        "Accel_pct": Accel_pct,     # now respects cfg.ACCEL_LOOKBACK_D
        "ADX": ADX,
        "ADX_ROC": ADX_ROC,
        "Val_pct": Val_pct,
    }

def compute_unpredictable_flag(features: Dict[str, pd.DataFrame], cfg: Config) -> pd.Series:
    """
    Simple 'unpredictable' flag per ticker:
      1) Low auto-predictability of RS (RS vs 20d lag)
      2) Persistently low ADX (too whippy)
    """
    RS = features["RS"]
    ADX = features["ADX"]

    look = 63
    RS_win = RS.tail(look)
    corr = {}
    for c in RS.columns:
        s = RS_win[c]
        y = s.shift(20)
        m = s.notna() & y.notna()
        corr[c] = s[m].corr(y[m]) if m.any() else np.nan

    ADX_win = ADX.tail(look)
    ADX_pct = ADX_win.rank(axis=1, pct=True)
    adx_med = ADX_pct.median(axis=0)

    out = []
    for c in RS.columns:
        cond1 = np.isnan(corr[c]) or abs(corr[c]) <= cfg.UNPRED_CORR_MAX
        cond2 = (adx_med.get(c, np.nan) <= cfg.UNPRED_ADX_MAX_PCT)
        out.append(bool(cond1 or cond2))
    return pd.Series(out, index=RS.columns)
