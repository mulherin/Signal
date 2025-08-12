# feature_engine.py
# Copy-paste ready
# Implements RS standardization by idiosyncratic volatility so cross-sectional
# rankings use RS_std rather than raw RS, following Blitz, Huij & Martens (2011).
# Residual Momentum paper reference: 
# Blitz, Huij, Martens (2011), 'Residual Momentum'. :contentReference[oaicite:0]{index=0}

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from config import Config
from utils import (
    cross_sectional_percentile,
    residual_returns,
    residual_pseudo_price,
    adx_close_only,
    within_name_trailing_percentile,
    years_to_days,
)

def build_features(prices: pd.DataFrame,
                   betas: pd.Series,
                   valuation_raw: pd.DataFrame,
                   cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Returns a feature dict keyed by:
      - "Resid": residual daily returns
      - "Pseudo": pseudo price built from residuals
      - "RS": rolling sum of residuals over cfg.RS_LOOKBACK_D
      - "RS_std": RS divided by residual vol over the same window
      - "RS_pct": cross-sectional percentile of RS_std (0..1)
      - "RS_pct_raw": cross-sectional percentile of raw RS (0..1) for diagnostics
      - "Accel": day-over-day change in RS_pct over cfg.ACCEL_LOOKBACK_D
      - "Accel_pct": cross-sectional percentile of Accel
      - "ADX": ADX computed from pseudo price
      - "ADX_ROC": ADX - ADX shifted by cfg.ADX_ROC_LEN_D
      - "Val_pct": within-name trailing percentile of valuation (higher = richer)
    """
    # 1) Residual model and pseudo price
    resid = residual_returns(prices, betas)
    pseudo = residual_pseudo_price(resid)

    # 2) RS and volatility standardization
    RS = resid.rolling(cfg.RS_LOOKBACK_D).sum()
    RS_vol = resid.rolling(cfg.RS_LOOKBACK_D).std(ddof=0)
    RS_std = RS.divide(RS_vol.replace(0.0, np.nan))
    RS_std = RS_std.fillna(0.0)

    # 3) Percentile views (use standardized RS for production)
    RS_pct_raw = cross_sectional_percentile(RS)
    RS_pct = cross_sectional_percentile(RS_std)

    # 4) Acceleration on percentile RS
    Accel = RS_pct - RS_pct.shift(cfg.ACCEL_LOOKBACK_D)
    Accel_pct = cross_sectional_percentile(Accel)

    # 5) ADX and ADX_ROC from pseudo price (no high/low needed)
    ADX = pd.DataFrame(index=pseudo.index, columns=pseudo.columns, dtype=float)
    for c in pseudo.columns:
        ADX[c] = adx_close_only(pseudo[c], cfg.ADX_LEN_D)
    ADX_ROC = ADX - ADX.shift(cfg.ADX_ROC_LEN_D)

    # 6) Valuation trailing within-name percentiles
    if valuation_raw is None or valuation_raw.empty:
        Val_pct = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)  # neutral
    else:
        win_max = years_to_days(cfg.VAL_WINDOW_Y_MAX)
        win_min = years_to_days(cfg.VAL_WINDOW_Y_MIN)
        Val_pct = pd.DataFrame(index=valuation_raw.index, columns=valuation_raw.columns, dtype=float)
        for c in valuation_raw.columns:
            Val_pct[c] = within_name_trailing_percentile(valuation_raw[c], window_days=win_max, min_days=win_min)

        # align to prices index
        Val_pct = Val_pct.reindex(prices.index).ffill().clip(lower=0.0, upper=1.0)

    return {
        "Resid": resid,
        "Pseudo": pseudo,
        "RS": RS,
        "RS_std": RS_std,
        "RS_pct": RS_pct,
        "RS_pct_raw": RS_pct_raw,
        "Accel": Accel,
        "Accel_pct": Accel_pct,
        "ADX": ADX,
        "ADX_ROC": ADX_ROC,
        "Val_pct": Val_pct,
    }

def compute_unpredictable_flag(features: Dict[str, pd.DataFrame], cfg: Config) -> pd.Series:
    """
    Heuristic flag for names where short-horizon RS is weakly related to the
    pseudo price trend OR where ADX has been persistently low.
    Returns a boolean Series indexed by ticker.
    """
    RS = features["RS"]
    pseudo = features["Pseudo"]
    ADX = features["ADX"]

    # Look back roughly one year for these diagnostics
    look = years_to_days(1.0)
    RS_win = RS.tail(look)
    # Use simple pct-change in pseudo as a smoothed trend proxy
    trend = pseudo.pct_change().tail(look)

    # Correlation between RS and trend, by ticker (ignore missing)
    corr = {}
    for c in RS.columns:
        s = RS_win[c]
        y = trend[c]
        m = s.notna() & y.notna()
        corr[c] = s[m].corr(y[m]) if m.any() else np.nan

    # ADX cross-sectional percentile each day, then median per ticker over the window
    ADX_win = ADX.tail(look)
    ADX_pct = ADX_win.rank(axis=1, pct=True)
    adx_med = ADX_pct.median(axis=0)

    out = []
    for c in RS.columns:
        cond1 = np.isnan(corr[c]) or abs(corr[c]) <= cfg.UNPRED_CORR_MAX
        cond2 = (adx_med.get(c, np.nan) <= cfg.UNPRED_ADX_MAX_PCT)
        out.append(bool(cond1 or cond2))
    return pd.Series(out, index=RS.columns)
