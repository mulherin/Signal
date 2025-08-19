# feature_engine.py
# Streamlined: removes ADX/ADX_ROC; keeps only features used by Trend, Emergent, and Stars.
# Dependencies: utils.cross_sectional_percentile, utils.residual_returns, utils.residual_pseudo_price. :contentReference[oaicite:0]{index=0}

from typing import Dict
import numpy as np
import pandas as pd

from config import Config
from utils import (
    cross_sectional_percentile,
    residual_returns,
    residual_pseudo_price,
)


def _align_like(df: pd.DataFrame, idx: pd.Index, cols: pd.Index, fill: float = np.nan) -> pd.DataFrame:
    """
    Reindex df to (idx, cols). If df is empty, return a frame filled with `fill`.
    """
    if df is None or df.empty:
        return pd.DataFrame(fill, index=idx, columns=cols)
    return df.reindex(index=idx, columns=cols)


def build_features(
    prices: pd.DataFrame,
    betas: pd.Series,
    valuation_raw: pd.DataFrame,
    cfg: Config,
) -> Dict[str, pd.DataFrame]:
    """
    Build minimal, leak-free features for the trading stack.

    Returns a dict with:
      - "Resid":   residual daily log returns
      - "Pseudo":  residual pseudo-price (cum of Resid), base=100
      - "RS":      standardized k-day residual-sum (z-like)
      - "RS_pct":  cross-sectional percentile of RS (0..1) by day
      - "RS_med_pct": cross-sectional percentile of residual rolling-sum
                      at the medium lookback (apples-to-apples base for Emergent)
      - "Val_pct": aligned valuation percentile (ffill) or 0.5 if absent
    """
    # 1) Residual model & pseudo price
    resid = residual_returns(prices, betas).fillna(0.0)
    pseudo = residual_pseudo_price(resid).clip(lower=1e-12)

    # 2) RS (rolling residual-sum standardized by rolling vol)
    look = int(getattr(cfg, "RS_LOOKBACK_D", 126))
    RS_sum = resid.rolling(look).sum()
    vol = resid.rolling(look).std(ddof=0).replace(0.0, np.nan)
    RS_std = RS_sum.divide(vol).fillna(0.0)

    # 3) Cross-sectional percentiles (standardized RS)
    RS_pct = cross_sectional_percentile(RS_std)

    # 3b) Apples-to-apples medium horizon on raw RS rolling-sum (not standardized)
    med_look = int(getattr(cfg, "RS_MED_LOOKBACK_D", look))
    RS_med_sum = resid.rolling(med_look).sum()
    RS_med_pct = cross_sectional_percentile(RS_med_sum)

    # 4) Valuation percentile (aligned & ffilled), or neutral 0.5 if missing
    if valuation_raw is not None and not valuation_raw.empty:
        Val_pct = _align_like(valuation_raw, RS_pct.index, RS_pct.columns).ffill().fillna(0.5)
    else:
        Val_pct = pd.DataFrame(0.5, index=RS_pct.index, columns=RS_pct.columns)

    return {
        "Resid": resid,
        "Pseudo": pseudo,
        "RS": RS_std,
        "RS_pct": RS_pct,
        "RS_med_pct": RS_med_pct,
        "Val_pct": Val_pct,
    }


def compute_unpredictable_flag(features: Dict[str, pd.DataFrame], cfg: Config) -> pd.Series:
    """
    Simple 'unpredictable' badge per ticker using RS autocorrelation only.
    A name is flagged if |corr(RS, RS lag 20d)| <= UNPRED_CORR_MAX over the last 63 days.
    """
    RS = features.get("RS")
    if RS is None or RS.empty:
        return pd.Series(dtype=bool)

    look = 63
    lag = 20
    corr_thresh = float(getattr(cfg, "UNPRED_CORR_MAX", 0.05))

    RS_win = RS.tail(look)
    out = {}
    for c in RS_win.columns:
        s = RS_win[c]
        y = s.shift(lag)
        m = s.notna() & y.notna()
        corr = s[m].corr(y[m]) if m.any() else np.nan
        out[c] = (np.isnan(corr) or abs(corr) <= corr_thresh)
    return pd.Series(out, index=RS_win.columns)
