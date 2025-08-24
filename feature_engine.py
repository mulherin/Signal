# Streamlined: removes ADX/ADX_ROC; keeps only features used by Trend, Emergent, and Stars.
# Adds industry-relative variants for use by an industry-relative Trend track:
#   - Resid_ind, Pseudo_ind
#   - RS_pct_ind, RS_med_pct_ind
# Dependencies: utils.cross_sectional_percentile, utils.residual_returns, utils.residual_pseudo_price,
#               utils.cross_sectional_percentile_within_groups

from typing import Dict, Optional
import numpy as np
import pandas as pd

from config import Config
from data_loader import load_industry_map
from utils import (
    cross_sectional_percentile,
    cross_sectional_percentile_within_groups,
    residual_returns,
    residual_pseudo_price,
    demean_within_groups,
)


def _align_like(df: pd.DataFrame, idx: pd.Index, cols: pd.Index, fill: float = np.nan) -> pd.DataFrame:
    """
    Reindex df to (idx, cols). If df is empty, return a frame filled with `fill`.
    """
    if df is None or df.empty:
        return pd.DataFrame(fill, index=idx, columns=cols)
    return df.reindex(index=idx, columns=cols)


def _build_industry_map(cols: pd.Index, cfg: Config) -> pd.Series:
    """
    Load and align the ticker -> industry map to the provided columns.
    Missing labels are kept as NaN here. Downstream code handles fallbacks.
    """
    ind = load_industry_map(cfg.TREND_INPUT_PATH)
    if ind is None or ind.empty:
        return pd.Series(index=cols, dtype=object)
    ind = ind.reindex(cols)
    return ind


def _demean_within_groups_safe(resid: pd.DataFrame,
                               ind_map: Optional[pd.Series],
                               min_group_size: int) -> pd.DataFrame:
    """
    Row-wise demeaning within industry with a safe fallback:
      - If industry_map is empty: return input resid unchanged.
      - If a group has size < min_group_size: fall back to *global* demeaning for those tickers.
      - If a ticker has a missing label: fall back to *global* demeaning for that ticker.
    """
    if ind_map is None or ind_map.empty:
        return resid.copy()

    # Precompute group sizes per ticker (static across rows)
    size_map = ind_map.map(ind_map.value_counts(dropna=False)).fillna(0).astype(int)
    tiny_mask = size_map < int(min_group_size)

    # Base: industry-demeaned residuals (row-wise)
    base = resid.apply(lambda row: demean_within_groups(row, ind_map), axis=1)

    # Replace tiny groups with global demeaned residuals to avoid zeroing single-stock groups
    if tiny_mask.any():
        global_demeaned = resid.sub(resid.mean(axis=1), axis=0)
        base.loc[:, tiny_mask] = global_demeaned.loc[:, tiny_mask]
    return base


def build_features(
    prices: pd.DataFrame,
    betas: pd.Series,
    valuation_raw: pd.DataFrame,
    cfg: Config,
) -> Dict[str, pd.DataFrame]:
    """
    Build minimal, leak-free features for the trading stack.

    Returns a dict with:
      - "Resid":         residual daily log returns
      - "Pseudo":        residual pseudo-price (cum of Resid), base=100
      - "RS":            standardized k-day residual-sum (z-like)
      - "RS_pct":        cross-sectional percentile of RS (0..1) by day
      - "RS_med_pct":    cross-sectional percentile of residual rolling-sum at the medium lookback
      - "Val_pct":       aligned valuation percentile (ffill) or 0.5 if absent

      Industry-relative additions (parallel track):
      - "Resid_ind":       residuals demeaned within industry (tiny groups -> global demeaning)
      - "Pseudo_ind":      pseudo-price built from Resid_ind
      - "RS_pct_ind":      percentiles computed within industry each day
      - "RS_med_pct_ind":  medium-horizon percentiles within industry
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

    # 3b) Medium horizon on raw RS rolling-sum (not standardized)
    med_look = int(getattr(cfg, "RS_MED_LOOKBACK_D", look))
    RS_med_sum = resid.rolling(med_look).sum()
    RS_med_pct = cross_sectional_percentile(RS_med_sum)

    # 4) Valuation percentile (aligned & ffilled), or neutral 0.5 if missing
    if valuation_raw is not None and not valuation_raw.empty:
        Val_pct = _align_like(valuation_raw, RS_pct.index, RS_pct.columns).ffill().fillna(0.5)
    else:
        Val_pct = pd.DataFrame(0.5, index=RS_pct.index, columns=RS_pct.columns)

    # ---------------- Industry-relative track ----------------
    # Map and params
    ind_map = _build_industry_map(resid.columns, cfg)
    min_group = int(getattr(cfg, "TREND_GROUP_MIN_SIZE", 3))  # if not in Config, defaults to 3

    # 1) Industry-demeaned residuals with safe fallback for tiny/missing groups
    resid_ind = _demean_within_groups_safe(resid, ind_map, min_group_size=min_group)

    # 2) Pseudo from industry-demeaned residuals
    pseudo_ind = residual_pseudo_price(resid_ind).clip(lower=1e-12)

    # 3) RS on industry-demeaned residuals
    RS_ind_sum = resid_ind.rolling(look).sum()
    vol_ind = resid_ind.rolling(look).std(ddof=0).replace(0.0, np.nan)
    RS_ind_std = RS_ind_sum.divide(vol_ind).fillna(0.0)

    # 4) Percentiles within industry each day
    RS_pct_ind = cross_sectional_percentile_within_groups(
        RS_ind_std, groups=ind_map, min_group_size=min_group, fallback=None
    )
    RS_med_sum_ind = resid_ind.rolling(med_look).sum()
    RS_med_pct_ind = cross_sectional_percentile_within_groups(
        RS_med_sum_ind, groups=ind_map, min_group_size=min_group, fallback=None
    )

    return {
        "Resid": resid,
        "Pseudo": pseudo,
        "RS": RS_std,
        "RS_pct": RS_pct,
        "RS_med_pct": RS_med_pct,
        "Val_pct": Val_pct,

        # Industry-relative additions
        "Resid_ind": resid_ind,
        "Pseudo_ind": pseudo_ind,
        "RS_ind": RS_ind_std,
        "RS_pct_ind": RS_pct_ind,
        "RS_med_pct_ind": RS_med_pct_ind,
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
