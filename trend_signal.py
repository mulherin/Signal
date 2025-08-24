# trend_signal.py
# Streamlined Trend classification with NO ADX usage.
# - Computes rolling slope, t-stat, and R² on log residual pseudo-price.
# - Buckets names into Onside / Monitor / Offside using t-stat, RS percentile, and an R² floor.
# - Optional industry confirmation and hysteresis to reduce label flip-flops.
# - Provides a simple Score (t-stat pct + R²) and Score_for_weights (zeroed unless Onside).
#
# Public API:
#   compute_trend_time_series(features, cfg) -> Dict[str, pd.DataFrame]                # global reference frame
#   compute_trend_time_series_industry(features, cfg) -> Dict[str, pd.DataFrame]      # industry-relative twin
#   backtest_trend(resid, trend_ts, cfg) -> Tuple[pd.Series, Dict[str, float]]

from typing import Tuple, Dict
import numpy as np
import pandas as pd
from datetime import datetime

from config import Config
from utils import cross_sectional_percentile, demean_within_groups
from data_loader import load_industry_map  # only used if industry confirm is enabled


# ---------------- helpers (debug printing) ----------------

def _dbg(cfg: Config) -> bool:
    return bool(getattr(cfg, "TREND_LOG", False))

def _log(cfg: Config, msg: str) -> None:
    if _dbg(cfg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [trend] {msg}", flush=True)


# ---------------- rolling regression w/ t-stat and R² ----------------

def _rolling_slope_tstat_r2(series: pd.Series, window: int = 63) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Rolling OLS of log(pseudo-price) on time (0..n-1), over 'window' days.
    Returns (slope, tstat, r2), each aligned to series.index.
    """
    y = np.log(np.maximum(series.astype(float).values, 1e-12))
    n = int(window)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    Sxx = float(np.sum((x - x_mean) ** 2))

    slope = pd.Series(index=series.index, dtype=float)
    tstat = pd.Series(index=series.index, dtype=float)
    r2    = pd.Series(index=series.index, dtype=float)

    for i in range(n - 1, len(y)):
        win = y[i - n + 1:i + 1]
        y_mean = float(win.mean())
        Sxy = float(np.sum((x - x_mean) * (win - y_mean)))
        b = Sxy / Sxx if Sxx > 0 else np.nan
        a = y_mean - b * x_mean
        e = win - (a + b * x)
        dof = max(n - 2, 1)
        s2 = float(np.sum(e * e) / dof)
        se_b = np.sqrt(s2 / Sxx) if Sxx > 0 else np.nan
        t = float(b / se_b) if (se_b is not None and se_b > 0) else np.nan
        sst = float(np.sum((win - y_mean) ** 2))
        r2_i = float(1.0 - (np.sum(e * e) / sst)) if sst > 0 else 0.0

        slope.iloc[i] = b
        tstat.iloc[i] = t
        r2.iloc[i]    = r2_i

    return slope, tstat, r2


# ---------------- main API (GLOBAL reference frame) ----------------

def compute_trend_time_series(features: Dict[str, pd.DataFrame],
                              cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Trend with:
      - Rolling OLS slope & t-stat on log(residual pseudo-price) over TREND_TSTAT_LEN_D (default 63)
      - RS floor (TREND_MIN_RSPCT)
      - Optional R² floor (TREND_R2_MIN; default 0.15)
      - Optional industry soft-confirm: name stronger than its industry (TREND_USE_INDUSTRY_CONFIRM)
      - NO ADX usage.

    Returns dict of DataFrames:
      { "Tstat", "Slope", "R2", "Score", "Score_for_weights", "Class" }
    """
    pseudo  = features["Pseudo"]       # residual pseudo-price
    RS_pct  = features["RS_pct"]       # cross-sectional RS percentile (0..1)
    window  = int(getattr(cfg, "TREND_TSTAT_LEN_D", 63))

    # 1) rolling regression metrics
    slope_df = pd.DataFrame(index=pseudo.index, columns=pseudo.columns, dtype=float)
    tstat_df = pd.DataFrame(index=pseudo.index, columns=pseudo.columns, dtype=float)
    r2_df    = pd.DataFrame(index=pseudo.index, columns=pseudo.columns, dtype=float)

    _log(cfg, f"computing rolling slope/t-stat/R2 over window={window}d on {pseudo.shape[1]} names")
    for c in pseudo.columns:
        s, t, r2 = _rolling_slope_tstat_r2(pseudo[c], window=window)
        slope_df[c] = s
        tstat_df[c] = t
        r2_df[c]    = r2

    # 2) quality score (t-stat percentile + R²; NO ADX)
    tstat_pct = cross_sectional_percentile(tstat_df).fillna(0.0)
    r2_clip   = r2_df.clip(lower=0.0, upper=1.0).fillna(0.0)

    w_t, w_r2 = 0.75, 0.25
    score_df = (w_t * tstat_pct + w_r2 * r2_clip) / (w_t + w_r2)

    # 3) label classification
    on_thr    = float(cfg.TREND_TSTAT_UP)
    down_thr  = float(cfg.TREND_TSTAT_DOWN)
    rspct_thr = float(cfg.TREND_MIN_RSPCT)
    r2_min    = float(getattr(cfg, "TREND_R2_MIN", 0.15))  # set to 0.0 to disable

    Class = pd.DataFrame("Offside", index=tstat_df.index, columns=tstat_df.columns, dtype=object)

    # 3a) optional industry soft-confirm (name > industry on RS)
    use_ind = bool(getattr(cfg, "TREND_USE_INDUSTRY_CONFIRM", False))
    ind_margin = float(getattr(cfg, "TREND_INDUSTRY_MARGIN", 0.0))
    ind_map = None
    if use_ind:
        ind_map = load_industry_map(cfg.TREND_INPUT_PATH)
        if ind_map is None or ind_map.empty:
            _log(cfg, "industry map not found -> disabling industry confirm for trend.")
            use_ind = False
        else:
            # reindex to our columns; unseen names fall into their own group
            ind_map = ind_map.reindex(tstat_df.columns).fillna(
                pd.Series(range(len(tstat_df.columns)), index=tstat_df.columns)
            )

    if use_ind:
        # demean RS within industry each day
        ind_demeaned = RS_pct.apply(lambda row: demean_within_groups(row, ind_map), axis=1)
        ind_ok_long  = ind_demeaned >= ind_margin
    else:
        ind_ok_long  = pd.DataFrame(True, index=RS_pct.index, columns=RS_pct.columns)

    # 3b) raw gates
    cond_t  = (tstat_df >= on_thr)
    cond_rs = (RS_pct   >= rspct_thr)
    cond_r2 = (r2_clip  >= r2_min)

    # "Onside" requires all: strong t-stat + RS + R² + (optionally) industry-uniqueness
    cond_on = cond_t & cond_rs & cond_r2 & ind_ok_long

    # "Monitor" = decent trend shape or RS but missing one of the ON requirements
    cond_monitor = ((tstat_df >= on_thr) & (~cond_on)) | (((tstat_df > 0) & (tstat_df < on_thr)) & cond_rs)

    Class[cond_on] = "Onside"
    Class[~cond_on & cond_monitor] = "Monitor"

    # 3c) optional hysteresis to stabilize labels
    if bool(getattr(cfg, "TREND_USE_HYSTERESIS", False)):
        _log(cfg, "applying hysteresis to labels")
        for c in tstat_df.columns:
            prev = "Offside"
            for t in tstat_df.index:
                tval = tstat_df.at[t, c]
                rsv  = RS_pct.at[t, c]
                if np.isnan(tval) or np.isnan(rsv):
                    Class.at[t, c] = prev
                else:
                    if (tval >= on_thr) and (rsv >= rspct_thr) and (r2_clip.at[t, c] >= r2_min) and bool(ind_ok_long.at[t, c]):
                        prev = "Onside"
                    elif prev == "Onside" and (tval > down_thr) and (rsv >= rspct_thr):
                        prev = "Monitor"
                    else:
                        prev = "Offside"
                    Class.at[t, c] = prev

    # 4) score for portfolio weights: zero out non-onside
    Score_for_weights = score_df.where(Class.eq("Onside"), other=0.0)

    # 5) lightweight sanity logging
    if _dbg(cfg):
        every = max(1, int(getattr(cfg, "TREND_LOG_EVERY_D", 20)))
        for i, t in enumerate(tstat_df.index):
            if (i % every) != 0:
                continue
            N = int(tstat_df.iloc[i].notna().sum())
            on_cnt = int(cond_on.iloc[i].sum())
            mon_cnt = int(cond_monitor.iloc[i].sum())
            ts_row = tstat_df.iloc[i].dropna()
            r2_row = r2_df.iloc[i].dropna()
            q_ts = np.quantile(ts_row, [0.25, 0.5, 0.75]).tolist() if len(ts_row) else [np.nan]*3
            q_r2 = np.quantile(r2_row, [0.25, 0.5, 0.75]).tolist() if len(r2_row) else [np.nan]*3
            _log(cfg, f"{t.date()} sanity N={N} Onside={on_cnt} Monitor={mon_cnt} "
                      f"tstat_q25/50/75={q_ts[0]:.2f}/{q_ts[1]:.2f}/{q_ts[2]:.2f} "
                      f"R2_q25/50/75={q_r2[0]:.2f}/{q_r2[1]:.2f}/{q_r2[2]:.2f}")

    return {
        "Tstat": tstat_df,
        "Slope": slope_df,
        "R2": r2_df,
        "Score": score_df,
        "Score_for_weights": Score_for_weights,
        "Class": Class,
    }


# ---------------- industry-relative twin ----------------

def compute_trend_time_series_industry(features: Dict[str, pd.DataFrame],
                                       cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Industry-relative Trend:
      - Rolling OLS slope & t-stat on log(Pseudo_ind) over TREND_TSTAT_LEN_D.
      - RS floor uses RS_pct_ind (within-industry percentile).
      - Optional R² floor via TREND_R2_MIN.
      - Same hysteresis behavior as the global Trend.
      - No extra "industry confirm" gate (the inputs are already industry-relative).

    Expects keys built in feature_engine:
      features["Pseudo_ind"], features["RS_pct_ind"].
    """
    # --- pull inputs; raise a clear error if industry features are missing ---
    if "Pseudo_ind" not in features or "RS_pct_ind" not in features:
        raise KeyError("Industry-relative features not found. Build_features must provide "
                       "'Pseudo_ind' and 'RS_pct_ind' before calling compute_trend_time_series_industry().")

    pseudo_ind = features["Pseudo_ind"]   # industry-demeaned pseudo price
    RS_pct_ind = features["RS_pct_ind"]   # within-industry RS percentile (0..1)
    window = int(getattr(cfg, "TREND_TSTAT_LEN_D", 63))

    # --- rolling regression metrics on Pseudo_ind ---
    slope_df = pd.DataFrame(index=pseudo_ind.index, columns=pseudo_ind.columns, dtype=float)
    tstat_df = pd.DataFrame(index=pseudo_ind.index, columns=pseudo_ind.columns, dtype=float)
    r2_df    = pd.DataFrame(index=pseudo_ind.index, columns=pseudo_ind.columns, dtype=float)

    _log(cfg, f"[industry] computing rolling slope/t-stat/R2 over window={window}d on {pseudo_ind.shape[1]} names")
    for c in pseudo_ind.columns:
        s, t, r2 = _rolling_slope_tstat_r2(pseudo_ind[c], window=window)
        slope_df[c] = s
        tstat_df[c] = t
        r2_df[c]    = r2

    # --- quality score: t-stat percentile (within industry if map exists) + R² ---
    try:
        # groupwise ranks if industry map is available
        from utils import cross_sectional_percentile_within_groups  # added in utils.py
        from data_loader import load_industry_map
        ind_map = load_industry_map(cfg.TREND_INPUT_PATH).reindex(tstat_df.columns)
        if ind_map is None or ind_map.empty:
            tstat_pct = cross_sectional_percentile(tstat_df).fillna(0.0)
        else:
            min_group = int(getattr(cfg, "TREND_GROUP_MIN_SIZE", 3))
            tstat_pct = cross_sectional_percentile_within_groups(
                tstat_df, groups=ind_map, min_group_size=min_group, fallback=None
            ).fillna(0.0)
    except Exception:
        # Fallback to global percentile if groupwise helper or map is unavailable
        tstat_pct = cross_sectional_percentile(tstat_df).fillna(0.0)

    r2_clip = r2_df.clip(lower=0.0, upper=1.0).fillna(0.0)

    w_t, w_r2 = 0.75, 0.25
    score_df = (w_t * tstat_pct + w_r2 * r2_clip) / (w_t + w_r2)

    # --- classification gates (identical thresholds to global Trend) ---
    on_thr    = float(cfg.TREND_TSTAT_UP)
    down_thr  = float(cfg.TREND_TSTAT_DOWN)
    rspct_thr = float(cfg.TREND_MIN_RSPCT)
    r2_min    = float(getattr(cfg, "TREND_R2_MIN", 0.15))  # set to 0.0 to disable

    Class = pd.DataFrame("Offside", index=tstat_df.index, columns=tstat_df.columns, dtype=object)

    cond_t  = (tstat_df >= on_thr)
    cond_rs = (RS_pct_ind >= rspct_thr)
    cond_r2 = (r2_clip   >= r2_min)

    cond_on = cond_t & cond_rs & cond_r2

    # "Monitor" = decent trend shape or RS but missing one of the ON requirements
    cond_monitor = ((tstat_df >= on_thr) & (~cond_on)) | (((tstat_df > 0) & (tstat_df < on_thr)) & cond_rs)

    Class[cond_on] = "Onside"
    Class[~cond_on & cond_monitor] = "Monitor"

    # --- optional hysteresis to stabilize labels (uses industry RS floor) ---
    if bool(getattr(cfg, "TREND_USE_HYSTERESIS", False)):
        _log(cfg, "[industry] applying hysteresis to labels")
        for c in tstat_df.columns:
            prev = "Offside"
            for t in tstat_df.index:
                tval = tstat_df.at[t, c]
                rsv  = RS_pct_ind.at[t, c]
                if np.isnan(tval) or np.isnan(rsv):
                    Class.at[t, c] = prev
                else:
                    if (tval >= on_thr) and (rsv >= rspct_thr) and (r2_clip.at[t, c] >= r2_min):
                        prev = "Onside"
                    elif prev == "Onside" and (tval > down_thr) and (rsv >= rspct_thr):
                        prev = "Monitor"
                    else:
                        prev = "Offside"
                    Class.at[t, c] = prev

    # --- score for portfolio weights: zero out non-onside ---
    Score_for_weights = score_df.where(Class.eq("Onside"), other=0.0)

    # --- light sanity logging ---
    if _dbg(cfg):
        every = max(1, int(getattr(cfg, "TREND_LOG_EVERY_D", 20)))
        for i, t in enumerate(tstat_df.index):
            if (i % every) != 0:
                continue
            N = int(tstat_df.iloc[i].notna().sum())
            on_cnt = int(cond_on.iloc[i].sum())
            mon_cnt = int(cond_monitor.iloc[i].sum())
            ts_row = tstat_df.iloc[i].dropna()
            r2_row = r2_df.iloc[i].dropna()
            q_ts = np.quantile(ts_row, [0.25, 0.5, 0.75]).tolist() if len(ts_row) else [np.nan]*3
            q_r2 = np.quantile(r2_row, [0.25, 0.5, 0.75]).tolist() if len(r2_row) else [np.nan]*3
            _log(cfg, f"{t.date()} [industry] N={N} Onside={on_cnt} Monitor={mon_cnt} "
                      f"tstat_q25/50/75={q_ts[0]:.2f}/{q_ts[1]:.2f}/{q_ts[2]:.2f} "
                      f"R2_q25/50/75={q_r2[0]:.2f}/{q_r2[1]:.2f}/{q_r2[2]:.2f}")

    return {
        "Tstat": tstat_df,
        "Slope": slope_df,
        "R2": r2_df,
        "Score": score_df,
        "Score_for_weights": Score_for_weights,
        "Class": Class,
    }


# ---------------- backtest (unchanged) ----------------

def backtest_trend(resid: pd.DataFrame,
                   trend: Dict[str, pd.DataFrame],
                   cfg: Config) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Long-only by default:
      - At each rebalance date, hold equal weights across names labeled Onside.
      - Total long notional = cfg.TREND_LONG_BUDGET. No shorts unless cfg.TREND_SHORT_BUDGET > 0.
      - t+1 execution, arithmetic compounding.
    """
    Class = trend["Class"]
    dates = Class.index
    if len(dates) == 0:
        return pd.Series(dtype=float), {}

    step = max(1, int(cfg.TREND_REBALANCE_FREQ_D))
    rb = dates[::step]
    if len(rb) == 0 or rb[-1] != dates[-1]:
        rb = rb.append(pd.Index([dates[-1]]))

    W = pd.DataFrame(0.0, index=dates, columns=Class.columns)
    long_budget = float(cfg.TREND_LONG_BUDGET)
    short_budget = float(cfg.TREND_SHORT_BUDGET)

    for i, t0 in enumerate(rb[:-1]):
        t1 = rb[i + 1]
        row = Class.loc[t0]
        longs = (row == "Onside")
        shorts = (row == "Offside") & (short_budget > 0.0)  # normally false (long-only)

        nL = int(longs.sum())
        nS = int(shorts.sum())

        if nL > 0:
            W.loc[t0:t1, longs] = long_budget / float(nL)
        if short_budget > 0.0 and nS > 0:
            W.loc[t0:t1, shorts] = -short_budget / float(nS)

    # t+1 execution, arithmetic compounding
    W_lag = W.shift(1).fillna(0.0)
    ar = np.expm1(resid.fillna(0.0))
    port = (W_lag.reindex_like(ar) * ar).sum(axis=1)
    eq = (1.0 + port).cumprod()

    dd = (eq / eq.cummax()) - 1.0
    stats = {
        "Sharpe_ann": float(port.mean() / port.std(ddof=0) * (252 ** 0.5)) if port.std(ddof=0) > 0 else float("nan"),
        "MaxDD": float(dd.min()),
    }
    return eq, stats
