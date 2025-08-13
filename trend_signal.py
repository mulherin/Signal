from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from config import Config
from utils import cross_sectional_percentile

def _rolling_slope_tstat(series: pd.Series, window: int = 63) -> Tuple[pd.Series, pd.Series]:
    """
    OLS slope & t-stat of log(pseudo-price) on time (0..n-1) over a rolling window.
    Returns (slope_series, tstat_series) aligned to series.index.
    """
    y = np.log(np.maximum(series.astype(float).values, 1e-12))
    n = int(window)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    Sxx = np.sum((x - x_mean) ** 2)

    def _beta(arr):
        y_win = arr
        y_mean = y_win.mean()
        Sxy = np.sum((x - x_mean) * (y_win - y_mean))
        b = Sxy / Sxx if Sxx > 0 else np.nan
        return b

    def _tstat(arr):
        y_win = arr
        y_mean = y_win.mean()
        Sxy = np.sum((x - x_mean) * (y_win - y_mean))
        b = Sxy / Sxx if Sxx > 0 else np.nan
        a = y_mean - b * x_mean
        e = y_win - (a + b * x)
        dof = max(n - 2, 1)
        s2 = float(np.sum(e * e) / dof)
        se_b = np.sqrt(s2 / Sxx) if Sxx > 0 else np.nan
        t = float(b / se_b) if (se_b is not None and se_b > 0) else np.nan
        return t

    slope = pd.Series(index=series.index, dtype=float)
    tstat = pd.Series(index=series.index, dtype=float)
    arr = y
    for i in range(n - 1, len(arr)):
        win = arr[i - n + 1:i + 1]
        slope.iloc[i] = _beta(win)
        tstat.iloc[i] = _tstat(win)
    return slope, tstat

def compute_trend_time_series(features: Dict[str, pd.DataFrame],
                              cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Minimal, first-principles Trend:
      - Trend_Tstat from 63d OLS slope of log(residual pseudo-price)
      - Guard: require RS_pct >= TREND_MIN_RSPCT to be Onside
      - No orthogonalization or overstretch gating in the LABEL
    """
    pseudo = features["Pseudo"]
    RS_pct = features["RS_pct"]
    window = 63

    slope_df = pd.DataFrame(index=pseudo.index, columns=pseudo.columns, dtype=float)
    tstat_df = pd.DataFrame(index=pseudo.index, columns=pseudo.columns, dtype=float)
    for c in pseudo.columns:
        s, t = _rolling_slope_tstat(pseudo[c], window=window)
        slope_df[c] = s
        tstat_df[c] = t

    # Percentile of t-stat for inspection/weights
    score_df = cross_sectional_percentile(tstat_df)

    # Classification
    on_thr = float(cfg.TREND_TSTAT_UP)
    down_thr = float(cfg.TREND_TSTAT_DOWN)
    rspct_thr = float(cfg.TREND_MIN_RSPCT)

    Class = pd.DataFrame("Offside", index=tstat_df.index, columns=tstat_df.columns, dtype=object)

    cond_on = (tstat_df >= on_thr) & (RS_pct >= rspct_thr)
    cond_monitor = ((tstat_df >= on_thr) & (RS_pct < rspct_thr)) | (((tstat_df > 0) & (tstat_df < on_thr)) & (RS_pct >= rspct_thr))

    Class[cond_on] = "Onside"
    Class[~cond_on & cond_monitor] = "Monitor"

    # Optional hysteresis (keeps labels stable)
    if bool(cfg.TREND_USE_HYSTERESIS):
        for c in tstat_df.columns:
            prev = "Offside"
            for t in tstat_df.index:
                tval = tstat_df.at[t, c]
                rsv = RS_pct.at[t, c]
                if np.isnan(tval) or np.isnan(rsv):
                    Class.at[t, c] = prev
                else:
                    if tval >= on_thr and rsv >= rspct_thr:
                        prev = "Onside"
                    elif prev == "Onside" and (tval > down_thr) and (rsv >= rspct_thr):
                        prev = "Monitor"
                    else:
                        prev = "Offside"
                    Class.at[t, c] = prev

    Score_for_weights = score_df.where(Class.eq("Onside"), other=0.0)

    return {
        "Tstat": tstat_df,
        "Slope": slope_df,
        "Score": score_df,
        "Score_for_weights": Score_for_weights,
        "Class": Class,
    }

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
        "MaxDD": float(dd.min()) if len(dd) else float("nan"),
        "Long_Budget": long_budget,
        "Short_Budget": short_budget,
    }
    return eq, stats
