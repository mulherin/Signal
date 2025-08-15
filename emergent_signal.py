# emergent_signal.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# ---- helpers ----
def _align_three(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = a.index.intersection(b.index).intersection(c.index)
    cols = a.columns.intersection(b.columns).intersection(c.columns)
    return a.loc[idx, cols], b.loc[idx, cols], c.loc[idx, cols]

def _kday_shock_z(resid: pd.DataFrame, k: int, vol_win: int) -> pd.DataFrame:
    ksum = resid.rolling(int(k)).sum()
    vol  = ksum.rolling(int(vol_win)).std(ddof=0).replace(0.0, np.nan)
    return ksum.divide(vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ---- core ----
def compute_emergent_time_series(RS_med_pct: pd.DataFrame,
                                 resid: pd.DataFrame,
                                 pseudo: pd.DataFrame,
                                 industry_map: Optional[pd.Series],
                                 cfg) -> Dict[str, pd.DataFrame]:
    # Align inputs
    RS_med_pct, resid, pseudo = _align_three(RS_med_pct.astype(float), resid.astype(float), pseudo.astype(float))
    dates, names = RS_med_pct.index, RS_med_pct.columns

    # Minimal params (override via Config if present)
    K          = int(getattr(cfg, "ADD_SHOCK_LEN_D", 5))
    Z_THR      = float(getattr(cfg, "ADD_SHOCK_Z_THR", 0.8))
    VOL_WIN    = int(getattr(cfg, "ADD_VOL_WIN_D", 60))
    TREND_WIN  = int(getattr(cfg, "ADD_TREND_GATE_LEN_D", 63))
    RS_FLOOR_L = float(getattr(cfg, "ADD_RS_FLOOR_LONG", 0.50))
    RS_CEIL_S  = float(getattr(cfg, "ADD_RS_CEIL_SHORT", 0.50))
    NEWS_SIG   = float(getattr(cfg, "ADD_NEWS_SIGMA_MAX", 3.0))

    # Trend gate: pseudo slope over TREND_WIN and RS alignment
    logp = np.log(pseudo.clip(lower=1e-12))
    d63  = logp - logp.shift(TREND_WIN)
    uptrend   = (d63 > 0) & (RS_med_pct >= RS_FLOOR_L)
    downtrend = (d63 < 0) & (RS_med_pct <= RS_CEIL_S)

    # Shock + "freshness" + news filter
    z = _kday_shock_z(resid, K, VOL_WIN)
    day_vol = resid.rolling(20).std(ddof=0).replace(0.0, np.nan)
    news_bad = resid.abs() >= (NEWS_SIG * day_vol)   # skip extreme prints

    today_neg = resid < 0
    today_pos = resid > 0

    buydip    = uptrend & (~news_bad) & (z <= -Z_THR) & today_neg
    faderally = downtrend & (~news_bad) & (z >=  Z_THR) & today_pos

    # Build labels
    State = pd.DataFrame("", index=dates, columns=names, dtype=object)
    State[buydip]    = "BuyDip"
    State[faderally] = "FadeRally"

    TTL_Rem = pd.DataFrame(0, index=dates, columns=names, dtype=int)  # no TTL in this design
    return {"State": State, "TTL_Rem": TTL_Rem}

def compute_emergent_daily(emergent_ts: Dict[str, pd.DataFrame]):
    S = emergent_ts["State"]; T = emergent_ts["TTL_Rem"]
    if S.empty: return pd.Series(dtype=object), pd.Series(dtype=int)
    last = S.index[-1]
    return S.loc[last], T.loc[last].astype(int)

def backtest_emergent(resid: pd.DataFrame, emergent_ts: Dict[str, pd.DataFrame], cfg):
    labels = emergent_ts["State"].reindex_like(resid)
    dates = labels.index
    if len(dates) == 0: return pd.Series(dtype=float), {}

    step = int(getattr(cfg, "EMERGENT_REBALANCE_FREQ_D", 1))  # daily for tactical sleeve
    rb = dates[::max(1, step)]
    if len(rb) == 0 or rb[-1] != dates[-1]:
        rb = rb.append(pd.Index([dates[-1]]))

    long_budget  = float(getattr(cfg, "EMERGENT_LONG_BUDGET", 1.0))
    short_budget = float(getattr(cfg, "EMERGENT_SHORT_BUDGET", 0.0))

    W = pd.DataFrame(0.0, index=dates, columns=labels.columns)
    for i, t0 in enumerate(rb[:-1]):
        t1 = rb[i + 1]
        row = labels.loc[t0]
        longs  = row.eq("BuyDip")
        shorts = row.eq("FadeRally") & (short_budget > 0.0)
        nL, nS = int(longs.sum()), int(shorts.sum())
        if nL > 0:                        W.loc[t0:t1, longs]  =  long_budget / float(nL)
        if short_budget > 0.0 and nS > 0: W.loc[t0:t1, shorts] = -short_budget / float(nS)

    W_lag = W.shift(1).fillna(0.0)
    ar = np.expm1(resid.fillna(0.0))
    port = (W_lag.reindex_like(ar) * ar).sum(axis=1)
    eq = (1.0 + port).cumprod()
    dd = (eq / eq.cummax()) - 1.0
    stats = {"Sharpe_ann": float(port.mean() / port.std(ddof=0) * (252 ** 0.5)) if port.std(ddof=0) > 0 else float("nan"),
             "MaxDD": float(dd.min())}
    return eq, stats
