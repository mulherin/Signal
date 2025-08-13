# emergent_signal.py
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from config import Config
from utils import demean_within_groups  # industry confirm (soft sign gate)

# ---------- small helpers ----------

def _pct_rank_rowwise(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)

def _rs_sum(resid: pd.DataFrame, look: int) -> pd.DataFrame:
    return resid.rolling(int(look)).sum()

def _dbg(cfg: Config) -> bool:
    return bool(getattr(cfg, "EMERGENT_LOG", False))

def _log(cfg: Config, msg: str):
    if _dbg(cfg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [emergent] {msg}", flush=True)

def _rolling_tstat_r2(series: pd.Series, window: int = 63) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling OLS of log(pseudo) on time over 'window' days.
    Returns (tstat, R2), aligned to series.index.
    """
    y = np.log(np.maximum(series.astype(float).values, 1e-12))
    n = int(window)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    Sxx = np.sum((x - x_mean) ** 2)

    tstat = pd.Series(index=series.index, dtype=float)
    r2 = pd.Series(index=series.index, dtype=float)

    for i in range(n - 1, len(y)):
        win = y[i - n + 1:i + 1]
        y_mean = win.mean()
        Sxy = np.sum((x - x_mean) * (win - y_mean))
        b = Sxy / Sxx if Sxx > 0 else np.nan
        a = y_mean - b * x_mean
        e = win - (a + b * x)
        dof = max(n - 2, 1)
        s2 = float(np.sum(e * e) / dof)
        se_b = np.sqrt(s2 / Sxx) if Sxx > 0 else np.nan
        t = float(b / se_b) if (se_b is not None and se_b > 0) else np.nan
        sst = float(np.sum((win - y_mean) ** 2))
        r2_i = float(1.0 - (np.sum(e * e) / sst)) if sst > 0 else 0.0
        tstat.iloc[i] = t
        r2.iloc[i] = r2_i

    return tstat, r2

# ---------- main API ----------

def compute_emergent_time_series(rs_med_pct: pd.DataFrame,
                                 resid: pd.DataFrame,
                                 pseudo: pd.DataFrame,
                                 industry_map: Optional[pd.Series],
                                 cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Emergent label (inflection/breakdown) without ADX, without percentile accel quotas,
    without strict RS floors.

    Triggers require:
      LONG (Inflection): cross_up + long_anchor + accel_size & nonnegative sign + trend_shape + industry soft sign (optional)
      SHORT (Breakdown): cross_dn + short_anchor + accel_size & nonpositive sign + trend_shape + industry soft sign (optional)

    - Cross uses RS_med_pct vs RS_long_pct with configurable margins.
    - Trend shape uses rolling slope t-stat and R² on log(pseudo) over EMERGENT_TSTAT_LEN_D.
    - Industry confirm is a soft sign gate on industry-demeaned RS_med_pct if enabled.

    Lifecycle: side-specific TTL (if provided) else EMERGENT_TTL_D; then cooldown.
    """

    idx, cols = rs_med_pct.index, list(rs_med_pct.columns)

    # 1) RS short/long for cross logic
    RS_short_pct = _pct_rank_rowwise(_rs_sum(resid, cfg.RS_SHORT_D))
    RS_long_pct  = _pct_rank_rowwise(_rs_sum(resid, cfg.RS_LONG_D))

    # 2) Cross with margins
    cross_up = (rs_med_pct >= (RS_long_pct + float(cfg.EMERGENT_LONG_CROSS_MARGIN)))
    cross_dn = (rs_med_pct <= (RS_long_pct - float(cfg.EMERGENT_SHORT_CROSS_MARGIN)))

    # 3) Directional anchors (asymmetric)
    anchor_up = (rs_med_pct >= float(cfg.DIR_ANCHOR_LONG_PCT))
    anchor_dn = (rs_med_pct <= float(cfg.DIR_ANCHOR_SHORT_PCT))

    # 4) Acceleration size + sign gate (no percentile quota)
    RS_sum_for_accel = _rs_sum(resid, cfg.RS_LOOKBACK_D)
    accel_raw = RS_sum_for_accel - RS_sum_for_accel.shift(int(cfg.ACCEL_LOOKBACK_D))
    accel_size_pass = accel_raw.abs() >= float(cfg.ACCEL_DELTA_MIN)
    accel_sign_pos = accel_raw >= 0.0
    accel_sign_neg = accel_raw <= 0.0

    # 5) Trend shape: rolling t-stat and R² on pseudo
    tlen = int(getattr(cfg, "EMERGENT_TSTAT_LEN_D", 63))
    tstat_df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    r2_df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for c in cols:
        t, r = _rolling_tstat_r2(pseudo[c], window=tlen)
        tstat_df[c] = t
        r2_df[c] = r

    t_up_min = float(getattr(cfg, "EMERGENT_TSTAT_MIN_UP", 0.5))
    t_dn_min = float(getattr(cfg, "EMERGENT_TSTAT_MIN_DN", 0.7))
    r2_min   = float(getattr(cfg, "EMERGENT_R2_MIN", 0.15))

    trend_ok_long  = (tstat_df >=  t_up_min) & (r2_df >= r2_min)
    trend_ok_short = (tstat_df <= -t_dn_min) & (r2_df >= r2_min)

    # 6) Industry confirmation (soft sign gate on industry-demeaned RS_med_pct)
    use_ind = bool(getattr(cfg, "EMERGENT_USE_INDUSTRY_CONFIRM", True))
    if use_ind and isinstance(industry_map, pd.Series) and not industry_map.empty:
        ind_ok_L = pd.DataFrame(False, index=idx, columns=cols)
        ind_ok_S = pd.DataFrame(False, index=idx, columns=cols)
        groups = industry_map.reindex(cols)
        for t in idx:
            row = rs_med_pct.loc[t]
            ex = demean_within_groups(row, groups)  # subtract industry mean per day
            ind_ok_L.loc[t] = (ex >= 0.0).values
            ind_ok_S.loc[t] = (ex <= 0.0).values
    else:
        ind_ok_L = pd.DataFrame(True, index=idx, columns=cols)
        ind_ok_S = pd.DataFrame(True, index=idx, columns=cols)

    # 7) Final guards
    long_guard = cross_up & anchor_up & accel_size_pass & accel_sign_pos & trend_ok_long & ind_ok_L
    short_guard = cross_dn & anchor_dn & accel_size_pass & accel_sign_neg & trend_ok_short & ind_ok_S

    # 8) State, TTL (asymmetric allowed), cooldown
    ttl_L = int(getattr(cfg, "EMERGENT_TTL_LONG_D", getattr(cfg, "EMERGENT_TTL_D", 28)))
    ttl_S = int(getattr(cfg, "EMERGENT_TTL_SHORT_D", getattr(cfg, "EMERGENT_TTL_D", 28)))
    cdn   = int(getattr(cfg, "EMERGENT_COOLDOWN_D", 10))

    state = pd.DataFrame("", index=idx, columns=cols)
    ttl = pd.DataFrame(0, index=idx, columns=cols, dtype=int)
    cooldown = pd.DataFrame(0, index=idx, columns=cols, dtype=int)

    # Logging cadence
    every = max(1, int(getattr(cfg, "EMERGENT_LOG_EVERY_D", 21)))
    _log(cfg, f"build: RS_short={cfg.RS_SHORT_D} RS_med={cfg.RS_MED_LOOKBACK_D} "
              f"RS_long={cfg.RS_LONG_D} tlen={tlen} "
              f"margins(L/S)=({cfg.EMERGENT_LONG_CROSS_MARGIN:.2f}/{cfg.EMERGENT_SHORT_CROSS_MARGIN:.2f}) "
              f"ACCEL_DELTA_MIN={cfg.ACCEL_DELTA_MIN} "
              f"TSTAT_MIN(up/dn)=({t_up_min:.2f}/{t_dn_min:.2f}) R2_MIN={r2_min:.2f} "
              f"IndustryConfirm={'on' if use_ind else 'off'}")

    for i, t in enumerate(idx):
        if _dbg(cfg) and (i % every) == 0:
            N = int(resid.iloc[i].notna().sum())
            day_counts = {
                "cross_up": int(cross_up.iloc[i].sum()),
                "cross_dn": int(cross_dn.iloc[i].sum()),
                "anchor_up": int(anchor_up.iloc[i].sum()),
                "anchor_dn": int(anchor_dn.iloc[i].sum()),
                "accel_size": int(accel_size_pass.iloc[i].sum()),
                "accel_pos": int(accel_sign_pos.iloc[i].sum()),
                "accel_neg": int(accel_sign_neg.iloc[i].sum()),
                "trend_ok_L": int(trend_ok_long.iloc[i].sum()),
                "trend_ok_S": int(trend_ok_short.iloc[i].sum()),
                "ind_ok_L": int(ind_ok_L.iloc[i].sum()),
                "ind_ok_S": int(ind_ok_S.iloc[i].sum()),
            }
            # quick distribution check for t-stat and R²
            ts_row = tstat_df.iloc[i].dropna()
            r2_row = r2_df.iloc[i].dropna()
            if len(ts_row):
                q_ts = np.quantile(ts_row, [0.25, 0.5, 0.75]).tolist()
            else:
                q_ts = [np.nan, np.nan, np.nan]
            if len(r2_row):
                q_r2 = np.quantile(r2_row, [0.25, 0.5, 0.75]).tolist()
            else:
                q_r2 = [np.nan, np.nan, np.nan]
            _log(cfg, f"{t.date()} sanity N={N} {day_counts} "
                      f"tstat_q25/50/75={q_ts[0]:.2f}/{q_ts[1]:.2f}/{q_ts[2]:.2f} "
                      f"R2_q25/50/75={q_r2[0]:.2f}/{q_r2[1]:.2f}/{q_r2[2]:.2f}")

        if i == 0:
            L0 = long_guard.iloc[i].fillna(False).to_numpy()
            S0 = (short_guard.iloc[i].fillna(False).to_numpy()) & (~L0)
            if L0.any():
                state.iloc[i, L0] = "Inflection"
                ttl.iloc[i, L0] = ttl_L
            if S0.any():
                state.iloc[i, S0] = "Breakdown"
                ttl.iloc[i, S0] = ttl_S
            continue

        prev_ttl = ttl.iloc[i - 1].to_numpy()
        prev_state = state.iloc[i - 1].to_numpy()
        ttl_today = np.maximum(prev_ttl - 1, 0)
        cooldown_today = np.maximum(cooldown.iloc[i - 1].to_numpy() - 1, 0)

        # carry active labels
        row_state = np.array([""] * len(cols), dtype=object)
        carry = ttl_today > 0
        if carry.any():
            row_state[carry] = prev_state[carry]

        # new triggers only if not active and not cooling
        can_trigger = (ttl_today == 0) & (cooldown_today == 0)
        nl = long_guard.iloc[i].fillna(False).to_numpy() & can_trigger
        ns = short_guard.iloc[i].fillna(False).to_numpy() & can_trigger & (~nl)

        if nl.any():
            row_state[nl] = "Inflection"
            ttl_today[nl] = ttl_L
        if ns.any():
            row_state[ns] = "Breakdown"
            ttl_today[ns] = ttl_S

        # start cooldown where trades ended today
        ended = (ttl_today == 0) & (prev_ttl > 0)
        cooldown_today[ended] = cdn

        ttl.iloc[i] = ttl_today
        cooldown.iloc[i] = cooldown_today
        state.iloc[i] = row_state

    return {"State": state, "TTL_Rem": ttl}


def compute_emergent_daily(emergent_ts: Dict[str, pd.DataFrame]) -> Tuple[pd.Series, pd.Series]:
    state = emergent_ts["State"]
    ttl = emergent_ts["TTL_Rem"]
    last = state.index[-1]
    today = state.loc[last].copy()
    today_ttl = ttl.loc[last].copy()
    today[today_ttl == 0] = ""
    return today.fillna(""), today_ttl.fillna(0).astype(int)


def backtest_emergent(resid: pd.DataFrame, emergent_ts: Dict[str, pd.DataFrame], cfg: Config) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Equal-weight across active Inflection/Breakdown, using EMERGENT_*_BUDGET.
    Execute t+1. Arithmetic compounding from residual log returns.
    Trade stats use actual TTL paths (supports asymmetric TTL).
    """
    state = emergent_ts["State"]
    ttl = emergent_ts["TTL_Rem"]
    idx, cols = state.index, list(state.columns)

    long_budget = float(cfg.EMERGENT_LONG_BUDGET)
    short_budget = float(cfg.EMERGENT_SHORT_BUDGET)

    # daily sleeve weights
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    for t in idx:
        row = state.loc[t]
        L = (row == "Inflection")
        S = (row == "Breakdown")
        nL, nS = int(L.sum()), int(S.sum())
        if long_budget > 0.0 and nL > 0:
            W.loc[t, L] = long_budget / nL
        if short_budget > 0.0 and nS > 0:
            W.loc[t, S] = -short_budget / nS

    # t+1 execution
    W_lag = W.shift(1).fillna(0.0)
    ar = np.expm1(resid.fillna(0.0))
    port = (W_lag.reindex_like(ar) * ar).sum(axis=1)
    eq = (1.0 + port).cumprod()
    dd = (eq / eq.cummax()) - 1.0

    # Trade-level stats using actual TTL paths
    trades: List[float] = []
    for c in cols:
        st_col = state[c].fillna("")
        ttl_col = ttl[c].fillna(0).astype(int)
        i = 0
        while i < len(idx):
            if st_col.iloc[i] == "Inflection" and ttl_col.iloc[i] > 0:
                j = i
                while j + 1 < len(idx) and ttl_col.iloc[j + 1] > 0 and st_col.iloc[j + 1] == "Inflection":
                    j += 1
                r = float(resid[c].iloc[i + 1:j + 1].sum()) if j >= i + 1 else 0.0
                trades.append(r)  # long wins if > 0
                i = j + 1
                continue
            if st_col.iloc[i] == "Breakdown" and ttl_col.iloc[i] > 0:
                j = i
                while j + 1 < len(idx) and ttl_col.iloc[j + 1] > 0 and st_col.iloc[j + 1] == "Breakdown":
                    j += 1
                r = float(resid[c].iloc[i + 1:j + 1].sum()) if j >= i + 1 else 0.0
                trades.append(-r)  # short wins if > 0
                i = j + 1
                continue
            i += 1

    wins = [x for x in trades if x > 0]
    losses = [abs(x) for x in trades if x <= 0]
    hit = (len(wins) / len(trades)) if trades else float("nan")
    slug = (np.mean(wins) / np.mean(losses)) if (wins and losses) else float("nan")

    stats = {
        "Sharpe_ann": float(port.mean() / port.std(ddof=0) * (252 ** 0.5)) if port.std(ddof=0) > 0 else float("nan"),
        "MaxDD": float(dd.min()) if len(dd) else float("nan"),
        "Trade_Count": int(len(trades)),
        "Trade_HitRate": float(hit),
        "Trade_Slugging": float(slug),
        "Trade_WinMean": float(np.mean(wins)) if wins else float("nan"),
        "Trade_LossMeanAbs": float(np.mean(losses)) if losses else float("nan"),
    }
    return eq, stats
