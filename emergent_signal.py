from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from config import Config

# --- small helpers
def _pct_rank_rowwise(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)

def _rs_sum(resid: pd.DataFrame, look: int) -> pd.DataFrame:
    return resid.rolling(look).sum()

def compute_emergent_time_series(rs_med_pct: pd.DataFrame,
                                 resid: pd.DataFrame,
                                 accel_pct: pd.DataFrame,
                                 adx_roc: pd.DataFrame,
                                 val_pct: pd.DataFrame,   # used only for warnings/reporting
                                 cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Emergent timing alerts on residuals.

    Long (Inflection):
      RS_med_pct >= RS_long_pct + EMERGENT_LONG_CROSS_MARGIN
      RS_med_pct >= DIR_ANCHOR_LONG_PCT
      RS_short_pct >= EMERGENT_LONG_RS_SHORT_MIN_PCT
      ADX_ROC >= 0
      (valuation is a warning only; not a gate)

    Short (Breakdown):
      Accel_pct <= ACCEL_BOT_PCT
      RS_med_pct <= DIR_ANCHOR_SHORT_PCT
      ADX_ROC < 0
      RS_short_pct >= EMERGENT_SHORT_RS_SHORT_FLOOR_PCT   # avoid bottom-ticks
      (valuation is a warning only; not a gate)

    Lifecycle: TTL active days, then Cooldown; labels are blank when TTL==0.
    """

    # Build short/long RS percentiles from residuals
    RS_short_pct = _pct_rank_rowwise(_rs_sum(resid, cfg.RS_SHORT_D))
    RS_long_pct  = _pct_rank_rowwise(_rs_sum(resid, cfg.RS_LONG_D))

    # LONG crossover mask
    long_crossover = (
        (rs_med_pct >= RS_long_pct + float(cfg.EMERGENT_LONG_CROSS_MARGIN)) &
        (rs_med_pct >= float(cfg.DIR_ANCHOR_LONG_PCT)) &
        (RS_short_pct >= float(cfg.EMERGENT_LONG_RS_SHORT_MIN_PCT)) &
        (adx_roc >= 0)
    )

    # SHORT guarded accel mask
    short_guard = (
        (accel_pct <= float(cfg.ACCEL_BOT_PCT)) &
        (rs_med_pct <= float(cfg.DIR_ANCHOR_SHORT_PCT)) &
        (adx_roc < 0) &
        (RS_short_pct >= float(cfg.EMERGENT_SHORT_RS_SHORT_FLOOR_PCT))
    )

    idx, cols = rs_med_pct.index, list(rs_med_pct.columns)
    state = pd.DataFrame("", index=idx, columns=cols)
    ttl = pd.DataFrame(0, index=idx, columns=cols, dtype=int)
    cooldown = pd.DataFrame(0, index=idx, columns=cols, dtype=int)

    for i, t in enumerate(idx):
        if i == 0:
            # first row: allow triggers
            new_long = long_crossover.iloc[i].fillna(False).to_numpy()
            new_short = (short_guard.iloc[i].fillna(False).to_numpy()) & (~new_long)

            row_state = np.array([""] * len(cols), dtype=object)
            row_state[new_long] = "Inflection"
            row_state[new_short] = "Breakdown"

            state.iloc[i] = row_state
            ttl.iloc[i, new_long] = cfg.EMERGENT_TTL_D
            ttl.iloc[i, new_short] = cfg.EMERGENT_TTL_D
            continue

        # roll TTL + cooldown
        prev_ttl = ttl.iloc[i - 1].to_numpy()
        ttl_today = np.maximum(prev_ttl - 1, 0)
        cooldown_today = np.maximum(cooldown.iloc[i - 1].to_numpy() - 1, 0)

        # carry label only while active
        row_state = np.array([""] * len(cols), dtype=object)
        carry = ttl_today > 0
        if carry.any():
            row_state[carry] = state.iloc[i - 1].to_numpy()[carry]

        # new triggers only if not active and not cooling
        can_trigger = (ttl_today == 0) & (cooldown_today == 0)
        nl = (long_crossover.iloc[i].fillna(False).to_numpy()) & can_trigger
        ns = (short_guard.iloc[i].fillna(False).to_numpy()) & can_trigger & (~nl)

        ttl_today[nl] = cfg.EMERGENT_TTL_D
        ttl_today[ns] = cfg.EMERGENT_TTL_D
        row_state[nl] = "Inflection"
        row_state[ns] = "Breakdown"

        # start cooldown where trades ended today
        ended = (ttl_today == 0) & (prev_ttl > 0)
        cooldown_today[ended] = cfg.EMERGENT_COOLDOWN_D

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
    # guard: no label when TTL==0
    today[today_ttl == 0] = ""
    return today.fillna(""), today_ttl.fillna(0).astype(int)

def backtest_emergent(resid: pd.DataFrame, emergent_ts: Dict[str, pd.DataFrame], cfg: Config) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Equal-weight all active Inflection (long) and Breakdown (short). No costs.

    FIXED:
      - Execute at t+1: weights are lagged by 1 day
      - Compound arithmetic portfolio returns built from residual log returns
    """
    state = emergent_ts["State"]
    ttl = emergent_ts["TTL_Rem"]
    idx, cols = state.index, list(state.columns)

    # daily sleeve weights
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    for t in idx:
        row = state.loc[t]
        L = (row == "Inflection")
        S = (row == "Breakdown")
        nL, nS = int(L.sum()), int(S.sum())
        if nL > 0:
            W.loc[t, L] = 0.5 / nL
        if nS > 0:
            W.loc[t, S] = -0.5 / nS

    # 1) execute at t+1
    W_lag = W.shift(1).fillna(0.0)

    # 2) convert residual log returns to arithmetic residual returns
    ar = np.expm1(resid.fillna(0.0))

    # 3) daily portfolio arithmetic return and compounded equity
    port = (W_lag.reindex_like(ar) * ar).sum(axis=1)
    eq = (1.0 + port).cumprod()
    dd = (eq / eq.cummax()) - 1.0

    # trade-level stats (unchanged clock: t+1 inside the slices)
    trades: List[float] = []
    for i, _ in enumerate(idx):
        if i == 0:
            continue
        prev_ttl = ttl.iloc[i - 1]
        now_ttl = ttl.iloc[i]
        new_long = (now_ttl > 0) & (prev_ttl == 0) & (state.iloc[i] == "Inflection")
        new_short = (now_ttl > 0) & (prev_ttl == 0) & (state.iloc[i] == "Breakdown")

        for c in state.columns[new_long.to_numpy()]:
            start_i = i
            end_i = min(i + cfg.EMERGENT_TTL_D, len(idx) - 1)
            r = float(resid[c].iloc[start_i + 1:end_i + 1].sum()) if end_i > start_i else 0.0
            trades.append(r)      # long wins if >0
        for c in state.columns[new_short.to_numpy()]:
            start_i = i
            end_i = min(i + cfg.EMERGENT_TTL_D, len(idx) - 1)
            r = float(resid[c].iloc[start_i + 1:end_i + 1].sum()) if end_i > start_i else 0.0
            trades.append(-r)     # short wins if >0

    wins = [x for x in trades if x > 0]
    losses = [abs(x) for x in trades if x <= 0]
    hit = (len(wins) / len(trades)) if trades else float("nan")
    slug = (np.mean(wins) / np.mean(losses)) if (wins and losses) else float("nan")

    stats = {
        "Sharpe_ann": float(port.mean() / port.std(ddof=0) * (252 ** 0.5)) if port.std(ddof=0) > 0 else float("nan"),
        "MaxDD": float(dd.min()),
        "Trade_Count": int(len(trades)),
        "Trade_HitRate": float(hit),
        "Trade_Slugging": float(slug),
        "Trade_WinMean": float(np.mean(wins)) if wins else float("nan"),
        "Trade_LossMeanAbs": float(np.mean(losses)) if losses else float("nan"),
    }
    return eq, stats
