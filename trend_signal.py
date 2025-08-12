from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from config import Config

def compute_trend_daily(RS_pct: pd.DataFrame, cfg: Config) -> pd.Series:
    last = RS_pct.index[-1]
    row = RS_pct.loc[last]
    longs = row >= (1.0 - cfg.TILT_TOP_PCT)
    shorts = row <= cfg.TILT_BOT_PCT
    out = pd.Series("None", index=RS_pct.columns)
    out.loc[longs] = "Long"
    out.loc[shorts] = "Short"
    return out

def backtest_trend_tilt(resid: pd.DataFrame, RS: pd.DataFrame, cfg: Config) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Cross-sectional tilt backtest with skip/hold/frequency (cfg.BACKTEST_*).
    No costs. Equal-weight per side.

    FIXED:
      - Execute with a 1-day lag on weights
      - Compound arithmetic portfolio returns (from residual log returns)
    """
    freq = cfg.BACKTEST_REBALANCE_FREQ_D
    skip = cfg.BACKTEST_SKIP_D
    hold = cfg.BACKTEST_HOLD_D
    top_q = cfg.BACKTEST_TILT_TOP_PCT
    bot_q = cfg.BACKTEST_TILT_BOT_PCT

    RS_pct = RS.rank(axis=1, pct=True)
    dates = RS_pct.index
    if RS.first_valid_index() is None:
        return pd.Series(dtype=float), {}

    # Rebalance (signal) dates
    start_pos = dates.get_loc(RS.first_valid_index())
    signal_dates = dates[start_pos::freq]

    # Build cohort entries/exits
    entries, exits = [], []
    for t in signal_dates:
        ti = dates.get_loc(t)
        ei = ti + 1 + skip
        xi = ei + hold
        if xi < len(dates):
            entries.append(dates[ei])
            exits.append(dates[xi])

    # Weights over the holding windows
    W = pd.DataFrame(0.0, index=dates, columns=RS.columns)
    for sig_t, e_t, x_t in zip(signal_dates[:len(entries)], entries, exits):
        row = RS_pct.loc[sig_t]
        longs = row >= (1.0 - top_q)
        shorts = row <= bot_q
        if longs.sum() > 0:
            W.loc[e_t:x_t, longs] += 0.5 / float(longs.sum())
        if shorts.sum() > 0:
            W.loc[e_t:x_t, shorts] -= 0.5 / float(shorts.sum())

    # 1) 1-day lag
    W_lag = W.shift(1).fillna(0.0)

    # 2) arithmetic returns
    ar = np.expm1(resid.fillna(0.0))

    # 3) portfolio return and compounded equity
    port_ret = (W_lag.reindex_like(ar).fillna(0.0) * ar).sum(axis=1)
    eq = (1.0 + port_ret).cumprod()
    dd = (eq / eq.cummax()) - 1.0

    # --- Trade-level stats (kept on residual-sum convention)
    trade_pnl_signed: List[float] = []
    for sig_t, e_t, x_t in zip(signal_dates[:len(entries)], entries, exits):
        row = RS_pct.loc[sig_t]
        longs = row[row >= (1.0 - top_q)].index.tolist()
        shorts = row[row <= bot_q].index.tolist()
        start_i = resid.index.get_loc(e_t)
        end_i = resid.index.get_loc(x_t)

        for c in longs:
            ret_sum = float(resid[c].iloc[start_i+1:end_i+1].sum()) if end_i > start_i else 0.0
            trade_pnl_signed.append(ret_sum)  # long: positive good
        for c in shorts:
            ret_sum = float(resid[c].iloc[start_i+1:end_i+1].sum()) if end_i > start_i else 0.0
            trade_pnl_signed.append(-ret_sum) # short: negative residual is good → sign-flipped

    wins = [x for x in trade_pnl_signed if x > 0]
    losses = [abs(x) for x in trade_pnl_signed if x <= 0]
    trade_hit = (len(wins) / len(trade_pnl_signed)) if trade_pnl_signed else float("nan")
    trade_slug = (np.mean(wins) / np.mean(losses)) if (wins and losses) else float("nan")

    stats = {
        "Sharpe_ann": float(port_ret.mean() / port_ret.std(ddof=0) * (252 ** 0.5)) if port_ret.std(ddof=0) > 0 else float("nan"),
        "MaxDD": float(dd.min()),
        "Trade_Count": int(len(trade_pnl_signed)),
        "Trade_HitRate": float(trade_hit),
        "Trade_Slugging": float(trade_slug),
        "Trade_WinMean": float(np.mean(wins)) if wins else float("nan"),
        "Trade_LossMeanAbs": float(np.mean(losses)) if losses else float("nan"),
    }
    return eq, stats
