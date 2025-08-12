from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from config import Config

def compute_stars_time_series(RS_pct: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    look = cfg.STAR_LOOKBACK_D
    thr = cfg.STAR_RS_THRESH_PCT
    rs = RS_pct.copy()
    long_share = (rs >= thr).rolling(window=look, min_periods=look).mean()
    short_share = (rs <= (1.0 - thr)).rolling(window=look, min_periods=look).mean()
    label = pd.DataFrame("", index=rs.index, columns=rs.columns)
    label[(long_share >= cfg.STAR_SUSTAIN_FRAC)] = "Star-Long"
    label[(short_share >= cfg.STAR_SUSTAIN_FRAC)] = "Star-Short"
    return label

def compute_stars_daily(star_ts: pd.DataFrame) -> pd.Series:
    last = star_ts.index[-1]
    return star_ts.loc[last].fillna("")

def backtest_stars(resid: pd.DataFrame, star_ts: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Portfolio equity: equal-weight Star-Long vs Star-Short.

    FIXED:
      - Execute at t+1 using lagged weights
      - Compound arithmetic portfolio returns from residual log returns
    """
    idx, cols = resid.index, resid.columns

    # Daily portfolio weights from labels
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    for t in star_ts.index:
        row = star_ts.loc[t]
        L = row == "Star-Long"
        S = row == "Star-Short"
        nL, nS = int(L.sum()), int(S.sum())
        if nL > 0:
            W.loc[t, L] = 0.5 / nL
        if nS > 0:
            W.loc[t, S] = -0.5 / nS

    # 1) 1-day lag
    W_lag = W.shift(1).fillna(0.0)

    # 2) arithmetic residual returns
    ar = np.expm1(resid.fillna(0.0))

    # 3) portfolio daily return and compounded equity
    port = (W_lag.reindex_like(ar) * ar).sum(axis=1)
    eq = (1.0 + port).cumprod()
    dd = (eq / eq.cummax()) - 1.0

    # Trade-level episodes (unchanged)
    trade_pnl_signed: List[float] = []
    for c in cols:
        lab = star_ts[c].fillna("")
        i = 0
        while i < len(lab):
            if lab.iloc[i] == "Star-Long":
                j = i
                while j+1 < len(lab) and lab.iloc[j+1] == "Star-Long":
                    j += 1
                ret = float(resid[c].iloc[i+1:j+1].sum()) if j >= i+1 else 0.0
                trade_pnl_signed.append(ret)  # long wins if >0
                i = j + 1
                continue
            if lab.iloc[i] == "Star-Short":
                j = i
                while j+1 < len(lab) and lab.iloc[j+1] == "Star-Short":
                    j += 1
                ret = float(resid[c].iloc[i+1:j+1].sum()) if j >= i+1 else 0.0
                trade_pnl_signed.append(-ret)  # short wins if >0
                i = j + 1
                continue
            i += 1

    wins = [x for x in trade_pnl_signed if x > 0]
    losses = [abs(x) for x in trade_pnl_signed if x <= 0]
    trade_hit = (len(wins) / len(trade_pnl_signed)) if trade_pnl_signed else float("nan")
    trade_slug = (np.mean(wins) / np.mean(losses)) if (wins and losses) else float("nan")

    stats = {
        "Sharpe_ann": float(port.mean() / port.std(ddof=0) * (252 ** 0.5)) if port.std(ddof=0) > 0 else float("nan"),
        "MaxDD": float(dd.min()),
        "Trade_Count": int(len(trade_pnl_signed)),
        "Trade_HitRate": float(trade_hit),
        "Trade_Slugging": float(trade_slug),
        "Trade_WinMean": float(np.mean(wins)) if wins else float("nan"),
        "Trade_LossMeanAbs": float(np.mean(losses)) if losses else float("nan"),
    }
    return eq, stats
