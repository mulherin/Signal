# stars_signal.py
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from datetime import datetime

from config import Config
from utils import demean_within_groups
from data_loader import load_industry_map


def _dbg(cfg: Config) -> bool:
    return bool(getattr(cfg, "STARS_LOG", False))

def _log(cfg: Config, msg: str) -> None:
    if _dbg(cfg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [stars] {msg}", flush=True)


def compute_stars_time_series(RS_pct: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Stars with sustained RS + (optional) industry-uniqueness confirmation.

    - Base: Star-Long if share of last STAR_LOOKBACK_D days with RS_pct >= STAR_RS_THRESH_PCT
            exceeds STAR_SUSTAIN_FRAC (and analogously for Star-Short).
    - Industry confirm (optional, soft):
        Long side: fraction of days name's RS_pct >= industry-mean RS_pct exceeds STAR_INDUSTRY_CONFIRM_FRAC
        Short side: fraction of days name's RS_pct <= industry-mean RS_pct - STAR_INDUSTRY_MARGIN exceeds STAR_INDUSTRY_CONFIRM_FRAC
    """
    look = int(cfg.STAR_LOOKBACK_D)
    thr  = float(cfg.STAR_RS_THRESH_PCT)

    rs = RS_pct.copy()

    # sustained RS shares
    long_share  = (rs >= thr).rolling(window=look, min_periods=look).mean()
    short_share = (rs <= (1.0 - thr)).rolling(window=look, min_periods=look).mean()

    # optional industry confirmation
    use_ind = bool(getattr(cfg, "STAR_USE_INDUSTRY_CONFIRM", True))  # default ON for stars
    ind_frac = float(getattr(cfg, "STAR_INDUSTRY_CONFIRM_FRAC", 0.55))
    ind_margin = float(getattr(cfg, "STAR_INDUSTRY_MARGIN", 0.0))

    if use_ind:
        ind_map = load_industry_map(cfg.TREND_INPUT_PATH)
        if ind_map is None or ind_map.empty:
            _log(cfg, "industry map not found → disabling industry confirm for stars.")
            use_ind = False
        else:
            ind_map = ind_map.reindex(rs.columns).fillna(pd.Series(range(len(rs.columns)), index=rs.columns))

    if use_ind:
        # industry-demeaned RS each day
        ind_demeaned = rs.apply(lambda row: demean_within_groups(row, ind_map), axis=1)

        # fraction of lookback with industry-beating (or lagging) RS
        ind_long_share  = (ind_demeaned >= ind_margin).rolling(window=look, min_periods=look).mean()
        ind_short_share = (ind_demeaned <= -ind_margin).rolling(window=look, min_periods=look).mean()
    else:
        ind_long_share  = pd.DataFrame(1.0, index=rs.index, columns=rs.columns)
        ind_short_share = pd.DataFrame(1.0, index=rs.index, columns=rs.columns)

    # labels
    label = pd.DataFrame("", index=rs.index, columns=rs.columns)

    cond_L = (long_share  >= float(cfg.STAR_SUSTAIN_FRAC)) & (ind_long_share  >= ind_frac)
    cond_S = (short_share >= float(cfg.STAR_SUSTAIN_FRAC)) & (ind_short_share >= ind_frac)

    label[cond_L] = "Star-Long"
    label[cond_S] = "Star-Short"

    # lightweight sanity logging
    if _dbg(cfg):
        every = max(1, int(getattr(cfg, "STARS_LOG_EVERY_D", 20)))
        for i, t in enumerate(label.index):
            if (i % every) != 0:
                continue
            N = int(rs.iloc[i].notna().sum())
            nL = int(cond_L.iloc[i].sum())
            nS = int(cond_S.iloc[i].sum())
            _log(cfg, f"{t.date()} sanity N={N} Star-Long={nL} Star-Short={nS} "
                      f"use_ind={use_ind} look={look} thr={thr:.2f}")

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

    # Trade-level stats (unchanged)
    trade_pnl_signed: List[float] = []
    for c in cols:
        lab = star_ts[c].fillna("")
        i = 0
        while i < len(lab):
            if lab.iloc[i] == "Star-Long":
                j = i
                while j + 1 < len(lab) and lab.iloc[j + 1] == "Star-Long":
                    j += 1
                ret = float(resid[c].iloc[i + 1:j + 1].sum()) if j >= i + 1 else 0.0
                trade_pnl_signed.append(ret)  # long wins if > 0
                i = j + 1
                continue
            if lab.iloc[i] == "Star-Short":
                j = i
                while j + 1 < len(lab) and lab.iloc[j + 1] == "Star-Short":
                    j += 1
                ret = float(resid[c].iloc[i + 1:j + 1].sum()) if j >= i + 1 else 0.0
                trade_pnl_signed.append(-ret)  # short wins if > 0
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
