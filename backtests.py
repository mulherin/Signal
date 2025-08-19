# backtests.py
# New backtests module — removes stop overlays and enforces Trend-gated Emergent trades.
# - Trend: unchanged, uses trend_signal.backtest_trend (t+1, equal-weight). :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
# - Emergent: BuyDip only when Trend.Class == "Onside"; FadeRally only when Trend.Class == "Offside". :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
# - Stars: unchanged, uses stars_signal.backtest_stars. 
#
# Copy/paste ready.

from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

from config import Config
from trend_signal import backtest_trend
from stars_signal import backtest_stars


# --------------------- helpers ---------------------

# --- add near other helpers in backtests.py ---
def _apply_min_hold(state: pd.DataFrame,
                    trend_class: pd.DataFrame,
                    min_hold: int) -> pd.DataFrame:
    """
    Carry each label for at least 'min_hold' days (inclusive of entry day),
    but only while the Trend gate remains supportive:
      - BuyDip requires Trend.Class == 'Onside'
      - FadeRally requires Trend.Class == 'Offside'
    """
    if min_hold <= 1:
        return state

    idx = state.index
    cols = state.columns
    out = pd.DataFrame("", index=idx, columns=cols, dtype=object)

    def _ok(tag: str, cls: str) -> bool:
        return (tag == "BuyDip" and cls == "Onside") or (tag == "FadeRally" and cls == "Offside")

    for c in cols:
        cur = ""
        days_left = 0  # days remaining *after today*; we set to (min_hold - 1) on entry
        for t in idx:
            cls = str(trend_class.at[t, c]) if (t in trend_class.index and c in trend_class.columns) else ""
            trig = str(state.at[t, c]) if (t in state.index and c in state.columns) else ""

            if trig and _ok(trig, cls):
                # new entry (or refresh) → (min_hold) days including today
                cur = trig
                days_left = max(0, min_hold - 1)
            elif days_left > 0 and _ok(cur, cls):
                # continue carrying
                days_left -= 1
            else:
                # drop if not allowed or persistence exhausted
                cur = ""
                days_left = 0

            out.at[t, c] = cur
    return out

def _normalize_emergent_labels(state: pd.DataFrame) -> pd.DataFrame:
    """
    Map legacy labels to canonical ones and ensure empty string for 'no signal'.
    Legacy: 'Inflection'->'BuyDip', 'Breakdown'->'FadeRally'.
    """
    S = state.astype(str).fillna("")
    S = S.replace({"Inflection": "BuyDip", "Breakdown": "FadeRally"})
    S[~S.isin(["", "BuyDip", "FadeRally"])] = ""
    return S


def _gate_emergent_by_trend(state: pd.DataFrame, trend_class: pd.DataFrame) -> pd.DataFrame:
    idx = state.index.intersection(trend_class.index)
    cols = state.columns.intersection(trend_class.columns)
    if len(idx) == 0 or len(cols) == 0:
        return pd.DataFrame("", index=state.index, columns=state.columns, dtype=object)

    S = state.loc[idx, cols].astype(str).fillna("")
    C_raw = trend_class.loc[idx, cols].astype(str).fillna("")

    def _canon(x: str) -> str:
        s = str(x).strip().lower()
        while s and s[0] in "0123456789-_. ":
            s = s[1:]
        if   s.startswith("on"):  return "Onside"
        elif s.startswith("off"): return "Offside"
        elif s.startswith("mon"): return "Monitor"
        return ""

    C = C_raw.stack().map(_canon).unstack().reindex_like(C_raw)

    mask_buy  = (S == "BuyDip")    & (C == "Onside")
    mask_fade = (S == "FadeRally") & (C == "Offside")

    G = pd.DataFrame("", index=idx, columns=cols, dtype=object)
    G[mask_buy]  = "BuyDip"
    G[mask_fade] = "FadeRally"
    return G

def _weights_from_emergent_labels(labels: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Build equal-weight long/short sleeves from Emergent labels.
    - Rebalance every EMERGENT_REBALANCE_FREQ_D days (default: 1)
    - t+1 execution is applied later by shifting weights by 1 day
    - Long and short budgets taken from Config (EMERGENT_LONG_BUDGET / EMERGENT_SHORT_BUDGET)
    """
    dates = labels.index
    if len(dates) == 0:
        return pd.DataFrame(0.0, index=labels.index, columns=labels.columns)

    step = int(getattr(cfg, "EMERGENT_REBALANCE_FREQ_D", 1))
    rb = dates[::max(1, step)]
    if len(rb) == 0 or rb[-1] != dates[-1]:
        rb = rb.append(pd.Index([dates[-1]]))

    long_budget = float(getattr(cfg, "EMERGENT_LONG_BUDGET", 1.0))
    short_budget = float(getattr(cfg, "EMERGENT_SHORT_BUDGET", 0.0))

    W = pd.DataFrame(0.0, index=dates, columns=labels.columns)
    for i, t0 in enumerate(rb[:-1]):
        t1 = rb[i + 1]
        row = labels.loc[t0]
        L = row.eq("BuyDip")
        S = row.eq("FadeRally")
        nL, nS = int(L.sum()), int(S.sum())
        if nL > 0 and long_budget > 0.0:
            W.loc[t0:t1, L] = long_budget / float(nL)
        if nS > 0 and short_budget > 0.0:
            W.loc[t0:t1, S] = -short_budget / float(nS)
    return W


def _portfolio_equity_from_weights(W: pd.DataFrame, resid: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Convert daily weights to equity curve with t+1 execution and arithmetic compounding.
    - resid are daily log returns (residual); convert to arithmetic via expm1.
    - Returns equity series and stats (Sharpe_ann, MaxDD).
    """
    W = W.reindex_like(resid).fillna(0.0)
    W_lag = W.shift(1).fillna(0.0)  # t+1 execution
    ar = np.expm1(resid.fillna(0.0))
    port = (W_lag * ar).sum(axis=1)
    eq = (1.0 + port).cumprod()

    dd = (eq / eq.cummax()) - 1.0
    sharpe = float(port.mean() / port.std(ddof=0) * (252 ** 0.5)) if port.std(ddof=0) > 0 else float("nan")
    stats = {"Sharpe_ann": sharpe, "MaxDD": float(dd.min())}
    return eq, stats


def _segment_lengths(label_df: pd.DataFrame, tag: str) -> List[int]:
    """
    Lengths (in days) of contiguous runs of 'tag' across all tickers.
    Useful for sanity (median hold length).
    """
    segs: List[int] = []
    for c in label_df.columns:
        s = label_df[c].fillna("").values
        run = 0
        for x in s:
            if x == tag:
                run += 1
            else:
                if run > 0:
                    segs.append(run)
                run = 0
        if run > 0:
            segs.append(run)
    return segs


# --------------------- public API ---------------------

def run_all_backtests(features: Dict[str, pd.DataFrame],
                      trend_ts: Dict[str, pd.DataFrame],
                      emergent_ts: Dict[str, pd.DataFrame],
                      star_ts: pd.DataFrame,
                      cfg: Config) -> Dict[str, Dict[str, object]]:
    """
    Execute backtests for Trend, Emergent (Trend-gated), and Stars.
    - No stop overlays (removed).
    - Emergent labels are normalized and then strictly gated by Trend.Class.
    """
    resid = features["Resid"]

    # --- Trend (unchanged) ---
    eq_trend, st_trend = backtest_trend(resid, trend_ts, cfg)

    # --- Emergent (Trend-gated) ---
    raw_state = emergent_ts.get("State", pd.DataFrame(index=resid.index, columns=resid.columns, data=""))
    state = _normalize_emergent_labels(raw_state)
    gated = _gate_emergent_by_trend(state, trend_ts["Class"])  # enforce Onside/Offside gating

    # NEW: enforce minimum hold
    min_hold = int(getattr(cfg, "EMERGENT_MIN_HOLD_D", 1))
    gated_persist = _apply_min_hold(gated, trend_ts["Class"], min_hold)

    W_em = _weights_from_emergent_labels(gated_persist, cfg)
    eq_emergent, st_emergent = _portfolio_equity_from_weights(W_em, resid)

    # Stats based on persisted labels (so Median_Hold reflects persistence)
    segL = _segment_lengths(gated_persist, "BuyDip")
    segS = _segment_lengths(gated_persist, "FadeRally")

    if isinstance(st_emergent, dict):
        st_emergent.update({
            "Trade_Count_L": int(len(segL)),
            "Trade_Count_S": int(len(segS)),
            "Median_Hold_L": float(np.median(segL)) if segL else float("nan"),
            "Median_Hold_S": float(np.median(segS)) if segS else float("nan"),
        })

    # --- Stars (unchanged) ---
    eq_stars, st_stars = backtest_stars(resid, star_ts)

    return {
        "Trend": {"Equity": eq_trend, "Stats": st_trend},
        "Emergent": {"Equity": eq_emergent, "Stats": st_emergent},
        "Stars": {"Equity": eq_stars, "Stats": st_stars},
    }
