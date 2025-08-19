# main.py
# Streamlined runner wired to the new path and simplified pipeline:
# Data → Features → Trend → Emergent (Trend-gated) → Stars → Reporting → Backtests.
# - Uses signals_input.xlsm as the default input workbook.
# - Removes Emergent stop overlays; gating is done strictly by Trend.Class.
# - Renames TTL to a clear, user-facing "trade_Age_D".
# - Keeps Stars as a first-class label.
#
# References:
#   config.load_config (reads workbook config) :contentReference[oaicite:0]{index=0}
#   feature_engine.build_features / compute_unpredictable_flag :contentReference[oaicite:1]{index=1}
#   trend_signal.compute_trend_time_series :contentReference[oaicite:2]{index=2}
#   emergent_signal.compute_emergent_time_series (raw, ungated) :contentReference[oaicite:3]{index=3}
#   stars_signal.compute_stars_time_series / compute_stars_daily :contentReference[oaicite:4]{index=4}
#   reporting.write_signals_daily / write_backtests_summary :contentReference[oaicite:5]{index=5}
#   backtests.run_all_backtests (applies the same gating for BTs) 
#   utils.current_run_length (used here to compute trade_Age_D) :contentReference[oaicite:6]{index=6}

from __future__ import annotations

from pathlib import Path
from typing import Dict
import time
import numpy as np
import pandas as pd

from config import load_config, Config
from data_loader import load_trend_input, load_valuation, load_industry_map
from feature_engine import build_features, compute_unpredictable_flag  # :contentReference[oaicite:7]{index=7}
from trend_signal import compute_trend_time_series  # :contentReference[oaicite:8]{index=8}
from emergent_signal import compute_emergent_time_series  # :contentReference[oaicite:9]{index=9}
from stars_signal import compute_stars_time_series, compute_stars_daily  # :contentReference[oaicite:10]{index=10}
from backtests import run_all_backtests
from reporting import write_signals_daily, write_backtests_summary  # :contentReference[oaicite:11]{index=11}
from utils import current_run_length  # :contentReference[oaicite:12]{index=12}


# ---------- runtime config ----------

# Default location of your input workbook on Windows (override by editing below if needed)
DEFAULT_INPUT_PATH = Path(r"C:\Users\TaylorMulherin\Documents\Signals\Signals_Script\signals_input.xlsm")


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ---------- emergent helpers (normalize, gate, compute age) ----------

def _normalize_emergent_labels(state: pd.DataFrame) -> pd.DataFrame:
    """
    Map legacy labels to canonical ones and ensure empty string for 'no signal'.
    Legacy: 'Inflection' -> 'BuyDip', 'Breakdown' -> 'FadeRally'.
    """
    S = state.astype(str).fillna("")
    S = S.replace({"Inflection": "BuyDip", "Breakdown": "FadeRally"})
    S[~S.isin(["", "BuyDip", "FadeRally"])] = ""
    return S

# --- add near other helpers in main.py ---
def _apply_min_hold(state: pd.DataFrame,
                    trend_class: pd.DataFrame,
                    min_hold: int) -> pd.DataFrame:
    if min_hold <= 1 or state.empty:
        return state.reindex_like(state).fillna("")
    idx, cols = state.index, state.columns
    out = pd.DataFrame("", index=idx, columns=cols, dtype=object)

    def _ok(tag: str, cls: str) -> bool:
        return (tag == "BuyDip" and cls == "Onside") or (tag == "FadeRally" and cls == "Offside")

    for c in cols:
        cur = ""
        days_left = 0
        for t in idx:
            cls = str(trend_class.at[t, c]) if (t in trend_class.index and c in trend_class.columns) else ""
            trig = str(state.at[t, c]) if (t in state.index and c in state.columns) else ""
            if trig and _ok(trig, cls):
                cur = trig
                days_left = max(0, min_hold - 1)
            elif days_left > 0 and _ok(cur, cls):
                days_left -= 1
            else:
                cur = ""
                days_left = 0
            out.at[t, c] = cur
    return out

def _gate_emergent_by_trend(state: pd.DataFrame, trend_class: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce human logic with robust canonicalization:
      - BuyDip only while Trend.Class == 'Onside'
      - FadeRally only while Trend.Class == 'Offside'
    Anything else → blank.
    """
    idx = state.index.intersection(trend_class.index)
    cols = state.columns.intersection(trend_class.columns)
    if len(idx) == 0 or len(cols) == 0:
        return pd.DataFrame("", index=state.index, columns=state.columns, dtype=object)

    S = state.loc[idx, cols].astype(str).fillna("")

    # Canonicalize Trend.Class strings to {"Onside","Offside","Monitor"}.
    def _canon(x: str) -> str:
        s = str(x).strip().lower()
        while s and s[0] in "0123456789-_. ":
            s = s[1:]
        if   s.startswith("on"):  return "Onside"
        elif s.startswith("off"): return "Offside"
        elif s.startswith("mon"): return "Monitor"
        return ""

    C_raw = trend_class.loc[idx, cols].astype(str).fillna("")
    C = C_raw.stack().map(_canon).unstack().reindex_like(C_raw)

    mask_buy  = (S == "BuyDip")    & (C == "Onside")
    mask_fade = (S == "FadeRally") & (C == "Offside")

    G = pd.DataFrame("", index=idx, columns=cols, dtype=object)
    G[mask_buy]  = "BuyDip"
    G[mask_fade] = "FadeRally"
    return G.reindex(index=state.index, columns=state.columns).fillna("")



def _compute_trade_age_daily(gated_state: pd.DataFrame) -> pd.Series:
    """
    Compute 'trade_Age_D' for the *current* day per ticker:
    - If 'BuyDip' today, return current run-length of BuyDip for that ticker.
    - If 'FadeRally' today, return current run-length of FadeRally.
    - Else 0.
    """
    if gated_state.empty:
        return pd.Series(dtype=int)

    last = gated_state.index[-1]
    row = gated_state.loc[last].fillna("")

    is_long = gated_state.eq("BuyDip")
    is_short = gated_state.eq("FadeRally")

    age_L = current_run_length(is_long)
    age_S = current_run_length(is_short)

    out = pd.Series(0, index=gated_state.columns, dtype=int)
    mask_L = row.eq("BuyDip")
    mask_S = row.eq("FadeRally")
    if mask_L.any():
        out.loc[mask_L] = age_L.loc[last, mask_L].astype(int)
    if mask_S.any():
        out.loc[mask_S] = age_S.loc[last, mask_S].astype(int)
    return out


# ---------- main ----------

def main():
    log("main starting...")

    # 1) Config ----------------------------------------------------------------
    cfg: Config = load_config(trend_input_path=DEFAULT_INPUT_PATH)  # reads Config sheet from the workbook :contentReference[oaicite:13]{index=13}
    log(f"Config loaded. TREND_INPUT_PATH={cfg.TREND_INPUT_PATH}")

    # 2) Data -------------------------------------------------------------------
    prices, betas = load_trend_input(cfg.TREND_INPUT_PATH)
    log(f"Prices shape={prices.shape}, dates {prices.index.min().date()}..{prices.index.max().date()}")

    valuation = load_valuation(cfg.VALUATION_WORKBOOK, cfg.VALUATION_SHEET, prices.index, list(prices.columns))
    log("Valuation frame loaded" if valuation is not None and not valuation.empty else "No valuation data (using neutral 0.5).")

    industry_map = load_industry_map(cfg.TREND_INPUT_PATH)
    if industry_map is not None and not industry_map.empty:
        log(f"Industry map loaded with {int(industry_map.nunique())} groups.")
    else:
        log("No industry map found (industry confirm off or ignored).")

    # 3) Features ---------------------------------------------------------------
    log("Building features...")
    t0 = time.time()
    feats: Dict[str, pd.DataFrame] = build_features(prices, betas, valuation, cfg)  # Resid, Pseudo, RS_pct, RS_med_pct, Val_pct :contentReference[oaicite:14]{index=14}
    log(f"Features ready in {time.time() - t0:.2f}s")

    # 4) Trend ------------------------------------------------------------------
    log("Computing Trend time series...")
    trend_ts = compute_trend_time_series(feats, cfg)  # returns dict with Tstat/Slope/R2/Score/Class :contentReference[oaicite:15]{index=15}

    # 5) Emergent (raw → normalize → Trend-gate) --------------------------------
    log("Computing Emergent (raw, ungated)...")
    emergent_ts_raw = compute_emergent_time_series(
        feats["RS_med_pct"], feats["Resid"], feats["Pseudo"], industry_map, cfg
    )  # API-compatible with prior versions; returns {"State": DataFrame, ...} :contentReference[oaicite:16]{index=16}

    log("Normalizing and Trend-gating Emergent labels...")
    state_raw = emergent_ts_raw.get("State", pd.DataFrame(index=feats["Resid"].index, columns=feats["Resid"].columns, data=""))
    state_norm = _normalize_emergent_labels(state_raw)
    state_gated = _gate_emergent_by_trend(state_norm, trend_ts["Class"])

    # --- DIAGNOSTIC: do we have signals recently and when was the last day? ---
    recent = 60
    cnt_raw   = (state_norm.eq("BuyDip") | state_norm.eq("FadeRally")).tail(recent).sum(axis=1)
    cnt_gated = (state_gated.eq("BuyDip") | state_gated.eq("FadeRally")).tail(recent).sum(axis=1)

    last_raw_day   = cnt_raw[cnt_raw > 0].index[-1]   if (cnt_raw > 0).any()   else None
    last_gated_day = cnt_gated[cnt_gated > 0].index[-1] if (cnt_gated > 0).any() else None

    log(f"Emergent last {recent}d: raw days-with-signal={int((cnt_raw>0).sum())}, "
        f"gated days-with-signal={int((cnt_gated>0).sum())}.")
    log(f"Most recent day with any -> raw={last_raw_day.date() if last_raw_day else 'None'}, "
        f"gated={last_gated_day.date() if last_gated_day else 'None'}")

    # --- after state_gated = _gate_emergent_by_trend(...)
    min_hold = int(getattr(cfg, "EMERGENT_MIN_HOLD_D", 1))
    state_persist = _apply_min_hold(state_gated, trend_ts["Class"], min_hold)

    # Use the PERSISTED state for the outputs
    last_day = feats["Resid"].index[-1]
    emergent_daily = state_persist.loc[last_day].fillna("").astype(str)
    trade_age_daily = _compute_trade_age_daily(state_persist)

    # And pass the persisted series to backtests to keep everything consistent
    emergent_ts_for_bt = {"State": state_persist, "Meta": {"GatedByTrend": True, "MinHoldD": min_hold}}

    # --- quick diags (last 20d and today) ---
    last = feats["Resid"].index[-1]
    raw_20 = state_norm.tail(20).isin(["BuyDip","FadeRally"]).sum().sum()
    gat_20 = state_gated.tail(20).isin(["BuyDip","FadeRally"]).sum().sum()
    raw_today_L = int((state_norm.loc[last] == "BuyDip").sum())
    raw_today_S = int((state_norm.loc[last] == "FadeRally").sum())
    gat_today_L = int((state_gated.loc[last] == "BuyDip").sum())
    gat_today_S = int((state_gated.loc[last] == "FadeRally").sum())
    log(f"Emergent (raw 20d={raw_20}, gated 20d={gat_20}) | today raw L={raw_today_L} S={raw_today_S} | gated L={gat_today_L} S={gat_today_S}")


    # Current-day emergent labels and trade age
    nL_raw = int((state_norm.iloc[-1] == "BuyDip").sum());  nS_raw = int((state_norm.iloc[-1] == "FadeRally").sum())
    nL_gate = int((state_gated.iloc[-1] == "BuyDip").sum()); nS_gate = int((state_gated.iloc[-1] == "FadeRally").sum())
    log(f"Emergent today: raw L={nL_raw} S={nS_raw} | gated L={nL_gate} S={nS_gate}")

    last_day = feats["Resid"].index[-1]
    emergent_daily = state_gated.loc[last_day].fillna("").astype(str)  # series indexed by ticker
    trade_age_daily = _compute_trade_age_daily(state_gated)

    # 6) Stars ------------------------------------------------------------------
    log("Computing Stars time series...")
    star_ts = compute_stars_time_series(feats["RS_pct"], cfg)  # :contentReference[oaicite:17]{index=17}
    stars_daily = compute_stars_daily(star_ts)  # current-day labels

    # 7) Unpredictable badge ----------------------------------------------------
    unpredictable = compute_unpredictable_flag(feats, cfg)  # simple RS autocorr badge :contentReference[oaicite:18]{index=18}

    # 8) Signals workbook -------------------------------------------------------
    tickers = list(prices.columns)
    log(f"Writing signals → {cfg.OUTPUT_SIGNALS_PATH} ...")
    # We pass trade_age_daily twice: as 'emergent_ttl_daily' (for backward compatibility) and as 'emergent_age_daily'.
    write_signals_daily(
        tickers=tickers,
        features=feats,
        trend=trend_ts,
        emergent_daily=emergent_daily,
        emergent_ttl_daily=trade_age_daily,
        stars_daily=stars_daily,
        unpredictable=unpredictable,
        cfg=cfg,
        emergent_age_daily=trade_age_daily,   # ensures column name "trade_Age_D"
        stars_ts=star_ts,                     # pass full Stars TS for 5D change
    )

    log("Signals workbook written.")

    # 9) Backtests --------------------------------------------------------------
    log("Running backtests...")
    # Provide Emergent TS as a dict with the GATED state so backtests see exactly what the user sees.
    emergent_ts_for_bt = {"State": state_gated, "Meta": {"GatedByTrend": True}}
    bt = run_all_backtests(feats, trend_ts, emergent_ts_for_bt, star_ts, cfg)

    for name, res in bt.items():
        st = res.get("Stats", {})
        if st:
            sharpe = st.get("Sharpe_ann", float("nan"))
            maxdd = st.get("MaxDD", float("nan"))
            try:
                log(f"{name}: Sharpe={float(sharpe):.3f} MaxDD={float(maxdd):.3f}")
            except Exception:
                log(f"{name}: Stats={st}")

    # 10) Backtests summary workbook -------------------------------------------
    log(f"Writing backtests summary → {cfg.OUTPUT_BACKTESTS_PATH} ...")
    write_backtests_summary(bt, cfg)
    log("Backtests summary written.")

    log("main done.")


if __name__ == "__main__":
    main()
