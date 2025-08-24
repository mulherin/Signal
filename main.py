# Streamlined runner wired to the new path and simplified pipeline:
# Data → Features → Trend → Emergent (Trend-gated) → Stars → Reporting → Backtests.
# - Uses signals_input.xlsm as the default input workbook.
# - Removes Emergent stop overlays; gating is done strictly by Trend.Class.
# - Renames TTL to a clear, user-facing "trade_Age_D".
# - Keeps Stars as a first-class label.

from __future__ import annotations

from pathlib import Path
from typing import Dict
import time
import numpy as np
import pandas as pd

from config import load_config, Config
from data_loader import load_trend_input, load_valuation, load_industry_map
from feature_engine import build_features, compute_unpredictable_flag
from trend_signal import compute_trend_time_series, compute_trend_time_series_industry  # NEW import
from emergent_signal import compute_emergent_time_series
from stars_signal import compute_stars_time_series, compute_stars_daily
from backtests import run_all_backtests
from reporting import write_signals_daily, write_backtests_summary
from utils import current_run_length


# ---------- runtime config ----------

DEFAULT_INPUT_PATH = Path(r"C:\Users\TaylorMulherin\Documents\Signals\Signals_Script\signals_input.xlsm")


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# ---------- emergent helpers (normalize, gate, compute age) ----------

def _apply_recent_perf_kill(state: pd.DataFrame,
                            resid: pd.DataFrame,
                            lookback: int) -> pd.DataFrame:
    """
    Early-cancel overlay:
      - Remove 'FadeRally' if last N-day residual sum <= 0 (underperformed vs market)
      - Remove 'BuyDip'   if last N-day residual sum >= 0 (outperformed vs market)
    Uses residual log returns (already market-adjusted).
    """
    if state is None or state.empty or resid is None or resid.empty:
        return state

    # Align and compute trailing N-day sum of residual log returns
    R = resid.reindex_like(state)
    rrN = R.rolling(int(lookback), min_periods=int(lookback)).sum()

    out = state.copy()
    fade_kill = out.eq("FadeRally") & (rrN <= 0.0)
    buy_kill  = out.eq("BuyDip")    & (rrN >= 0.0)
    out[fade_kill | buy_kill] = ""
    return out

def _normalize_emergent_labels(state: pd.DataFrame) -> pd.DataFrame:
    S = state.astype(str).fillna("")
    S = S.replace({"Inflection": "BuyDip", "Breakdown": "FadeRally"})
    S[~S.isin(["", "BuyDip", "FadeRally"])] = ""
    return S

def _apply_min_hold(state: pd.DataFrame,
                    trend_class: pd.DataFrame,
                    min_hold: int,
                    max_hold: int | None = None) -> pd.DataFrame:
    if state.empty:
        return pd.DataFrame("", index=state.index, columns=state.columns, dtype=object)

    idx, cols = state.index, state.columns
    out = pd.DataFrame("", index=idx, columns=cols, dtype=object)

    def _ok(tag: str, cls: str) -> bool:
        return (tag == "BuyDip" and cls == "Onside") or (tag == "FadeRally" and cls == "Offside")

    max_hold = int(max_hold or 0)

    for c in cols:
        cur = ""
        age = 0
        for t in idx:
            cls = str(trend_class.at[t, c]) if (t in trend_class.index and c in trend_class.columns) else ""
            trig = str(state.at[t, c]) if (t in state.index and c in state.columns) else ""

            # drop if gate no longer supports current tag
            if cur and not _ok(cur, cls):
                cur = ""
                age = 0

            # start only if not already open (no refresh)
            if (not cur) and trig and _ok(trig, cls):
                cur = trig
                age = 1
            elif cur:
                age += 1
                if max_hold > 0 and age > max_hold:
                    cur = ""
                    age = 0

            out.at[t, c] = cur
    return out


def _gate_emergent_by_trend(state: pd.DataFrame, trend_class: pd.DataFrame) -> pd.DataFrame:
    idx = state.index.intersection(trend_class.index)
    cols = state.columns.intersection(trend_class.columns)
    if len(idx) == 0 or len(cols) == 0:
        return pd.DataFrame("", index=state.index, columns=state.columns, dtype=object)

    S = state.loc[idx, cols].astype(str).fillna("")

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

    # 1) Config
    cfg: Config = load_config(trend_input_path=DEFAULT_INPUT_PATH)
    log(f"Config loaded. TREND_INPUT_PATH={cfg.TREND_INPUT_PATH}")

    # 2) Data
    prices, betas = load_trend_input(cfg.TREND_INPUT_PATH)
    log(f"Prices shape={prices.shape}, dates {prices.index.min().date()}..{prices.index.max().date()}")

    valuation = load_valuation(cfg.VALUATION_WORKBOOK, cfg.VALUATION_SHEET, prices.index, list(prices.columns))
    log("Valuation frame loaded" if valuation is not None and not valuation.empty else "No valuation data (using neutral 0.5).")

    industry_map = load_industry_map(cfg.TREND_INPUT_PATH)
    if industry_map is not None and not industry_map.empty:
        log(f"Industry map loaded with {int(industry_map.nunique())} groups.")
    else:
        log("No industry map found (industry confirm off or ignored).")

    # 3) Features
    log("Building features...")
    t0 = time.time()
    feats: Dict[str, pd.DataFrame] = build_features(prices, betas, valuation, cfg)
    log(f"Features ready in {time.time() - t0:.2f}s")

    # 4) Trend (global)
    log("Computing Trend time series (global)...")
    trend_ts = compute_trend_time_series(feats, cfg)

    # 4b) Trend (industry-relative) if enabled
    trend_ts_ind = None
    if bool(getattr(cfg, "TREND_IND_ENABLED", True)):
        log("Computing Trend time series (industry-relative)...")
        trend_ts_ind = compute_trend_time_series_industry(feats, cfg)

    # 5) Emergent (raw → normalize → Trend-gate)
    log("Computing Emergent (raw, ungated)...")
    emergent_ts_raw = compute_emergent_time_series(
        feats["RS_med_pct"], feats["Resid"], feats["Pseudo"], industry_map, cfg
    )

    log("Normalizing and Trend-gating Emergent labels...")
    state_raw = emergent_ts_raw.get("State", pd.DataFrame(index=feats["Resid"].index, columns=feats["Resid"].columns, data=""))
    state_norm = _normalize_emergent_labels(state_raw)
    state_gated = _gate_emergent_by_trend(state_norm, trend_ts["Class"])

    # Persistence
    min_hold = int(getattr(cfg, "EMERGENT_MIN_HOLD_D", 1))
    max_hold = int(getattr(cfg, "EMERGENT_MAX_HOLD_D", 5))
    state_persist = _apply_min_hold(state_gated, trend_ts["Class"], min_hold, max_hold)

    # Early-cancel overlay
    kill_look = int(getattr(cfg, "EMERGENT_KILL_LOOKBACK_D", 5))
    state_final = _apply_recent_perf_kill(state_persist, feats["Resid"], kill_look)

    last_day = feats["Resid"].index[-1]
    emergent_daily = state_final.loc[last_day].fillna("").astype(str)
    trade_age_daily = _compute_trade_age_daily(state_final)


    # Early-cancel overlay: drop FadeRally if 5d underperformed; drop BuyDip if 5d outperformed
    kill_look = int(getattr(cfg, "EMERGENT_KILL_LOOKBACK_D", 5))
    state_final = _apply_recent_perf_kill(state_persist, feats["Resid"], kill_look)

    last_day = feats["Resid"].index[-1]
    emergent_daily = state_final.loc[last_day].fillna("").astype(str)
    trade_age_daily = _compute_trade_age_daily(state_final)


    # 6) Stars
    log("Computing Stars time series...")
    star_ts = compute_stars_time_series(feats["RS_pct"], cfg)
    stars_daily = compute_stars_daily(star_ts)

    # 7) Unpredictable badge
    unpredictable = compute_unpredictable_flag(feats, cfg)

    # 8) Signals workbook
    tickers = list(prices.columns)
    log(f"Writing signals → {cfg.OUTPUT_SIGNALS_PATH} ...")
    write_signals_daily(
        tickers=tickers,
        features=feats,
        trend=trend_ts,
        emergent_daily=emergent_daily,
        emergent_ttl_daily=trade_age_daily,
        stars_daily=stars_daily,
        unpredictable=unpredictable,
        cfg=cfg,
        emergent_age_daily=trade_age_daily,
        stars_ts=star_ts,
        trend_ind=trend_ts_ind,  # NEW: pass industry-relative trend block
    )
    log("Signals workbook written.")

    # 9) Backtests (unchanged)
    log("Running backtests...")
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

    # 10) Backtests summary workbook
    log(f"Writing backtests summary → {cfg.OUTPUT_BACKTESTS_PATH} ...")
    write_backtests_summary(bt, cfg)
    log("Backtests summary written.")

    log("main done.")


if __name__ == "__main__":
    main()
