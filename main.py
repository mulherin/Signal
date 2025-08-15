import time
import pandas as pd

from config import load_config
from data_loader import load_trend_input, load_valuation, load_industry_map
from feature_engine import build_features, compute_unpredictable_flag
from trend_signal import compute_trend_time_series
from emergent_signal import compute_emergent_time_series, compute_emergent_daily
from stars_signal import compute_stars_time_series, compute_stars_daily
from backtests import run_all_backtests
from reporting import write_signals_daily, write_backtests_summary

# Stop overlay (chronology-safe) for Emergent
from emergent_stop import apply_emergent_stop_overlays

# For computing consecutive-days "Age" of live labels
from utils import current_run_length


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _compute_emergent_age_series_after_overlay(emergent_ts: dict) -> pd.Series:
    """
    Compute 'age in days' for the *current* Emergent label per ticker,
    using the post-overlay State.

    We intentionally use the new strings ("BuyDip"/"FadeRally") but also
    accept legacy ("Inflection"/"Breakdown") for safety.
    """
    state = emergent_ts["State"]
    if state.empty:
        return pd.Series(dtype=int)

    # Flags for each side (accept new & legacy strings)
    is_long = state.eq("BuyDip") | state.eq("Inflection")
    is_short = state.eq("FadeRally") | state.eq("Breakdown")

    # Run-length of True per column
    age_long = current_run_length(is_long)
    age_short = current_run_length(is_short)

    # At the last date, pick the side that's live; else zero
    last = state.index[-1]
    last_row = state.loc[last]

    out = pd.Series(0, index=state.columns, dtype=int)
    if (last_row == "BuyDip").any() or (last_row == "Inflection").any():
        mask = last_row.eq("BuyDip") | last_row.eq("Inflection")
        out.loc[mask] = age_long.loc[last, mask].astype(int)
    if (last_row == "FadeRally").any() or (last_row == "Breakdown").any():
        mask = last_row.eq("FadeRally") | last_row.eq("Breakdown")
        out.loc[mask] = age_short.loc[last, mask].astype(int)

    return out


def main():
    log("main starting...")

    # 1) Config ----------------------------------------------------------------
    cfg = load_config()  # reads Trend_Input.xlsm
    log(f"Config loaded. TREND_INPUT_PATH={cfg.TREND_INPUT_PATH}")

    # 2) Data -------------------------------------------------------------------
    prices, betas = load_trend_input(cfg.TREND_INPUT_PATH)
    log(f"Prices shape={prices.shape}, dates {prices.index.min().date()}..{prices.index.max().date()}")

    valuation = load_valuation(cfg.VALUATION_WORKBOOK, cfg.VALUATION_SHEET, prices.index, list(prices.columns))
    log("Valuation frame loaded" if not valuation.empty else "No valuation data (using neutral 0.5).")

    industry_map = load_industry_map(cfg.TREND_INPUT_PATH)
    if industry_map is not None and not industry_map.empty:
        log(f"Industry map loaded with {industry_map.nunique()} groups.")
    else:
        log("No industry map found (industry confirm off or ignored).")

    # 3) Features ---------------------------------------------------------------
    log("Building features...")
    t0 = time.time()
    feats = build_features(prices, betas, valuation, cfg)
    log(f"Features ready in {time.time() - t0:.2f}s")

    # 4) Time-series signals ----------------------------------------------------
    log("Computing Trend time series...")
    trend_ts = compute_trend_time_series(feats, cfg)  # returns dict with Tstat/Slope/R2/Score/Class 

    log("Computing Emergent time series...")
    # NOTE: emergent_signal expects RS_med_pct, Resid, Pseudo (+ industry map, cfg)
    emergent_ts_pre = compute_emergent_time_series(
        feats["RS_med_pct"], feats["Resid"], feats["Pseudo"], industry_map, cfg
    )  # compatible with your earlier API 

    log("Applying Emergent stop overlays...")
    emergent_ts = apply_emergent_stop_overlays(emergent_ts_pre, feats, trend_ts, cfg)  # idempotent & chronology-safe :contentReference[oaicite:5]{index=5}
    # Lightweight diff logging (may skip on shape mismatch)
    try:
        pre_state = emergent_ts_pre["State"]
        post_state = emergent_ts["State"]
        killed_days = int((pre_state.ne(post_state)).sum().sum())
        log(f"Emergent stop overlay applied. killed_days={killed_days}")
    except Exception:
        pass

    log("Computing Stars time series...")
    star_ts = compute_stars_time_series(feats["RS_pct"], cfg)

    # 5) Daily snapshots for the Signals workbook --------------------------------
    log("Preparing daily snapshots...")
    emergent_daily, emergent_ttl_daily_original = compute_emergent_daily(emergent_ts)  # returns (labels, TTL); TTL is zero in the new design 

    # Replace the zero TTL with a real "Age in days" computed from *post-overlay* state
    emergent_age_daily = _compute_emergent_age_series_after_overlay(emergent_ts)
    emergent_ttl_daily = emergent_age_daily  # reuse the existing column in the workbook with real age

    stars_daily = compute_stars_daily(star_ts)
    unpredictable = compute_unpredictable_flag(feats, cfg)
    tickers = list(prices.columns)

    # 6) Write Signals workbook --------------------------------------------------
    log(f"Writing signals → {cfg.OUTPUT_SIGNALS_PATH} ...")
    # Writer expects: (tickers, features, trend_ts, emergent_daily, emergent_ttl_daily, stars_daily, unpredictable, cfg)
    # We pass 'emergent_ttl_daily' = Age so you can see how old the current label is without changing reporting.py. 
    write_signals_daily(
        tickers, feats, trend_ts, emergent_daily, emergent_ttl_daily, stars_daily, unpredictable, cfg
    )
    log("Signals workbook written.")

    # 7) Backtests ---------------------------------------------------------------
    log("Running backtests...")
    bt = run_all_backtests(feats, trend_ts, emergent_ts, star_ts, cfg)  # overlay is defensively re-applied inside backtests too 
    for name, res in bt.items():
        st = res.get("Stats", {})
        if st:
            sharpe = st.get("Sharpe_ann", float("nan"))
            maxdd = st.get("MaxDD", float("nan"))
            try:
                log(f"{name}: Sharpe={float(sharpe):.3f} MaxDD={float(maxdd):.3f}")
            except Exception:
                log(f"{name}: Stats={st}")

    # 8) Backtests summary workbook ---------------------------------------------
    log(f"Writing backtests summary → {cfg.OUTPUT_BACKTESTS_PATH} ...")
    write_backtests_summary(bt, cfg)
    log("Backtests summary written.")

    log("main done.")


if __name__ == "__main__":
    main()
