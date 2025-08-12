# main.py
from pathlib import Path
import pandas as pd
from config import load_config
from data_loader import load_trend_input, load_valuation
from feature_engine import build_features, compute_unpredictable_flag
from trend_signal import compute_trend_daily
from emergent_signal import compute_emergent_time_series, compute_emergent_daily
from stars_signal import compute_stars_time_series, compute_stars_daily
from backtests import run_all_backtests
from reporting import write_signals_daily, write_backtests_summary

def main() -> None:
    cfg = load_config()

    prices, betas = load_trend_input(cfg.TREND_INPUT_PATH)
    valuation_raw = load_valuation(cfg.VALUATION_WORKBOOK, cfg.VALUATION_SHEET, prices.index, list(prices.columns))

    feat = build_features(prices, betas, valuation_raw, cfg)
    # feat must include: "Resid", "RS_pct" (med horizon), "Accel_pct", "ADX_ROC", "Val_pct"

    trend_daily = compute_trend_daily(feat["RS_pct"], cfg)

    # UPDATED: pass Resid and Accel_pct so emergent can form crossover/guard masks
    emergent_ts = compute_emergent_time_series(
        feat["RS_pct"],         # RS_med_pct
        feat["Resid"],          # residual returns
        feat["Accel_pct"],      # acceleration percentile
        feat["ADX_ROC"],        # ADX_ROC
        feat["Val_pct"],        # valuation percentile (warning only)
        cfg
    )
    emergent_daily, emergent_ttl_daily = compute_emergent_daily(emergent_ts)

    star_ts = compute_stars_time_series(feat["RS_pct"], cfg)
    stars_daily = compute_stars_daily(star_ts)

    unpredictable = compute_unpredictable_flag(feat, cfg)

    tickers = list(prices.columns)
    write_signals_daily(tickers, feat, trend_daily, emergent_daily, emergent_ttl_daily, stars_daily, unpredictable, cfg)

    bt = run_all_backtests(feat, emergent_ts, star_ts, cfg)
    write_backtests_summary(bt, cfg)

if __name__ == "__main__":
    main()
