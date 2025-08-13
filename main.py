# main.py
import time
from config import load_config
from data_loader import load_trend_input, load_valuation, load_industry_map
from feature_engine import build_features, compute_unpredictable_flag
from trend_signal import compute_trend_time_series
from emergent_signal import compute_emergent_time_series, compute_emergent_daily
from stars_signal import compute_stars_time_series, compute_stars_daily
from backtests import run_all_backtests
from reporting import write_signals_daily, write_backtests_summary


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def main():
    log("main starting...")

    # 1) Config
    cfg = load_config()  # reads Trend_Input.xlsm
    log(f"Config loaded. TREND_INPUT_PATH={cfg.TREND_INPUT_PATH}")

    # 2) Data
    prices, betas = load_trend_input(cfg.TREND_INPUT_PATH)
    log(f"Prices shape={prices.shape}, dates {prices.index.min().date()}..{prices.index.max().date()}")
    valuation = load_valuation(cfg.VALUATION_WORKBOOK, cfg.VALUATION_SHEET, prices.index, list(prices.columns))
    log("Valuation frame loaded" if not valuation.empty else "No valuation data (using neutral 0.5).")
    industry_map = load_industry_map(cfg.TREND_INPUT_PATH)
    if industry_map is not None and not industry_map.empty:
        log(f"Industry map loaded with {industry_map.nunique()} groups.")
    else:
        log("No industry map found (industry confirm off or ignored).")

    # 3) Features
    log("Building features...")
    t0 = time.time()
    feats = build_features(prices, betas, valuation, cfg)
    log(f"Features ready in {time.time() - t0:.2f}s")

    # 4) Signals (time series)
    log("Computing Trend time series...")
    trend_ts = compute_trend_time_series(feats, cfg)

    log("Computing Emergent time series...")
    emergent_ts = compute_emergent_time_series(
        feats["RS_med_pct"], feats["Resid"], feats["Pseudo"], industry_map, cfg
    )

    log("Computing Stars time series...")
    star_ts = compute_stars_time_series(feats["RS_pct"], cfg)

    # 5) Daily snapshots for the Signals workbook
    log("Preparing daily snapshots...")
    emergent_daily, emergent_ttl_daily = compute_emergent_daily(emergent_ts)
    stars_daily = compute_stars_daily(star_ts)
    unpredictable = compute_unpredictable_flag(feats, cfg)
    tickers = list(prices.columns)

    # 6) Write Signals workbook
    log(f"Writing signals → {cfg.OUTPUT_SIGNALS_PATH} ...")
    write_signals_daily(
        tickers, feats, trend_ts, emergent_daily, emergent_ttl_daily, stars_daily, unpredictable, cfg
    )
    log("Signals workbook written.")

    # 7) Backtests
    log("Running backtests...")
    bt = run_all_backtests(feats, trend_ts, emergent_ts, star_ts, cfg)
    for name, res in bt.items():
        st = res.get("Stats", {})
        if st:
            sharpe = st.get("Sharpe_ann", float("nan"))
            maxdd = st.get("MaxDD", float("nan"))
            try:
                log(f"{name}: Sharpe={float(sharpe):.3f} MaxDD={float(maxdd):.3f}")
            except Exception:
                log(f"{name}: Stats={st}")

    # 8) Write Backtests summary
    log(f"Writing backtests summary → {cfg.OUTPUT_BACKTESTS_PATH} ...")
    write_backtests_summary(bt, cfg)
    log("Backtests summary written.")

    log("main done.")


if __name__ == "__main__":
    main()
