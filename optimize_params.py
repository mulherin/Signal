"""
optimize_params.py  (Emergent v2-ready)

Walk-forward hyperparameter tuning with:
- Robust fold construction
- Dynamic warmup (override with --warmup_days)
- Frequent progress prints with timestamps (--verbose to increase frequency)

Updated for Emergent v2:
- Recompute Emergent labels per-parameter (cfg_i) using new API:
  compute_emergent_time_series(rs_med_pct, resid, pseudo, industry_map, cfg_i)
- New search space: t-stat/R², anchors, cross margins, accel size gate,
  optional industry confirm, asymmetric TTLs.
"""

import argparse
import sys
import json
import time
from datetime import datetime
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pathlib import Path as _Path
if str(_Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(_Path(__file__).parent))

from config import load_config, Config
from data_loader import load_trend_input, load_valuation, load_industry_map
from feature_engine import build_features
from trend_signal import compute_trend_time_series
from emergent_signal import compute_emergent_time_series
from stars_signal import compute_stars_time_series

# ----------------------------- logging --------------------------------

def log(msg: str, verbose: bool = True):
    if not verbose:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ----------------------------- helpers --------------------------------

def _annualized_sharpe(port: pd.Series) -> float:
    if port.std(ddof=0) == 0 or port.isna().all():
        return float("nan")
    return float(port.mean() / port.std(ddof=0) * (252 ** 0.5))

def _max_drawdown(eq: pd.Series) -> float:
    if eq.empty:
        return float("nan")
    dd = (eq / eq.cummax()) - 1.0
    return float(dd.min())

def _label_segments_lengths(label_df: pd.DataFrame, tag: str, window: Tuple[pd.Timestamp,pd.Timestamp]) -> List[int]:
    start, end = window
    segs: List[int] = []
    df = label_df.loc[(label_df.index >= start) & (label_df.index <= end)].fillna("")
    for c in df.columns:
        s = df[c].values
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

def _build_weights_trend(Class: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    dates = Class.index
    step = max(1, int(cfg.TREND_REBALANCE_FREQ_D))
    rb = dates[::step]
    if len(rb) == 0 or rb[-1] != dates[-1]:
        rb = rb.insert(len(rb), dates[-1])
    W = pd.DataFrame(0.0, index=dates, columns=Class.columns)
    long_budget = float(cfg.TREND_LONG_BUDGET)
    short_budget = float(cfg.TREND_SHORT_BUDGET)

    for i, t0 in enumerate(rb[:-1]):
        t1 = rb[i + 1]
        row = Class.loc[t0]
        longs = (row == "Onside")
        shorts = (row == "Offside") & (short_budget > 0.0)
        nL = int(longs.sum()); nS = int(shorts.sum())
        if nL > 0:
            W.loc[t0:t1, longs] = long_budget / float(nL)
        if short_budget > 0.0 and nS > 0:
            W.loc[t0:t1, shorts] = -short_budget / float(nS)
    return W.shift(1).fillna(0.0)  # t+1 execution

def _build_weights_emergent(state: pd.DataFrame) -> pd.DataFrame:
    # Equal-weight long/short sleeves at 50/50 notional, t+1 execution
    W = pd.DataFrame(0.0, index=state.index, columns=state.columns)
    for t in state.index:
        row = state.loc[t]
        L = (row == "Inflection")
        S = (row == "Breakdown")
        nL, nS = int(L.sum()), int(S.sum())
        if nL > 0:
            W.loc[t, L] = 0.5 / nL
        if nS > 0:
            W.loc[t, S] = -0.5 / nS
    return W.shift(1).fillna(0.0)

def _build_weights_stars(star_ts: pd.DataFrame) -> pd.DataFrame:
    W = pd.DataFrame(0.0, index=star_ts.index, columns=star_ts.columns)
    for t in star_ts.index:
        row = star_ts.loc[t]
        L = (row == "Star-Long")
        S = (row == "Star-Short")
        nL, nS = int(L.sum()), int(S.sum())
        if nL > 0:
            W.loc[t, L] = 0.5 / nL
        if nS > 0:
            W.loc[t, S] = -0.5 / nS
    return W.shift(1).fillna(0.0)

def _net_portfolio_series(W_lag: pd.DataFrame, resid: pd.DataFrame, cost_bps: float = 0.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ar = np.expm1(resid.fillna(0.0))
    port_gross = (W_lag.reindex_like(ar) * ar).sum(axis=1)
    dW = W_lag.diff().abs().sum(axis=1).fillna(0.0)
    cost = (cost_bps / 1e4) * dW
    port_net = port_gross - cost
    eq = (1.0 + port_net).cumprod()
    return port_net, eq, dW

def _compute_dynamic_warmup(cfg: Config) -> int:
    """
    Pick a warmup that covers the longest lookback used by the signals.
    Emergent v2 depends on:
      - EMERGENT_TSTAT_LEN_D (t-stat/R²)
      - RS_MED_LOOKBACK_D (for rs_med_pct)
      - RS_LONG_D, RS_SHORT_D (for cross)
      - RS_LOOKBACK_D + ACCEL_LOOKBACK_D (accel size gate)
    """
    candidates = [
        int(getattr(cfg, "EMERGENT_TSTAT_LEN_D", 63)),
        int(getattr(cfg, "RS_MED_LOOKBACK_D", getattr(cfg, "RS_LOOKBACK_D", 126))),
        int(getattr(cfg, "RS_LONG_D", 126)),
        int(getattr(cfg, "RS_SHORT_D", 21)),
        int(getattr(cfg, "RS_LOOKBACK_D", 126)) + int(getattr(cfg, "ACCEL_LOOKBACK_D", 63)),
        126,  # floor for robustness
    ]
    return max(candidates) + 20  # small buffer

def _date_folds(idx: pd.DatetimeIndex, train_years: int, test_months: int, step_months: int, warmup_days: int) -> List[Tuple[pd.Timestamp,pd.Timestamp,pd.Timestamp,pd.Timestamp]]:
    n = len(idx)
    if n < 2:
        return []
    end = idx[-1]

    t0 = idx[min(max(warmup_days, 0), n - 1)]
    folds: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []

    while True:
        train_target = t0 + pd.DateOffset(years=train_years)
        pos_train_end = idx.searchsorted(train_target, side="right") - 1
        pos_t0 = idx.get_indexer([t0])[0]
        if pos_train_end <= pos_t0:
            break
        pos_train_end = min(max(pos_train_end, 0), n - 1)
        train_end = idx[pos_train_end]

        pos_test_start = pos_train_end + 1
        if pos_test_start >= n:
            break
        test_start = idx[pos_test_start]
        if test_start >= end:
            break

        test_target = test_start + pd.DateOffset(months=test_months)
        pos_test_end = idx.searchsorted(test_target, side="left") - 1
        if pos_test_end < pos_test_start:
            break
        pos_test_end = min(pos_test_end, n - 1)
        test_end = idx[pos_test_end]

        folds.append((t0, train_end, test_start, test_end))

        step_target = t0 + pd.DateOffset(months=step_months)
        pos_next_t0 = idx.searchsorted(step_target, side="left")
        if pos_next_t0 >= n - 1:
            break
        t0 = idx[pos_next_t0]
        if t0 >= end or len(folds) > 200:
            break

    return folds

def _median_hold_constraint_ok(lengths: List[int], min_hold: int) -> Tuple[bool, float]:
    if not lengths:
        return False, float("nan")
    med = float(np.median(lengths))
    return (med >= min_hold), med

# ----------------------------- search spaces ----------------------------

def _space_trend():
    return {
        "TREND_TSTAT_UP":  [1.0, 1.25, 1.5, 1.75, 2.0],
        "TREND_TSTAT_DOWN":[0.6, 0.8, 1.0],
        "TREND_MIN_RSPCT": [0.40, 0.50, 0.60],
        "TREND_USE_HYSTERESIS": [False, True],
        "TREND_REBALANCE_FREQ_D": [5, 10],
    }

def _space_emergent():
    """
    Emergent v2 parameter space:
    - Anchors and cross margins
    - Trend shape (t-stat window/thresholds + R²)
    - Accel size gate
    - Optional industry confirm
    - Asymmetric TTLs + cooldown
    """
    return {
        # Anchors & cross margins
        "DIR_ANCHOR_LONG_PCT": [0.60, 0.65, 0.70],
        "DIR_ANCHOR_SHORT_PCT": [0.25, 0.30, 0.35],
        "EMERGENT_LONG_CROSS_MARGIN": [0.04, 0.05, 0.06, 0.07],
        "EMERGENT_SHORT_CROSS_MARGIN": [0.04, 0.05, 0.06, 0.07],

        # Trend shape
        "EMERGENT_TSTAT_LEN_D": [42, 63, 84],
        "EMERGENT_TSTAT_MIN_UP": [0.4, 0.5, 0.6, 0.8],
        "EMERGENT_TSTAT_MIN_DN": [0.6, 0.7, 0.9, 1.1],
        "EMERGENT_R2_MIN": [0.10, 0.15, 0.20, 0.30],

        # Accel size gate
        "ACCEL_DELTA_MIN": [0.10, 0.15, 0.20],

        # Industry confirmation toggle
        "EMERGENT_USE_INDUSTRY_CONFIRM": [True, False],

        # Lifecycle (asymmetric TTLs)
        "EMERGENT_TTL_LONG_D": [20, 24, 28, 32],
        "EMERGENT_TTL_SHORT_D": [20, 24, 28],
        "EMERGENT_COOLDOWN_D": [7, 10],
    }

def _space_stars():
    return {
        "STAR_LOOKBACK_D": [189, 210, 231, 252, 273],
        "STAR_RS_THRESH_PCT": [0.60, 0.65, 0.70, 0.75, 0.80],
        "STAR_SUSTAIN_FRAC": [0.60, 0.65, 0.70, 0.75, 0.80],
    }

def _random_combinations(space: Dict[str, List], n: int, rng: np.random.Generator) -> List[Dict[str, object]]:
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    out = []
    for _ in range(n):
        choice = [rng.choice(v) for v in vals]
        d = {k: (x.item() if hasattr(x, "item") else x) for k, x in zip(keys, choice)}
        out.append(d)
    # include a deterministic "first" combo for stability
    out.append({k: v[0] for k, v in space.items()})
    return out

# ----------------------------- tune per strategy ----------------------------

def _evaluate_trend(cfg: Config, prices: pd.DataFrame, betas: pd.Series, valuation_raw: pd.DataFrame,
                    folds, min_hold: int, cost_bps: float, seed: int, samples: int, verbose: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    space = _space_trend()
    params_list = _random_combinations(space, samples, rng)

    rows = []
    log("[trend] building features", verbose)
    t0 = time.perf_counter()
    feat = build_features(prices, betas, valuation_raw, cfg)
    trend_ts = compute_trend_time_series(feat, cfg)
    Class = trend_ts["Class"]
    log(f"[trend] features ready in {time.perf_counter() - t0:.1f}s", verbose)

    N = len(params_list)
    every = max(1, N // 10)  # ~10 updates
    best_so_far = None

    for i, params in enumerate(params_list, 1):
        cfg_i = replace(cfg, **params)

        sharpe_oos = []
        mdd_oos = []
        hold_meds = []
        for (tr_start, tr_end, te_start, te_end) in folds:
            W = _build_weights_trend(Class, cfg_i)
            resid = feat["Resid"].loc[te_start:te_end]
            W_te = W.loc[te_start:te_end]
            port, eq, dW = _net_portfolio_series(W_te, resid, cost_bps=cost_bps)
            sharpe_oos.append(_annualized_sharpe(port))
            mdd_oos.append(_max_drawdown(eq))
            segs = _label_segments_lengths(Class, "Onside", (te_start, te_end))
            _, med = _median_hold_constraint_ok(segs, min_hold)
            hold_meds.append(med)

        row = {**params,
               "Sharpe_OOS_mean": float(np.nanmean(sharpe_oos)),
               "Sharpe_OOS_median": float(np.nanmedian(sharpe_oos)),
               "MaxDD_OOS_median": float(np.nanmedian(mdd_oos)),
               "Hold_Median_days": float(np.nanmedian(hold_meds)),
               "Folds": len(folds)}
        rows.append(row)

        if best_so_far is None or row["Sharpe_OOS_mean"] > best_so_far["Sharpe_OOS_mean"]:
            best_so_far = row

        if (i % every == 0) or (i <= 3) or (i == N):
            log(f"[trend] {i}/{N} combos  best Sharpe_OOS_mean={best_so_far['Sharpe_OOS_mean']:.3f}  median_hold={best_so_far['Hold_Median_days']:.1f}d", verbose)

    df = pd.DataFrame(rows).sort_values(["Sharpe_OOS_mean", "Hold_Median_days"], ascending=[False, False])
    return df

def _evaluate_emergent(cfg: Config, prices: pd.DataFrame, betas: pd.Series, valuation_raw: pd.DataFrame,
                       folds, min_hold: int, cost_bps: float, seed: int, samples: int, verbose: bool,
                       industry_map: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    space = _space_emergent()
    params_list = _random_combinations(space, samples, rng)

    rows = []
    log("[emergent] building features", verbose)
    t0 = time.perf_counter()
    feat = build_features(prices, betas, valuation_raw, cfg)
    log(f"[emergent] features ready in {time.perf_counter() - t0:.1f}s", verbose)

    N = len(params_list)
    every = max(1, N // 10)
    best_so_far = None

    for i, params in enumerate(params_list, 1):
        cfg_i = replace(cfg, **params)

        # Recompute Emergent labels for *each* parameter combo (cfg_i)
        emergent_ts = compute_emergent_time_series(
            feat["RS_med_pct"],  # median lookback RS percentile
            feat["Resid"],       # residual log returns
            feat["Pseudo"],      # residual pseudo-price (for t-stat/R²)
            industry_map,        # optional industry map; soft gate if enabled
            cfg_i,
        )
        state = emergent_ts["State"]

        sharpe_oos = []
        mdd_oos = []
        hold_meds = []
        for (tr_start, tr_end, te_start, te_end) in folds:
            W = _build_weights_emergent(state)
            resid = feat["Resid"].loc[te_start:te_end]
            W_te = W.loc[te_start:te_end]
            port, eq, dW = _net_portfolio_series(W_te, resid, cost_bps=cost_bps)
            sharpe_oos.append(_annualized_sharpe(port))
            mdd_oos.append(_max_drawdown(eq))
            segs = _label_segments_lengths(state, "Inflection", (te_start, te_end))
            _, med = _median_hold_constraint_ok(segs, min_hold)
            hold_meds.append(med)

        row = {**params,
               "Sharpe_OOS_mean": float(np.nanmean(sharpe_oos)),
               "Sharpe_OOS_median": float(np.nanmedian(sharpe_oos)),
               "MaxDD_OOS_median": float(np.nanmedian(mdd_oos)),
               "Hold_Median_days": float(np.nanmedian(hold_meds)),
               "Folds": len(folds)}
        rows.append(row)

        if best_so_far is None or row["Sharpe_OOS_mean"] > best_so_far["Sharpe_OOS_mean"]:
            best_so_far = row

        if (i % every == 0) or (i <= 3) or (i == N):
            log(f"[emergent] {i}/{N} combos  best Sharpe_OOS_mean={best_so_far['Sharpe_OOS_mean']:.3f}  median_hold={best_so_far['Hold_Median_days']:.1f}d", verbose)

    df = pd.DataFrame(rows).sort_values(["Sharpe_OOS_mean", "Hold_Median_days"], ascending=[False, False])
    return df

def _evaluate_stars(cfg: Config, prices: pd.DataFrame, betas: pd.Series, valuation_raw: pd.DataFrame,
                    folds, min_hold: int, cost_bps: float, seed: int, samples: int, verbose: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    space = _space_stars()
    params_list = _random_combinations(space, samples, rng)

    rows = []
    log("[stars] building features", verbose)
    t0 = time.perf_counter()
    feat = build_features(prices, betas, valuation_raw, cfg)
    star_ts = compute_stars_time_series(feat["RS_pct"], cfg)
    log(f"[stars] features ready in {time.perf_counter() - t0:.1f}s", verbose)

    N = len(params_list)
    every = max(1, N // 10)
    best_so_far = None

    for i, params in enumerate(params_list, 1):
        cfg_i = replace(cfg, **params)

        sharpe_oos = []
        mdd_oos = []
        hold_meds = []
        for (tr_start, tr_end, te_start, te_end) in folds:
            W = _build_weights_stars(star_ts)
            resid = feat["Resid"].loc[te_start:te_end]
            W_te = W.loc[te_start:te_end]
            port, eq, dW = _net_portfolio_series(W_te, resid, cost_bps=cost_bps)
            sharpe_oos.append(_annualized_sharpe(port))
            mdd_oos.append(_max_drawdown(eq))
            segs = _label_segments_lengths(star_ts, "Star-Long", (te_start, te_end))
            _, med = _median_hold_constraint_ok(segs, min_hold)
            hold_meds.append(med)

        row = {**params,
               "Sharpe_OOS_mean": float(np.nanmean(sharpe_oos)),
               "Sharpe_OOS_median": float(np.nanmedian(sharpe_oos)),
               "MaxDD_OOS_median": float(np.nanmedian(mdd_oos)),
               "Hold_Median_days": float(np.nanmedian(hold_meds)),
               "Folds": len(folds)}
        rows.append(row)

        if best_so_far is None or row["Sharpe_OOS_mean"] > best_so_far["Sharpe_OOS_mean"]:
            best_so_far = row

        if (i % every == 0) or (i <= 3) or (i == N):
            log(f"[stars] {i}/{N} combos  best Sharpe_OOS_mean={best_so_far['Sharpe_OOS_mean']:.3f}  median_hold={best_so_far['Hold_Median_days']:.1f}d", verbose)

    df = pd.DataFrame(rows).sort_values(["Sharpe_OOS_mean", "Hold_Median_days"], ascending=[False, False])
    return df

# ----------------------------- main -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["trend", "emergent", "stars"], required=True)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--train_years", type=int, default=3)
    ap.add_argument("--test_months", type=int, default=6)
    ap.add_argument("--step_months", type=int, default=3)
    ap.add_argument("--min_hold", type=int, default=21)
    ap.add_argument("--cost_bps", type=float, default=5.0)
    ap.add_argument("--warmup_days", type=int, default=None, help="Override dynamic warmup. If not set, auto-compute from Config.")
    ap.add_argument("--verbose", action="store_true", help="Print frequent progress updates")
    args = ap.parse_args()

    print("optimize_params starting...", flush=True)
    print(f"strategy={args.strategy} samples={args.samples} seed={args.seed} train_years={args.train_years} test_months={args.test_months} step_months={args.step_months} min_hold={args.min_hold} cost_bps={args.cost_bps}", flush=True)

    cfg = load_config()
    log(f"Config loaded. TREND_INPUT_PATH={cfg.TREND_INPUT_PATH}", verbose=True)
    if cfg.VALUATION_WORKBOOK is not None:
        log(f"Valuation workbook={cfg.VALUATION_WORKBOOK}, sheet={cfg.VALUATION_SHEET}", verbose=True)

    prices, betas = load_trend_input(cfg.TREND_INPUT_PATH)
    log(f"Prices shape={prices.shape}, dates {prices.index.min().date()}..{prices.index.max().date()}", verbose=True)

    valuation_raw = None
    if cfg.VALUATION_WORKBOOK is not None and cfg.VALUATION_SHEET is not None:
        valuation_raw = load_valuation(cfg.VALUATION_WORKBOOK, cfg.VALUATION_SHEET, prices.index, list(prices.columns))
        log("Valuation frame loaded", verbose=True)

    industry_map = load_industry_map(cfg.TREND_INPUT_PATH)

    warmup_days = args.warmup_days if args.warmup_days is not None else _compute_dynamic_warmup(cfg)
    folds = _date_folds(prices.index, args.train_years, args.test_months, args.step_months, warmup_days=warmup_days)
    if not folds:
        print("No folds available. Consider smaller --train_years/--test_months, or set --warmup_days to a smaller value.", flush=True)
        print(f"Data points: {len(prices.index)}, warmup_days used: {warmup_days}", flush=True)
        return
    log(f"Warmup_days={warmup_days}, num_folds={len(folds)}", verbose=True)

    t_start = time.perf_counter()
    if args.strategy == "trend":
        log("Begin evaluating Trend params", verbose=True)
        df = _evaluate_trend(cfg, prices, betas, valuation_raw, folds, args.min_hold, args.cost_bps, args.seed, args.samples, verbose=args.verbose)
        sheet = "Trend"
    elif args.strategy == "emergent":
        log("Begin evaluating Emergent params", verbose=True)
        df = _evaluate_emergent(cfg, prices, betas, valuation_raw, folds, args.min_hold, args.cost_bps, args.seed, args.samples, verbose=args.verbose, industry_map=industry_map)
        sheet = "Emergent"
    else:
        log("Begin evaluating Stars params", verbose=True)
        df = _evaluate_stars(cfg, prices, betas, valuation_raw, folds, args.min_hold, args.cost_bps, args.seed, args.samples, verbose=args.verbose)
        sheet = "Stars"

    from pathlib import Path
    from datetime import datetime  # only needed for the fallback below

    out_xlsx = Path("Tuning_Results.xlsx")
    exists = out_xlsx.exists()

    try:
        if exists:
            # Replace the sheet when the file already exists
            with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
                df.to_excel(xw, sheet_name=sheet, index=False)
        else:
            # Create a brand-new file — don't pass if_sheet_exists here
            with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as xw:
                df.to_excel(xw, sheet_name=sheet, index=False)

        print(f"Wrote {out_xlsx} (sheet={sheet})", flush=True)

    except PermissionError:
        # Excel might be holding the file open — fall back to a timestamped copy
        alt = out_xlsx.with_name(f"Tuning_Results_{datetime.now():%Y%m%d_%H%M%S}.xlsx")
        with pd.ExcelWriter(alt, engine="openpyxl", mode="w") as xw:
            df.to_excel(xw, sheet_name=sheet, index=False)
        print(f"{out_xlsx} locked. Wrote fallback {alt}", flush=True)

    best = df.iloc[0].to_dict()
    summary = {sheet: best}
    out_json = _Path("Tuning_Summary.json")
    if out_json.exists():
        try:
            prev = json.loads(out_json.read_text())
        except Exception:
            prev = {}
        prev.update(summary)
        out_json.write_text(json.dumps(prev, indent=2))
    else:
        out_json.write_text(json.dumps(summary, indent=2))
    log(f"Wrote {out_json}", verbose=True)

    elapsed = time.perf_counter() - t_start
    print(f"Top {sheet} params: {best}", flush=True)
    print(f"Done in {elapsed:.1f}s", flush=True)

if __name__ == "__main__":
    main()
