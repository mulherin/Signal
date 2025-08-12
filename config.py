# config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import pandas as pd

@dataclass
class Config:
    # Paths
    TREND_INPUT_PATH: Path
    VALUATION_WORKBOOK: Optional[Path]
    VALUATION_SHEET: Optional[str]

    # Residual model
    BETA_FLOOR: float

    # Lookbacks (trend + emergent)
    RS_LOOKBACK_D: int          # "medium" horizon for RS (existing)
    RS_SHORT_D: int             # NEW: short RS horizon (e.g., 21)
    RS_LONG_D: int              # NEW: long RS horizon (e.g., 126)
    ACCEL_LOOKBACK_D: int
    ADX_LEN_D: int
    ADX_ROC_LEN_D: int
    VAL_WINDOW_Y_MAX: float
    VAL_WINDOW_Y_MIN: float

    # Trend tilt (daily output)
    TILT_TOP_PCT: float
    TILT_BOT_PCT: float

    # Trend tilt (backtest only)
    BACKTEST_TILT_TOP_PCT: float
    BACKTEST_TILT_BOT_PCT: float
    BACKTEST_SKIP_D: int
    BACKTEST_HOLD_D: int
    BACKTEST_REBALANCE_FREQ_D: int

    # Emergent base thresholds
    DIR_ANCHOR_LONG_PCT: float
    DIR_ANCHOR_SHORT_PCT: float
    ACCEL_TOP_PCT: float
    ACCEL_BOT_PCT: float
    EMERGENT_TTL_D: int
    EMERGENT_COOLDOWN_D: int

    # Emergent LONG crossover params (all NEW)
    EMERGENT_LONG_CROSS_MARGIN: float          # e.g., 0.02 (2 pct-pts)
    EMERGENT_LONG_RS_SHORT_MIN_PCT: float      # e.g., 0.50 (short RS must be at least this)

    # Emergent SHORT anti-bottom-tick guard (NEW)
    EMERGENT_SHORT_RS_SHORT_FLOOR_PCT: float   # e.g., 0.15 (don’t short extreme oversold)

    # Stars
    STAR_LOOKBACK_D: int
    STAR_RS_THRESH_PCT: float
    STAR_SUSTAIN_FRAC: float

    # Unpredictable
    UNPRED_CORR_MAX: float
    UNPRED_ADX_MAX_PCT: float

    # Overstretched + Valuation warn thresholds (warnings only)
    RS_LONG_EXTREME_PCT: float
    RS_SHORT_EXTREME_PCT: float
    VAL_RICH_WARN_PCT: float
    VAL_CHEAP_WARN_PCT: float

    # Outputs
    OUTPUT_SIGNALS_PATH: Path
    OUTPUT_BACKTESTS_PATH: Path

def _coerce_bool(x):
    s = str(x).strip().lower()
    return s in ("1", "true", "yes", "y")

def _coerce_float(x, d):
    try:
        xs = str(x).strip()
        return float(xs[:-1]) / 100.0 if xs.endswith("%") else float(xs)
    except:
        return d

def _coerce_int(x, d):
    try:
        return int(float(x))
    except:
        return d

def _coerce_str(x, d):
    try:
        s = str(x)
        return d if s.strip() == "" else s
    except:
        return d

def _read_config_sheet(path: Path) -> Dict[str, object]:
    try:
        df = pd.read_excel(path, sheet_name="Config")
    except:
        return {}
    if df.shape[1] < 2:
        return {}
    df.columns = [str(c).strip() for c in df.columns]
    kc, vc = df.columns[0], df.columns[1]
    out = {}
    for _, r in df.iterrows():
        k = str(r.get(kc, "")).strip()
        if k and k.lower() != "nan":
            out[k] = r.get(vc, "")
    return out

def load_config(trend_input_path: Optional[Path] = None) -> Config:
    default_trend = Path("./Trend_Input.xlsm") if trend_input_path is None else Path(trend_input_path)
    raw = _read_config_sheet(default_trend)

    TREND_INPUT_PATH = Path(_coerce_str(raw.get("TREND_INPUT_PATH", str(default_trend)), str(default_trend)))
    vw = _coerce_str(raw.get("VALUATION_WORKBOOK", ""), "")
    VALUATION_WORKBOOK = Path(vw) if vw else None
    VALUATION_SHEET = _coerce_str(raw.get("VALUATION_SHEET", "Raw_EV"), "Raw_EV")

    BETA_FLOOR = _coerce_float(raw.get("BETA_FLOOR", 0.05), 0.05)

    # Lookbacks
    RS_LOOKBACK_D = _coerce_int(raw.get("RS_LOOKBACK_D", 126), 126)
    RS_SHORT_D = _coerce_int(raw.get("RS_SHORT_D", 21), 21)
    RS_LONG_D  = _coerce_int(raw.get("RS_LONG_D", 126), 126)
    ACCEL_LOOKBACK_D = _coerce_int(raw.get("ACCEL_LOOKBACK_D", 63), 63)
    ADX_LEN_D = _coerce_int(raw.get("ADX_LEN_D", 20), 20)
    ADX_ROC_LEN_D = _coerce_int(raw.get("ADX_ROC_LEN_D", 20), 20)
    VAL_WINDOW_Y_MAX = _coerce_float(raw.get("VAL_WINDOW_Y_MAX", 3.0), 3.0)
    VAL_WINDOW_Y_MIN = _coerce_float(raw.get("VAL_WINDOW_Y_MIN", 1.0), 1.0)

    # Tilt
    TILT_TOP_PCT = _coerce_float(raw.get("TILT_TOP_PCT", 0.20), 0.20)
    TILT_BOT_PCT = _coerce_float(raw.get("TILT_BOT_PCT", 0.20), 0.20)

    BACKTEST_TILT_TOP_PCT = _coerce_float(raw.get("BACKTEST_TILT_TOP_PCT", TILT_TOP_PCT), TILT_TOP_PCT)
    BACKTEST_TILT_BOT_PCT = _coerce_float(raw.get("BACKTEST_TILT_BOT_PCT", TILT_BOT_PCT), TILT_BOT_PCT)
    BACKTEST_SKIP_D = _coerce_int(raw.get("BACKTEST_SKIP_D", 20), 20)
    BACKTEST_HOLD_D = _coerce_int(raw.get("BACKTEST_HOLD_D", 126), 126)
    BACKTEST_REBALANCE_FREQ_D = _coerce_int(raw.get("BACKTEST_REBALANCE_FREQ_D", 20), 20)

    # Emergent base
    DIR_ANCHOR_LONG_PCT = _coerce_float(raw.get("DIR_ANCHOR_LONG_PCT", 0.55), 0.55)
    DIR_ANCHOR_SHORT_PCT = _coerce_float(raw.get("DIR_ANCHOR_SHORT_PCT", 0.45), 0.45)
    ACCEL_TOP_PCT = _coerce_float(raw.get("ACCEL_TOP_PCT", 0.30), 0.30)
    ACCEL_BOT_PCT = _coerce_float(raw.get("ACCEL_BOT_PCT", 0.30), 0.30)
    EMERGENT_TTL_D = _coerce_int(raw.get("EMERGENT_TTL_D", 20), 20)
    EMERGENT_COOLDOWN_D = _coerce_int(raw.get("EMERGENT_COOLDOWN_D", 10), 10)

    # Emergent LONG crossover params
    EMERGENT_LONG_CROSS_MARGIN = _coerce_float(raw.get("EMERGENT_LONG_CROSS_MARGIN", 0.02), 0.02)
    EMERGENT_LONG_RS_SHORT_MIN_PCT = _coerce_float(raw.get("EMERGENT_LONG_RS_SHORT_MIN_PCT", 0.50), 0.50)

    # Emergent SHORT anti-bottom-tick guard
    EMERGENT_SHORT_RS_SHORT_FLOOR_PCT = _coerce_float(raw.get("EMERGENT_SHORT_RS_SHORT_FLOOR_PCT", 0.15), 0.15)

    # Stars
    STAR_LOOKBACK_D = _coerce_int(raw.get("STAR_LOOKBACK_D", 252), 252)
    STAR_RS_THRESH_PCT = _coerce_float(raw.get("STAR_RS_THRESH_PCT", 0.70), 0.70)
    STAR_SUSTAIN_FRAC = _coerce_float(raw.get("STAR_SUSTAIN_FRAC", 0.70), 0.70)

    # Unpredictable
    UNPRED_CORR_MAX = _coerce_float(raw.get("UNPRED_CORR_MAX", 0.05), 0.05)
    UNPRED_ADX_MAX_PCT = _coerce_float(raw.get("UNPRED_ADX_MAX_PCT", 0.40), 0.40)

    # Warnings
    RS_LONG_EXTREME_PCT = _coerce_float(raw.get("RS_LONG_EXTREME_PCT", 0.85), 0.85)
    RS_SHORT_EXTREME_PCT = _coerce_float(raw.get("RS_SHORT_EXTREME_PCT", 0.15), 0.15)
    VAL_RICH_WARN_PCT = _coerce_float(raw.get("VAL_RICH_WARN_PCT", 0.75), 0.75)
    VAL_CHEAP_WARN_PCT = _coerce_float(raw.get("VAL_CHEAP_WARN_PCT", 0.25), 0.25)

    OUTPUT_SIGNALS_PATH = Path(_coerce_str(raw.get("OUTPUT_SIGNALS_PATH", "Signals_Daily.xlsx"), "Signals_Daily.xlsx"))
    OUTPUT_BACKTESTS_PATH = Path(_coerce_str(raw.get("OUTPUT_BACKTESTS_PATH", "Backtests_Summary.xlsx"), "Backtests_Summary.xlsx"))

    return Config(
        TREND_INPUT_PATH=TREND_INPUT_PATH,
        VALUATION_WORKBOOK=VALUATION_WORKBOOK,
        VALUATION_SHEET=VALUATION_SHEET,
        BETA_FLOOR=BETA_FLOOR,
        RS_LOOKBACK_D=RS_LOOKBACK_D,
        RS_SHORT_D=RS_SHORT_D,
        RS_LONG_D=RS_LONG_D,
        ACCEL_LOOKBACK_D=ACCEL_LOOKBACK_D,
        ADX_LEN_D=ADX_LEN_D,
        ADX_ROC_LEN_D=ADX_ROC_LEN_D,
        VAL_WINDOW_Y_MAX=VAL_WINDOW_Y_MAX,
        VAL_WINDOW_Y_MIN=VAL_WINDOW_Y_MIN,
        TILT_TOP_PCT=TILT_TOP_PCT,
        TILT_BOT_PCT=TILT_BOT_PCT,
        BACKTEST_TILT_TOP_PCT=BACKTEST_TILT_TOP_PCT,
        BACKTEST_TILT_BOT_PCT=BACKTEST_TILT_BOT_PCT,
        BACKTEST_SKIP_D=BACKTEST_SKIP_D,
        BACKTEST_HOLD_D=BACKTEST_HOLD_D,
        BACKTEST_REBALANCE_FREQ_D=BACKTEST_REBALANCE_FREQ_D,
        DIR_ANCHOR_LONG_PCT=DIR_ANCHOR_LONG_PCT,
        DIR_ANCHOR_SHORT_PCT=DIR_ANCHOR_SHORT_PCT,
        ACCEL_TOP_PCT=ACCEL_TOP_PCT,
        ACCEL_BOT_PCT=ACCEL_BOT_PCT,
        EMERGENT_TTL_D=EMERGENT_TTL_D,
        EMERGENT_COOLDOWN_D=EMERGENT_COOLDOWN_D,
        EMERGENT_LONG_CROSS_MARGIN=EMERGENT_LONG_CROSS_MARGIN,
        EMERGENT_LONG_RS_SHORT_MIN_PCT=EMERGENT_LONG_RS_SHORT_MIN_PCT,
        EMERGENT_SHORT_RS_SHORT_FLOOR_PCT=EMERGENT_SHORT_RS_SHORT_FLOOR_PCT,
        STAR_LOOKBACK_D=STAR_LOOKBACK_D,
        STAR_RS_THRESH_PCT=STAR_RS_THRESH_PCT,
        STAR_SUSTAIN_FRAC=STAR_SUSTAIN_FRAC,
        UNPRED_CORR_MAX=UNPRED_CORR_MAX,
        UNPRED_ADX_MAX_PCT=UNPRED_ADX_MAX_PCT,
        RS_LONG_EXTREME_PCT=RS_LONG_EXTREME_PCT,
        RS_SHORT_EXTREME_PCT=RS_SHORT_EXTREME_PCT,
        VAL_RICH_WARN_PCT=VAL_RICH_WARN_PCT,
        VAL_CHEAP_WARN_PCT=VAL_CHEAP_WARN_PCT,
        OUTPUT_SIGNALS_PATH=OUTPUT_SIGNALS_PATH,
        OUTPUT_BACKTESTS_PATH=OUTPUT_BACKTESTS_PATH,
    )
