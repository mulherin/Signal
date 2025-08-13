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

    # Feature lookbacks
    RS_LOOKBACK_D: int
    RS_SHORT_D: int
    RS_LONG_D: int
    RS_MED_LOOKBACK_D: int
    ACCEL_LOOKBACK_D: int
    ADX_LEN_D: int
    ADX_ROC_LEN_D: int
    VAL_WINDOW_Y_MAX: float
    VAL_WINDOW_Y_MIN: float

    # Emergent thresholds (directional anchors, accel size, cross margins)
    DIR_ANCHOR_LONG_PCT: float
    DIR_ANCHOR_SHORT_PCT: float
    ACCEL_TOP_PCT: float
    ACCEL_BOT_PCT: float
    ACCEL_DELTA_MIN: float
    EMERGENT_LONG_CROSS_MARGIN: float
    EMERGENT_SHORT_CROSS_MARGIN: float

    # ADX (kept for compatibility; not used in emergent v2)
    EMERGENT_ADX_ROC_PCTL_MIN: float
    ADX_MIN_ROC: float

    # Emergent trend-shape parameters (NEW)
    EMERGENT_TSTAT_LEN_D: int
    EMERGENT_TSTAT_MIN_UP: float
    EMERGENT_TSTAT_MIN_DN: float
    EMERGENT_R2_MIN: float

    # Industry confirmation toggle (NEW)
    EMERGENT_USE_INDUSTRY_CONFIRM: bool

    # TTL and cooldown
    EMERGENT_TTL_D: int
    EMERGENT_TTL_LONG_D: int   # NEW (optional, falls back to EMERGENT_TTL_D)
    EMERGENT_TTL_SHORT_D: int  # NEW (optional, falls back to EMERGENT_TTL_D)
    EMERGENT_COOLDOWN_D: int

    # RS floors (kept in config for compatibility; not used in emergent v2)
    EMERGENT_LONG_RS_SHORT_MIN_PCT: float
    EMERGENT_SHORT_RS_SHORT_FLOOR_PCT: float

    # Emergent sleeve budgets
    EMERGENT_LONG_BUDGET: float
    EMERGENT_SHORT_BUDGET: float

    # Emergent logging
    EMERGENT_LOG: bool
    EMERGENT_LOG_EVERY_D: int

    # ---------------- Stars (unchanged) ----------------
    STAR_LOOKBACK_D: int
    STAR_RS_THRESH_PCT: float
    STAR_SUSTAIN_FRAC: float

    # ---------------- Unpredictable (unchanged) ----------------
    UNPRED_CORR_MAX: float
    UNPRED_ADX_MAX_PCT: float

    # Warnings / reporting
    RS_LONG_EXTREME_PCT: float
    RS_SHORT_EXTREME_PCT: float
    VAL_RICH_WARN_PCT: float
    VAL_CHEAP_WARN_PCT: float

    # Outputs
    OUTPUT_SIGNALS_PATH: Path
    OUTPUT_BACKTESTS_PATH: Path

    # ---------------- Trend (minimal) ----------------
    TREND_TSTAT_UP: float
    TREND_TSTAT_DOWN: float
    TREND_MIN_RSPCT: float
    TREND_USE_HYSTERESIS: bool
    TREND_REBALANCE_FREQ_D: int
    TREND_LONG_BUDGET: float
    TREND_SHORT_BUDGET: float


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

    RS_LOOKBACK_D = _coerce_int(raw.get("RS_LOOKBACK_D", 126), 126)
    RS_SHORT_D    = _coerce_int(raw.get("RS_SHORT_D", 21), 21)
    RS_LONG_D     = _coerce_int(raw.get("RS_LONG_D", 126), 126)
    RS_MED_LOOKBACK_D = _coerce_int(raw.get("RS_MED_LOOKBACK_D", RS_LOOKBACK_D), RS_LOOKBACK_D)
    ACCEL_LOOKBACK_D = _coerce_int(raw.get("ACCEL_LOOKBACK_D", 63), 63)
    ADX_LEN_D     = _coerce_int(raw.get("ADX_LEN_D", 20), 20)
    ADX_ROC_LEN_D = _coerce_int(raw.get("ADX_ROC_LEN_D", 20), 20)
    VAL_WINDOW_Y_MAX = _coerce_float(raw.get("VAL_WINDOW_Y_MAX", 3.0), 3.0)
    VAL_WINDOW_Y_MIN = _coerce_float(raw.get("VAL_WINDOW_Y_MIN", 1.0), 1.0)

    # Directional anchors
    DIR_ANCHOR_LONG_PCT  = _coerce_float(raw.get("DIR_ANCHOR_LONG_PCT", 0.65), 0.65)
    DIR_ANCHOR_SHORT_PCT = _coerce_float(raw.get("DIR_ANCHOR_SHORT_PCT", 0.30), 0.30)

    # Acceleration and cross margins
    ACCEL_TOP_PCT = _coerce_float(raw.get("ACCEL_TOP_PCT", 0.25), 0.25)
    ACCEL_BOT_PCT = _coerce_float(raw.get("ACCEL_BOT_PCT", 0.25), 0.25)
    ACCEL_DELTA_MIN = _coerce_float(raw.get("ACCEL_DELTA_MIN", 0.15), 0.15)
    EMERGENT_LONG_CROSS_MARGIN  = _coerce_float(raw.get("EMERGENT_LONG_CROSS_MARGIN", 0.06), 0.06)
    EMERGENT_SHORT_CROSS_MARGIN = _coerce_float(
        raw.get("EMERGENT_SHORT_CROSS_MARGIN", raw.get("EMERGENT_LONG_CROSS_MARGIN", 0.06)), 0.06
    )

    # ADX ROC tail-based controls (kept for compatibility)
    EMERGENT_ADX_ROC_PCTL_MIN = _coerce_float(raw.get("EMERGENT_ADX_ROC_PCTL_MIN", 0.65), 0.65)
    ADX_MIN_ROC = _coerce_float(raw.get("ADX_MIN_ROC", 1.0), 1.0)

    # Emergent trend-shape (NEW)
    EMERGENT_TSTAT_LEN_D   = _coerce_int(raw.get("EMERGENT_TSTAT_LEN_D", 63), 63)
    EMERGENT_TSTAT_MIN_UP  = _coerce_float(raw.get("EMERGENT_TSTAT_MIN_UP", 0.5), 0.5)
    EMERGENT_TSTAT_MIN_DN  = _coerce_float(raw.get("EMERGENT_TSTAT_MIN_DN", 0.7), 0.7)
    EMERGENT_R2_MIN        = _coerce_float(raw.get("EMERGENT_R2_MIN", 0.15), 0.15)

    # Industry confirm toggle (NEW)
    EMERGENT_USE_INDUSTRY_CONFIRM = _coerce_bool(raw.get("EMERGENT_USE_INDUSTRY_CONFIRM", True))

    # TTL and cooldown (with optional asym)
    EMERGENT_TTL_D = _coerce_int(raw.get("EMERGENT_TTL_D", 28), 28)
    EMERGENT_TTL_LONG_D = _coerce_int(raw.get("EMERGENT_TTL_LONG_D", raw.get("EMERGENT_TTL_D", 28)), 28)
    EMERGENT_TTL_SHORT_D = _coerce_int(raw.get("EMERGENT_TTL_SHORT_D", raw.get("EMERGENT_TTL_D", 28)), 28)
    EMERGENT_COOLDOWN_D = _coerce_int(raw.get("EMERGENT_COOLDOWN_D", 10), 10)

    # RS floors (kept for compatibility; not used in emergent v2)
    EMERGENT_LONG_RS_SHORT_MIN_PCT   = _coerce_float(raw.get("EMERGENT_LONG_RS_SHORT_MIN_PCT", 0.50), 0.50)
    EMERGENT_SHORT_RS_SHORT_FLOOR_PCT = _coerce_float(raw.get("EMERGENT_SHORT_RS_SHORT_FLOOR_PCT", 0.25), 0.25)

    # Emergent budgets
    EMERGENT_LONG_BUDGET  = _coerce_float(raw.get("EMERGENT_LONG_BUDGET", 1.0), 1.0)
    EMERGENT_SHORT_BUDGET = _coerce_float(raw.get("EMERGENT_SHORT_BUDGET", 0.0), 0.0)

    # Emergent logging
    EMERGENT_LOG = _coerce_bool(raw.get("EMERGENT_LOG", False))
    EMERGENT_LOG_EVERY_D = _coerce_int(raw.get("EMERGENT_LOG_EVERY_D", 21), 21)

    # Stars (unchanged)
    STAR_LOOKBACK_D = _coerce_int(raw.get("STAR_LOOKBACK_D", 252), 252)
    STAR_RS_THRESH_PCT = _coerce_float(raw.get("STAR_RS_THRESH_PCT", 0.70), 0.70)
    STAR_SUSTAIN_FRAC = _coerce_float(raw.get("STAR_SUSTAIN_FRAC", 0.70), 0.70)

    # Unpredictable (unchanged)
    UNPRED_CORR_MAX = _coerce_float(raw.get("UNPRED_CORR_MAX", 0.05), 0.05)
    UNPRED_ADX_MAX_PCT = _coerce_float(raw.get("UNPRED_ADX_MAX_PCT", 0.40), 0.40)

    # Warnings / reporting
    RS_LONG_EXTREME_PCT = _coerce_float(raw.get("RS_LONG_EXTREME_PCT", 0.85), 0.85)
    RS_SHORT_EXTREME_PCT = _coerce_float(raw.get("RS_SHORT_EXTREME_PCT", 0.15), 0.15)
    VAL_RICH_WARN_PCT = _coerce_float(raw.get("VAL_RICH_WARN_PCT", 0.75), 0.75)
    VAL_CHEAP_WARN_PCT = _coerce_float(raw.get("VAL_CHEAP_WARN_PCT", 0.25), 0.25)

    # Outputs
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
        RS_MED_LOOKBACK_D=RS_MED_LOOKBACK_D,
        ACCEL_LOOKBACK_D=ACCEL_LOOKBACK_D,
        ADX_LEN_D=ADX_LEN_D,
        ADX_ROC_LEN_D=ADX_ROC_LEN_D,
        VAL_WINDOW_Y_MAX=VAL_WINDOW_Y_MAX,
        VAL_WINDOW_Y_MIN=VAL_WINDOW_Y_MIN,
        DIR_ANCHOR_LONG_PCT=DIR_ANCHOR_LONG_PCT,
        DIR_ANCHOR_SHORT_PCT=DIR_ANCHOR_SHORT_PCT,
        ACCEL_TOP_PCT=ACCEL_TOP_PCT,
        ACCEL_BOT_PCT=ACCEL_BOT_PCT,
        ACCEL_DELTA_MIN=ACCEL_DELTA_MIN,
        EMERGENT_LONG_CROSS_MARGIN=EMERGENT_LONG_CROSS_MARGIN,
        EMERGENT_SHORT_CROSS_MARGIN=EMERGENT_SHORT_CROSS_MARGIN,
        EMERGENT_ADX_ROC_PCTL_MIN=EMERGENT_ADX_ROC_PCTL_MIN,
        ADX_MIN_ROC=ADX_MIN_ROC,
        EMERGENT_TSTAT_LEN_D=EMERGENT_TSTAT_LEN_D,
        EMERGENT_TSTAT_MIN_UP=EMERGENT_TSTAT_MIN_UP,
        EMERGENT_TSTAT_MIN_DN=EMERGENT_TSTAT_MIN_DN,
        EMERGENT_R2_MIN=EMERGENT_R2_MIN,
        EMERGENT_USE_INDUSTRY_CONFIRM=EMERGENT_USE_INDUSTRY_CONFIRM,
        EMERGENT_TTL_D=EMERGENT_TTL_D,
        EMERGENT_TTL_LONG_D=EMERGENT_TTL_LONG_D,
        EMERGENT_TTL_SHORT_D=EMERGENT_TTL_SHORT_D,
        EMERGENT_COOLDOWN_D=EMERGENT_COOLDOWN_D,
        EMERGENT_LONG_RS_SHORT_MIN_PCT=EMERGENT_LONG_RS_SHORT_MIN_PCT,
        EMERGENT_SHORT_RS_SHORT_FLOOR_PCT=EMERGENT_SHORT_RS_SHORT_FLOOR_PCT,
        EMERGENT_LONG_BUDGET=EMERGENT_LONG_BUDGET,
        EMERGENT_SHORT_BUDGET=EMERGENT_SHORT_BUDGET,
        EMERGENT_LOG=EMERGENT_LOG,
        EMERGENT_LOG_EVERY_D=EMERGENT_LOG_EVERY_D,
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
        TREND_TSTAT_UP=_coerce_float(raw.get("TREND_TSTAT_UP", 1.5), 1.5),
        TREND_TSTAT_DOWN=_coerce_float(raw.get("TREND_TSTAT_DOWN", 0.8), 0.8),
        TREND_MIN_RSPCT=_coerce_float(raw.get("TREND_MIN_RSPCT", 0.50), 0.50),
        TREND_USE_HYSTERESIS=_coerce_bool(raw.get("TREND_USE_HYSTERESIS", False)),
        TREND_REBALANCE_FREQ_D=_coerce_int(raw.get("TREND_REBALANCE_FREQ_D", 5), 5),
        TREND_LONG_BUDGET=_coerce_float(raw.get("TREND_LONG_BUDGET", 0.5), 0.5),
        TREND_SHORT_BUDGET=_coerce_float(raw.get("TREND_SHORT_BUDGET", 0.0), 0.0),
    )
