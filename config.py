# Streamlined, robust config loader for the Signals stack.
# - Defaults input to: C:\Users\TaylorMulherin\Documents\Signals\Signals_Script\signals_input.xlsm
# - Reads an optional "Config" sheet (two columns: Parameter | Value).
# - Ignores TREND_INPUT_PATH overrides inside the sheet so the path cannot go stale.
# - Robust coercers: tolerate TRUE/FALSE in numeric cells (maps to 1/0), skip blanks, ignore commented rows.

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


# ---------- Defaults ----------

DEFAULT_INPUT_PATH = Path(
    r"C:\Users\TaylorMulherin\Documents\Signals\Signals_Script\signals_input.xlsm"
)


@dataclass(frozen=True)
class Config:
    # Paths
    TREND_INPUT_PATH: Path
    OUTPUT_SIGNALS_PATH: Path
    OUTPUT_BACKTESTS_PATH: Path
    VALUATION_WORKBOOK: Optional[Path] = None
    VALUATION_SHEET: Optional[str] = None  # e.g., "Raw_EV"

    # Feature params
    RS_LOOKBACK_D: int = 126
    RS_MED_LOOKBACK_D: int = 126

    # Unpredictable badge
    UNPRED_CORR_MAX: float = 0.05

    # Trend (global)
    TREND_TSTAT_LEN_D: int = 63
    TREND_TSTAT_UP: float = 1.50
    TREND_TSTAT_DOWN: float = 0.80
    TREND_MIN_RSPCT: float = 0.50
    TREND_R2_MIN: float = 0.15

    TREND_USE_INDUSTRY_CONFIRM: bool = False
    TREND_INDUSTRY_MARGIN: float = 0.00

    TREND_USE_HYSTERESIS: bool = True
    TREND_REBALANCE_FREQ_D: int = 5
    TREND_LONG_BUDGET: float = 1.00
    TREND_SHORT_BUDGET: float = 0.00

    TREND_LOG: bool = False
    TREND_LOG_EVERY_D: int = 20

    # Trend (industry-relative twin)
    TREND_IND_ENABLED: bool = True
    TREND_GROUP_MIN_SIZE: int = 3

    # Emergent (k-day shock)
    ADD_SHOCK_LEN_D: int = 5
    ADD_SHOCK_Z_THR: float = 1.25
    ADD_VOL_WIN_D: int = 60
    ADD_TREND_GATE_LEN_D: int = 63
    ADD_RS_FLOOR_LONG: float = 0.50
    ADD_RS_CEIL_SHORT: float = 0.50
    ADD_NEWS_SIGMA_MAX: float = 3.0

    EMERGENT_REBALANCE_FREQ_D: int = 1
    EMERGENT_LONG_BUDGET: float = 0.50
    EMERGENT_SHORT_BUDGET: float = 0.50
    EMERGENT_MIN_HOLD_D: int = 3  # carry an Emergent tag for at least N days
    EMERGENT_MAX_HOLD_D: int = 5          # new: cap carry to about a week
    ADD_Z_RESET: float = 0.25             # new: re-arm threshold for edge triggers
    ADD_RS_LOOKBACK_RECENT: int = 20      # new: lookback window for recent RS pulse
    ADD_FADE_RS_MIN: float = 0.55         # new: require a recent RS high to allow FadeRally
    # Emergent pulse guards
    ADD_PULSE_FRESH_MAX_D: int = 3    # lookback for "fresh" daily pulse
    ADD_FADE_1D_MIN: float = 0.60     # FadeRally requires a recent day with 1d rank >= this
    ADD_BUY_1D_MAX: float = 0.40      # BuyDip requires a recent day with 1d rank <= this
    EMERGENT_PULSE_LOOKBACK_D: int = 5
    EMERGENT_FADE_TOP_PCT: float = 0.85
    EMERGENT_BUY_BOT_PCT: float = 0.15
    EMERGENT_KILL_LOOKBACK_D: int = 5
    EMERGENT_MAX_HOLD_D: int = 5
    EMERGENT_KILL_LOOKBACK_D: int = 5 


    # Stars
    STAR_LOOKBACK_D: int = 252
    STAR_RS_THRESH_PCT: float = 0.65
    STAR_SUSTAIN_FRAC: float = 0.65
    STAR_USE_INDUSTRY_CONFIRM: bool = True
    STAR_INDUSTRY_CONFIRM_FRAC: float = 0.55
    STAR_INDUSTRY_MARGIN: float = 0.00

    STARS_LOG: bool = False
    STARS_LOG_EVERY_D: int = 20

    # Reporting
    VAL_RICH_WARN_PCT: float = 0.80
    VAL_CHEAP_WARN_PCT: float = 0.20


# ---------- Coercers ----------

def _is_blank(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"


def _coerce_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "y", "yes", "on"):
        return True
    if s in ("0", "false", "f", "n", "no", "off"):
        return False
    raise ValueError(f"Expected bool-like, got: {x!r}")


def _coerce_int(x: Any) -> int:
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    s = str(x).strip()
    if s.lower() in ("true", "t", "yes", "on"):
        return 1
    if s.lower() in ("false", "f", "no", "off"):
        return 0
    return int(float(s))


def _coerce_float(x: Any) -> float:
    if isinstance(x, (bool, np.bool_)):
        return 1.0 if bool(x) else 0.0
    s = str(x).strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    if s.lower() in ("true", "t", "yes", "on"):
        return 1.0
    if s.lower() in ("false", "f", "no", "off"):
        return 0.0
    return float(s)


def _coerce_path(x: Any, base: Path) -> Path:
    if _is_blank(x):
        return base
    p = Path(str(x).strip())
    return p if p.is_absolute() else (base / p)


# Map of key → coercer (do not include TREND_INPUT_PATH here)
_COERCERS: Dict[str, Any] = {
    # Paths
    "OUTPUT_SIGNALS_PATH": _coerce_path,
    "OUTPUT_BACKTESTS_PATH": _coerce_path,
    "VALUATION_WORKBOOK": _coerce_path,
    "VALUATION_SHEET": str,

    # Ints
    "RS_LOOKBACK_D": _coerce_int,
    "RS_MED_LOOKBACK_D": _coerce_int,
    "TREND_TSTAT_LEN_D": _coerce_int,
    "TREND_REBALANCE_FREQ_D": _coerce_int,
    "TREND_LOG_EVERY_D": _coerce_int,
    "EMERGENT_REBALANCE_FREQ_D": _coerce_int,
    "STAR_LOOKBACK_D": _coerce_int,
    "STARS_LOG_EVERY_D": _coerce_int,
    "EMERGENT_MIN_HOLD_D": _coerce_int,
    "TREND_GROUP_MIN_SIZE": _coerce_int,
    "EMERGENT_MAX_HOLD_D": _coerce_int,
    "ADD_RS_LOOKBACK_RECENT": _coerce_int,
    "EMERGENT_KILL_LOOKBACK_D": _coerce_int,


    # Floats
    "ADD_FADE_RS_MIN": _coerce_float,
    "ADD_Z_RESET": _coerce_float,
    "UNPRED_CORR_MAX": _coerce_float,
    "TREND_TSTAT_UP": _coerce_float,
    "TREND_TSTAT_DOWN": _coerce_float,
    "TREND_MIN_RSPCT": _coerce_float,
    "TREND_R2_MIN": _coerce_float,
    "TREND_INDUSTRY_MARGIN": _coerce_float,
    "TREND_LONG_BUDGET": _coerce_float,
    "TREND_SHORT_BUDGET": _coerce_float,
    "ADD_SHOCK_LEN_D": _coerce_int,
    "ADD_SHOCK_Z_THR": _coerce_float,
    "ADD_VOL_WIN_D": _coerce_int,
    "ADD_TREND_GATE_LEN_D": _coerce_int,
    "ADD_RS_FLOOR_LONG": _coerce_float,
    "ADD_RS_CEIL_SHORT": _coerce_float,
    "ADD_NEWS_SIGMA_MAX": _coerce_float,
    "EMERGENT_LONG_BUDGET": _coerce_float,
    "EMERGENT_SHORT_BUDGET": _coerce_float,
    "STAR_RS_THRESH_PCT": _coerce_float,
    "STAR_SUSTAIN_FRAC": _coerce_float,
    "STAR_INDUSTRY_CONFIRM_FRAC": _coerce_float,
    "STAR_INDUSTRY_MARGIN": _coerce_float,
    "VAL_RICH_WARN_PCT": _coerce_float,
    "VAL_CHEAP_WARN_PCT": _coerce_float,
    "ADD_PULSE_FRESH_MAX_D": _coerce_int,
    "ADD_FADE_1D_MIN": _coerce_float,
    "ADD_BUY_1D_MAX": _coerce_float,
    "EMERGENT_PULSE_LOOKBACK_D": _coerce_int,
    "EMERGENT_FADE_TOP_PCT": _coerce_float,
    "EMERGENT_BUY_BOT_PCT": _coerce_float,
    "EMERGENT_MAX_HOLD_D": _coerce_int,
    "EMERGENT_KILL_LOOKBACK_D": _coerce_int,
   

    # Bools
    "TREND_USE_INDUSTRY_CONFIRM": _coerce_bool,
    "TREND_USE_HYSTERESIS": _coerce_bool,
    "TREND_LOG": _coerce_bool,
    "STAR_USE_INDUSTRY_CONFIRM": _coerce_bool,
    "STARS_LOG": _coerce_bool,
    "TREND_IND_ENABLED": _coerce_bool,
}


# ---------- Read "Config" sheet ----------

def _read_config_sheet(xlsx_path: Path) -> Dict[str, Any]:
    """
    Read a 2-column 'Config' sheet (Parameter | Value).
    - Skips blank keys, keys starting with '#', and entirely blank rows.
    - Returns {KEY: value} with KEY uppercased and stripped.
    """
    try:
        xl = pd.ExcelFile(xlsx_path)
    except Exception:
        return {}

    sheet_name = None
    for cand in xl.sheet_names:
        if str(cand).strip().lower() == "config":
            sheet_name = cand
            break
    if sheet_name is None:
        return {}

    df = xl.parse(sheet_name)
    if df.shape[1] < 2:
        return {}

    kcol, vcol = df.columns[:2]
    out: Dict[str, Any] = {}
    for _, row in df.iterrows():
        key_raw = row[kcol]
        val = row[vcol]
        if _is_blank(key_raw):
            continue
        key = str(key_raw).strip()
        if key.startswith("#"):
            continue
        out[key.upper()] = val
    return out


# ---------- Load + apply overrides ----------

def load_config(trend_input_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Build Config:
      1) Start from safe defaults.
      2) Read 'Config' sheet and overlay overrides (except TREND_INPUT_PATH).
      3) Ensure outputs live next to the input if not overridden.
      4) Fallback to common filenames if the input path does not exist.
    """
    in_path = Path(trend_input_path) if trend_input_path else DEFAULT_INPUT_PATH
    base_dir = in_path.parent

    cfg = Config(
        TREND_INPUT_PATH=in_path,
        OUTPUT_SIGNALS_PATH=base_dir / "Signals_Daily.xlsx",
        OUTPUT_BACKTESTS_PATH=base_dir / "Backtests_Summary.xlsx",
        VALUATION_WORKBOOK=None,
        VALUATION_SHEET=None,
    )

    overrides_raw = _read_config_sheet(in_path)

    # Aliases accepted in the sheet
    alias: Dict[str, str] = {
        "STAR_LOG": "STARS_LOG",
        "STAR_LOG_EVERY_D": "STARS_LOG_EVERY_D",

        # Emergent compatibility aliases
        "DIR_ANCHOR_LONG_PCT": "ADD_RS_FLOOR_LONG",
        "DIR_ANCHOR_SHORT_PCT": "ADD_RS_CEIL_SHORT",
        "EMERGENT_TSTAT_LEN_D": "ADD_TREND_GATE_LEN_D",
        "ACCEL_DELTA_MIN": "ADD_SHOCK_Z_THR",
    }

    if overrides_raw:
        for k_raw, v in overrides_raw.items():
            key0 = str(k_raw).strip().upper()
            if key0 == "TREND_INPUT_PATH":
                continue
            key = alias.get(key0, key0)

            if _is_blank(v):
                continue
            if not hasattr(cfg, key):
                continue

            coercer = _COERCERS.get(key, None)
            try:
                if coercer is None:
                    val = str(v).strip()
                else:
                    if coercer is _coerce_path:
                        val = coercer(v, base=base_dir)
                        if key in ("VALUATION_WORKBOOK",) and (val == base_dir):
                            val = None
                    else:
                        val = coercer(v)
                cfg = replace(cfg, **{key: val})
            except Exception as e:
                print(f"[config] warning: could not coerce value for {key}={v!r} ({e}); keeping default.", flush=True)

    # Ensure TREND_INPUT_PATH exists, with fallbacks
    if not cfg.TREND_INPUT_PATH.exists():
        candidates = [
            base_dir / "signals_input.xlsm",
            base_dir / "signals_input.XLSM",
            base_dir / "Trend_Input.xlsm",
            base_dir / "Trend_Input.XLSM",
        ]
        for cand in candidates:
            if cand.exists():
                cfg = replace(cfg, TREND_INPUT_PATH=cand)
                break

    # Ensure outputs are set next to the input path
    base_dir = cfg.TREND_INPUT_PATH.parent
    if not cfg.OUTPUT_SIGNALS_PATH:
        cfg = replace(cfg, OUTPUT_SIGNALS_PATH=base_dir / "Signals_Daily.xlsx")
    if not cfg.OUTPUT_BACKTESTS_PATH:
        cfg = replace(cfg, OUTPUT_BACKTESTS_PATH=base_dir / "Backtests_Summary.xlsx")

    return cfg
