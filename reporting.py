# reporting.py
from typing import Dict
import pandas as pd
from config import Config

def write_signals_daily(tickers: list,
                        features: Dict[str, pd.DataFrame],
                        trend_daily: pd.Series,
                        emergent_daily: pd.Series,
                        emergent_ttl_daily: pd.Series,
                        stars_daily: pd.Series,
                        unpredictable: pd.Series,
                        cfg: Config) -> None:
    last = features["RS_pct"].index[-1]

    # valuation warning only (no gating)
    def _val_warn(row_val: pd.Series) -> pd.Series:
        out = pd.Series("None", index=row_val.index, dtype=object)
        out[row_val >= cfg.VAL_RICH_WARN_PCT] = "Rich"
        out[row_val <= cfg.VAL_CHEAP_WARN_PCT] = "Cheap"
        return out

    # emergent reason code (for audit/readability)
    def _emergent_reason(label: pd.Series) -> pd.Series:
        out = pd.Series("", index=label.index, dtype=object)
        out[label == "Inflection"] = "Crossover"
        out[label == "Breakdown"] = "Accel"
        return out

    row_RS = features["RS_pct"].loc[last, tickers]
    row_accel = features["Accel_pct"].loc[last, tickers]
    row_adxroc = features["ADX_ROC"].loc[last, tickers]
    row_val = features["Val_pct"].loc[last, tickers]
    row_val_warn = _val_warn(row_val)

    df = pd.DataFrame({
        "Ticker": tickers,
        "RS_pct": row_RS.values,
        "Onside_Tilt": trend_daily.reindex(tickers).fillna("None").values,
        "Accel_pct": row_accel.values,
        "ADX_ROC": row_adxroc.values,
        "Emergent": emergent_daily.reindex(tickers).fillna("").values,
        "Emergent_Reason": _emergent_reason(emergent_daily.reindex(tickers).fillna("")),
        "Emergent_TTL_Rem": emergent_ttl_daily.reindex(tickers).fillna(0).astype(int).values,
        "Star": stars_daily.reindex(tickers).fillna("").values,
        "Val_pct": row_val.values,                 # stays 0–1
        "Valuation_Warn": row_val_warn.values,     # "Rich" / "Cheap" / "None"
        "Overstretched_Warn": _overstretched_warn_row(features, cfg, last, tickers),
        "Unpredictable": unpredictable.reindex(tickers).fillna(False).astype(bool).values,
    })

    with pd.ExcelWriter(cfg.OUTPUT_SIGNALS_PATH, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Signals")

def _overstretched_warn_row(features: Dict[str, pd.DataFrame], cfg: Config, last, tickers) -> list:
    RS_pct = features["RS_pct"].loc[last, tickers]
    ADX_ROC = features["ADX_ROC"].loc[last, tickers]
    Val_pct = features["Val_pct"].loc[last, tickers]

    long_warn = (RS_pct >= cfg.RS_LONG_EXTREME_PCT) & (ADX_ROC <= 0) & (Val_pct >= cfg.VAL_RICH_WARN_PCT)
    short_warn = (RS_pct <= cfg.RS_SHORT_EXTREME_PCT) & (ADX_ROC <= 0) & (Val_pct <= cfg.VAL_CHEAP_WARN_PCT)

    out = []
    for i in range(len(tickers)):
        if bool(long_warn.iloc[i]):
            out.append("Long")
        elif bool(short_warn.iloc[i]):
            out.append("Short")
        else:
            out.append("None")
    return out

def write_backtests_summary(bt: Dict[str, Dict[str, object]], cfg: Config) -> None:
    with pd.ExcelWriter(cfg.OUTPUT_BACKTESTS_PATH, engine="openpyxl") as xw:
        for name, d in bt.items():
            eq = d.get("Equity", pd.Series(dtype=float))
            st = d.get("Stats", {})
            if not eq.empty:
                eq.to_frame(name="Equity").to_excel(xw, sheet_name=f"{name}_Equity")
            if st:
                pd.DataFrame([st]).to_excel(xw, sheet_name=f"{name}_Stats", index=False)
