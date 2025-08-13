from typing import Dict, List
import pandas as pd
from config import Config

def _descend(x):
    return x.sort_index(ascending=False)

def write_signals_daily(tickers: List[str],
                        features: Dict[str, pd.DataFrame],
                        trend: Dict[str, pd.DataFrame],
                        emergent_daily: pd.Series,
                        emergent_ttl_daily: pd.Series,
                        stars_daily: pd.Series,
                        unpredictable: pd.Series,
                        cfg: Config) -> None:
    last = features["RS_pct"].index[-1]

    def _val_warn(row_val: pd.Series) -> pd.Series:
        out = pd.Series("None", index=row_val.index, dtype=object)
        out[row_val >= cfg.VAL_RICH_WARN_PCT] = "Rich"
        out[row_val <= cfg.VAL_CHEAP_WARN_PCT] = "Cheap"
        return out

    row_RS      = features["RS_pct"].loc[last, tickers]
    row_accel   = features["Accel_pct"].loc[last, tickers]
    row_adxroc  = features["ADX_ROC"].loc[last, tickers]
    row_val     = features["Val_pct"].loc[last, tickers]
    row_val_warn = _val_warn(row_val)

    tstat_row   = trend["Tstat"].loc[last, tickers]
    slope_row   = trend["Slope"].loc[last, tickers]
    score_row   = trend["Score"].loc[last, tickers]
    class_row   = trend["Class"].loc[last, tickers]

    df = pd.DataFrame({
        "Ticker": tickers,
        "RS_pct": row_RS.values,
        "Trend_Tstat": tstat_row.values,
        "Trend_Slope": slope_row.values,
        "OnsideScore": score_row.values,   # t-stat percentile (0..1)
        "Trend_Class": class_row.values,   # Onside / Monitor / Offside
        "Accel_pct": row_accel.values,
        "ADX_ROC": row_adxroc.values,
        "Emergent": emergent_daily.reindex(tickers).fillna("").values,
        "Emergent_TTL_Rem": emergent_ttl_daily.reindex(tickers).fillna(0).astype(int).values,
        "Star": stars_daily.reindex(tickers).fillna("").values,
        "Val_pct": row_val.values,
        "Valuation_Warn": row_val_warn.values,
        "Unpredictable": unpredictable.reindex(tickers).fillna(False).astype(bool).values,
    })

    with pd.ExcelWriter(cfg.OUTPUT_SIGNALS_PATH, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Signals")

def write_backtests_summary(bt: Dict[str, Dict[str, object]], cfg: Config) -> None:
    with pd.ExcelWriter(cfg.OUTPUT_BACKTESTS_PATH, engine="openpyxl") as xw:
        for name, d in bt.items():
            eq = d.get("Equity", pd.Series(dtype=float))
            st = d.get("Stats", {})
            if not eq.empty:
                eq.to_frame(name="Equity").to_excel(xw, sheet_name=f"{name}_Equity")
            if st:
                pd.DataFrame([st]).to_excel(xw, sheet_name=f"{name}_Stats", index=False)
