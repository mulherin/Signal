# Streamlined daily Signals workbook writer.
# - Drops ADX/ADX_ROC and Accel from outputs (we removed them from features).
# - Renames TTL to a clear, user-facing column: "trade_Age_D".
# - Uses Trend fields (Tstat, Slope, Score, Class) and current-day labels for Emergent and Stars.

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from config import Config


def write_signals_daily(tickers: List[str],
                        features: Dict[str, pd.DataFrame],
                        trend: Dict[str, pd.DataFrame],
                        emergent_daily: pd.Series,
                        emergent_ttl_daily: pd.Series,
                        stars_daily: pd.Series,
                        unpredictable: pd.Series,
                        cfg: Config,
                        emergent_age_daily: Optional[pd.Series] = None,
                        stars_ts: Optional[pd.DataFrame] = None,
                        report_date: Optional[pd.Timestamp] = None,
                        trend_ind: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    """
    Create a single 'Signals' sheet with trader-facing fields only.

    If report_date is provided and exists in the feature index, use that date.
    Otherwise use the latest available date.

    If trend_ind is provided, include industry-relative trend columns.
    """
    rs_idx = features["RS_pct"].index
    last = report_date if (report_date is not None and report_date in rs_idx) else rs_idx[-1]

    # ---- helpers ----
    def _val_warn(row_val: pd.Series) -> pd.Series:
        out = pd.Series("None", index=row_val.index, dtype=object)
        out[row_val >= cfg.VAL_RICH_WARN_PCT] = "Rich"
        out[row_val <= cfg.VAL_CHEAP_WARN_PCT] = "Cheap"
        return out

    def _trend_dir_5d(trend_class_ts: pd.DataFrame, names: List[str], last_date: pd.Timestamp) -> pd.Series:
        if trend_class_ts is None or trend_class_ts.empty or last_date not in trend_class_ts.index:
            return pd.Series("", index=names, dtype=object)
        pos = trend_class_ts.index.get_loc(last_date)
        if isinstance(pos, slice) or (isinstance(pos, (list, np.ndarray)) and len(pos) != 1):
            return pd.Series("", index=names, dtype=object)
        idx = int(pos)
        if idx < 5:
            return pd.Series("", index=names, dtype=object)

        curr = trend_class_ts.iloc[idx].reindex(names).astype(str).fillna("")
        prev = trend_class_ts.iloc[idx - 5].reindex(names).astype(str).fillna("")
        rank = {"Offside": 1, "Monitor": 2, "Onside": 3}
        out = {}
        for n in names:
            a = rank.get(prev.get(n, ""), None)
            b = rank.get(curr.get(n, ""), None)
            out[n] = "" if (a is None or b is None or a == b) else ("up" if b > a else "down")
        return pd.Series(out, index=names, dtype=object)

    def _star_changed_5d(star_ts: Optional[pd.DataFrame], names: List[str], last_date: pd.Timestamp) -> pd.Series:
        if star_ts is None or star_ts.empty or last_date not in star_ts.index:
            return pd.Series(False, index=names, dtype=bool)
        pos = star_ts.index.get_loc(last_date)
        if isinstance(pos, slice) or (isinstance(pos, (list, np.ndarray)) and len(pos) != 1):
            return pd.Series(False, index=names, dtype=bool)
        idx = int(pos)
        if idx < 5:
            return pd.Series(False, index=names, dtype=bool)
        curr = star_ts.iloc[idx].reindex(names).astype(str).fillna("")
        prev = star_ts.iloc[idx - 5].reindex(names).astype(str).fillna("")
        return curr.ne(prev)

    # ---- feature rows ----
    row_RS = features["RS_pct"].loc[last, tickers]
    if "Val_pct" in features and not features["Val_pct"].empty:
        row_val = features["Val_pct"].loc[last, tickers]
    else:
        row_val = pd.Series(0.5, index=tickers, dtype=float)  # neutral if valuation missing
    row_val_warn = _val_warn(row_val)

    # ---- trend rows (global) ----
    tstat_row = trend["Tstat"].loc[last, tickers]
    slope_row = trend["Slope"].loc[last, tickers]
    score_row = trend["Score"].loc[last, tickers]
    class_row_raw = trend["Class"].loc[last, tickers].fillna("").astype(str)

    class_display_map = {"Onside": "1-onside", "Monitor": "2-monitor", "Offside": "3-offside"}
    class_row = class_row_raw.map(class_display_map).fillna("")

    trend_class_ts = trend["Class"].reindex(features["RS_pct"].index)
    trend_change_5d = _trend_dir_5d(trend_class_ts, tickers, last)

    # ---- emergent + stars (current-day labels) ----
    emergent_now = emergent_daily.reindex(tickers).fillna("").astype(str)
    trade_age = (emergent_age_daily if emergent_age_daily is not None else emergent_ttl_daily)
    trade_age = trade_age.reindex(tickers).fillna(0).astype(int) if trade_age is not None else pd.Series(0, index=tickers, dtype=int)

    stars_now = stars_daily.reindex(tickers).fillna("").astype(str)
    unpred_now = unpredictable.reindex(tickers).fillna(False).astype(bool)
    star_changed_5d = _star_changed_5d(stars_ts, tickers, last)

    # ---- base output dict ----
    df_dict = {
        "Ticker": tickers,
        "RS_pct": row_RS.values,
        "Trend_Tstat": tstat_row.values,
        "Trend_Slope": slope_row.values,
        "OnsideScore": score_row.values,
        "Trend_Class": class_row.values,
        "Trend_Class_5D_Change": trend_change_5d.values,
        "Emergent": emergent_now.values,
        "trade_Age_D": trade_age.values,
        "Star": stars_now.values,
        "Star_5D_Changed": star_changed_5d.values,
        "Val_pct": row_val.values,
        "Valuation_Warn": row_val_warn.values,
        "Unpredictable": unpred_now.values,
    }

    # ---- optional: industry-relative trend block ----
    if trend_ind is not None and all(k in trend_ind for k in ["Tstat", "Slope", "Score", "Class"]):
        tstat_ind_row = trend_ind["Tstat"].loc[last, tickers]
        slope_ind_row = trend_ind["Slope"].loc[last, tickers]
        score_ind_row = trend_ind["Score"].loc[last, tickers]
        class_ind_raw = trend_ind["Class"].loc[last, tickers].fillna("").astype(str)
        class_ind = class_ind_raw.map(class_display_map).fillna("")
        trend_ind_class_ts = trend_ind["Class"].reindex(features["RS_pct"].index)
        trend_ind_change_5d = _trend_dir_5d(trend_ind_class_ts, tickers, last)

        df_dict.update({
            "Trend_Ind_Tstat": tstat_ind_row.values,
            "Trend_Ind_Slope": slope_ind_row.values,
            "Trend_Ind_Score": score_ind_row.values,
            "Trend_Ind_Class": class_ind.values,
            "Trend_Ind_Class_5D_Change": trend_ind_change_5d.values,
        })

    # ---- write workbook ----
    df = pd.DataFrame(df_dict)
    with pd.ExcelWriter(cfg.OUTPUT_SIGNALS_PATH, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Signals")


def write_backtests_summary(bt: Dict[str, Dict[str, object]], cfg: Config) -> None:
    """
    Write one sheet per strategy for equity and stats into OUTPUT_BACKTESTS_PATH.
    Expected 'bt' structure: { name: {"Equity": Series, "Stats": dict}, ... }
    """
    with pd.ExcelWriter(cfg.OUTPUT_BACKTESTS_PATH, engine="openpyxl") as xw:
        for name, d in bt.items():
            eq = d.get("Equity", pd.Series(dtype=float))
            st = d.get("Stats", {})
            if isinstance(eq, pd.Series) and not eq.empty:
                eq.to_frame(name="Equity").to_excel(xw, sheet_name=f"{name}_Equity")
            if isinstance(st, dict) and st:
                pd.DataFrame([st]).to_excel(xw, sheet_name=f"{name}_Stats", index=False)
