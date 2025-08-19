# data_loader.py
# Robust loaders for the streamlined Signals stack.
# - All inputs default to the unified workbook: signals_input.xlsm (see Config).
# - Flexible sheet detection (case-insensitive, common aliases).
# - Returns shapes compatible with downstream modules:
#     load_trend_input(path) -> (prices: DataFrame[dates x tickers], betas: Series[ticker])
#     load_valuation(path, sheet_name, idx, tickers) -> DataFrame[dates x tickers] (percentiles 0..1 if possible)
#     load_industry_map(path) -> Series[ticker -> industry string]  (or empty Series if missing)

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, List
import warnings

import numpy as np
import pandas as pd


# ----------------------- helpers: Excel sheet detection -----------------------

def _find_sheet(xl: pd.ExcelFile, candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first sheet in 'xl' whose name (case-insensitive) matches any of the candidate names.
    """
    lower_map = {str(s).strip().lower(): s for s in xl.sheet_names}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _coerce_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to coerce a 'Date' column (or the first column) to a DateTimeIndex, sort ascending.
    """
    if df is None or df.empty:
        return df

    cols = [str(c) for c in df.columns]
    if "date" in [c.lower() for c in cols]:
        # Use explicit Date column
        dcol = [c for c in cols if c.lower() == "date"][0]
        out = df.copy()
        out[dcol] = pd.to_datetime(out[dcol])
        out = out.set_index(dcol)
    else:
        # Attempt: first column looks like dates?
        out = df.copy()
        try:
            out.iloc[:, 0] = pd.to_datetime(out.iloc[:, 0])
            out = out.set_index(out.columns[0])
        except Exception:
            # Leave as-is (caller may pivot)
            return df

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _drop_all_na_rows_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.dropna(how="all").dropna(axis=1, how="all")


def _to_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ----------------------- core: trend input (prices + betas) -------------------

def load_trend_input(xlsx_path: Path | str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load level-adjusted prices (wide, dates x tickers) and betas (Series indexed by ticker).
    Expected sheet aliases (case-insensitive):
      Prices:  ["Prices", "AdjClose", "Adj Close", "Close", "PX_LAST", "Price", "Levels"]
      Betas:   ["Betas", "Beta", "Market_Beta", "Beta_Map"]
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Trend input workbook not found: {xlsx_path}")

    xl = pd.ExcelFile(xlsx_path)

    # ---- PRICES ----
    sh_prices = _find_sheet(xl, ["Prices", "AdjClose", "Adj Close", "Close", "PX_LAST", "Price", "Levels"])
    if sh_prices is None:
        raise ValueError(
            f"No 'Prices' sheet found in {xlsx_path}. "
            f"Tried aliases: Prices / AdjClose / Close / PX_LAST / Price / Levels"
        )

    dfp = xl.parse(sh_prices)
    dfp = _drop_all_na_rows_cols(dfp)
    dfp = _to_string_columns(dfp)

    # If it's already wide with a Date column, coerce index
    dfp_idx = _coerce_datetime_index(dfp)
    if dfp_idx is not None and isinstance(dfp_idx.index, pd.DatetimeIndex) and dfp_idx.shape[1] >= 1:
        prices = dfp_idx.astype(float)
    else:
        # Try tidy → wide: expect columns ["Date","Ticker","Price"] (any order)
        cols_lower = {c.lower(): c for c in dfp.columns}
        if {"date", "ticker"}.issubset(set(cols_lower.keys())):
            vcol = cols_lower.get("price", None)
            if vcol is None:
                # pick the first non-key column as value
                value_candidates = [c for c in dfp.columns if c not in {cols_lower["date"], cols_lower["ticker"]}]
                if not value_candidates:
                    raise ValueError(f"Prices sheet '{sh_prices}' does not contain a numeric value column.")
                vcol = value_candidates[0]
            tmp = dfp.rename(columns={cols_lower["date"]: "Date", cols_lower["ticker"]: "Ticker", vcol: "Price"})
            tmp["Date"] = pd.to_datetime(tmp["Date"])
            prices = tmp.pivot(index="Date", columns="Ticker", values="Price").sort_index()
        else:
            raise ValueError(
                f"Prices sheet '{sh_prices}' is neither wide (with a Date column) nor tidy (Date/Ticker/Price)."
            )

    prices = prices.sort_index().loc[~prices.index.duplicated(keep="last")]
    prices.columns = [str(c).strip() for c in prices.columns]

    # ---- BETAS ----
    sh_betas = _find_sheet(
    xl,
    [
        "Betas", "Beta", "Market_Beta", "Beta_Map",
        "Ticker_Map", "Ticker Map", "TickerMap", "ticker_map"
    ],
    )

    if sh_betas is None:
        warnings.warn(
            f"No 'Betas' sheet found in {xlsx_path}. Defaulting all betas to 1.0.",
            RuntimeWarning,
        )
        betas = pd.Series(1.0, index=prices.columns, dtype=float)
        return prices, betas

    dfb = xl.parse(sh_betas)
    dfb = _drop_all_na_rows_cols(dfb)
    dfb = _to_string_columns(dfb)

    betas: pd.Series

    # Case A: columns include Ticker + (Beta or Value)
    cols_lower = {c.lower(): c for c in dfb.columns}
    if "ticker" in cols_lower and ("beta" in cols_lower or "value" in cols_lower):
        bcol = cols_lower.get("beta", cols_lower.get("value"))
        betas = dfb.set_index(cols_lower["ticker"])[bcol].astype(float)
    # Case B: first column looks like ticker strings, second numeric
    elif dfb.shape[1] >= 2 and dfb.iloc[:, 0].dtype == object:
        tick = dfb.iloc[:, 0].astype(str).str.strip()
        val = pd.to_numeric(dfb.iloc[:, 1], errors="coerce")
        betas = pd.Series(val.values, index=tick.values, dtype=float).dropna()
    # Case C: single-row wide where columns are tickers
    elif dfb.shape[0] == 1:
        betas = dfb.iloc[0].astype(float)
        betas.index = [str(c).strip() for c in dfb.columns]
    # Case D: single-column with named index (row labels are tickers)
    elif dfb.shape[1] == 1 and dfb.index.name and dfb.index.name.lower() in ("ticker", "tickers"):
        betas = dfb.iloc[:, 0].astype(float)
        betas.index = [str(i).strip() for i in dfb.index]
    else:
        warnings.warn(
            f"Unrecognized Betas format on sheet '{sh_betas}'. Defaulting to 1.0.",
            RuntimeWarning,
        )
        betas = pd.Series(1.0, index=prices.columns, dtype=float)

    # Align to prices' columns; fill missing with 1.0
    betas = betas.reindex(prices.columns).fillna(1.0).astype(float)

    return prices, betas


# ----------------------- optional: valuation (percentiles) --------------------

def _to_wide_from_tidy(df: pd.DataFrame, date_col: str, ticker_col: str, value_col: str) -> pd.DataFrame:
    tmp = df.rename(columns={date_col: "Date", ticker_col: "Ticker", value_col: "Val"})
    tmp["Date"] = pd.to_datetime(tmp["Date"])
    wide = tmp.pivot(index="Date", columns="Ticker", values="Val").sort_index()
    return wide


def _ensure_percentile(df: pd.DataFrame) -> pd.DataFrame:
    """
    If values appear outside [0,1], convert each row to percentiles (0..1) across tickers.
    """
    if df.empty:
        return df
    vmin, vmax = float(np.nanmin(df.values)), float(np.nanmax(df.values))
    if (vmin >= 0.0) and (vmax <= 1.0):
        return df.astype(float)

    # Convert to cross-sectional ranks per day: higher value => higher percentile (richer)
    ranked = df.rank(axis=1, pct=True)
    return ranked.astype(float)


def load_valuation(xlsx_path: Optional[Path | str],
                   sheet_name: Optional[str],
                   index: pd.DatetimeIndex,
                   tickers: List[str]) -> pd.DataFrame:
    """
    Load an optional valuation panel (dates x tickers). If sheet is missing or invalid → empty DataFrame.
    - If values are not in [0,1], convert each row to cross-sectional percentiles.
    - Align to provided (index, tickers) and forward-fill over time.
    Expected sheet aliases if 'sheet_name' is None:
      ["Valuation", "Val_pct", "Valuation_pct", "ValuationPercentile", "Valuation_Pct"]
    """
    if xlsx_path is None:
        return pd.DataFrame(index=index, columns=tickers, dtype=float)

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        warnings.warn(f"Valuation workbook not found: {xlsx_path}. Skipping valuation.", RuntimeWarning)
        return pd.DataFrame(index=index, columns=tickers, dtype=float)

    xl = pd.ExcelFile(xlsx_path)
    sh_val = sheet_name or _find_sheet(xl, ["Valuation", "Val_pct", "Valuation_pct", "ValuationPercentile", "Valuation_Pct"])
    if sh_val is None:
        warnings.warn(f"No valuation sheet found in {xlsx_path}. Skipping valuation.", RuntimeWarning)
        return pd.DataFrame(index=index, columns=tickers, dtype=float)

    df = xl.parse(sh_val)
    df = _drop_all_na_rows_cols(df)
    df = _to_string_columns(df)

    # Wide with Date column
    df_idx = _coerce_datetime_index(df)
    if isinstance(df_idx.index, pd.DatetimeIndex) and df_idx.shape[1] >= 1:
        wide = df_idx
    else:
        # Tidy → wide heuristics
        cols_lower = {c.lower(): c for c in df.columns}
        if {"date", "ticker"}.issubset(set(cols_lower.keys())):
            # pick value column
            value_col = None
            for cand in ["val_pct", "value", "val", "score"]:
                if cand in cols_lower:
                    value_col = cols_lower[cand]
                    break
            if value_col is None:
                # default to the first non-key column
                value_candidates = [c for c in df.columns if c not in {cols_lower["date"], cols_lower["ticker"]}]
                if not value_candidates:
                    warnings.warn(f"Valuation sheet '{sh_val}': no numeric value column found.", RuntimeWarning)
                    return pd.DataFrame(index=index, columns=tickers, dtype=float)
                value_col = value_candidates[0]
            wide = _to_wide_from_tidy(df, cols_lower["date"], cols_lower["ticker"], value_col)
        else:
            warnings.warn(
                f"Valuation sheet '{sh_val}' is neither wide (Date as a column) nor tidy (Date/Ticker/Value).",
                RuntimeWarning,
            )
            return pd.DataFrame(index=index, columns=tickers, dtype=float)

    # Ensure percentiles and align
    wide = _ensure_percentile(wide)
    wide = wide.reindex(index=index, columns=[str(t) for t in tickers])
    wide = wide.ffill().astype(float)
    return wide


# ----------------------- optional: industry map (ticker -> group) ------------

def load_industry_map(xlsx_path: Path | str) -> pd.Series:
    """
    Load an optional mapping Ticker -> Industry (string). Returns an empty Series if not found.
    Expected sheet aliases (case-insensitive):
      ["Industry", "Industries", "Sector", "Sectors", "GICS", "Industry_Map"]
    Acceptable formats:
      - Two columns: [Ticker, Industry]
      - Any table with columns containing 'Ticker' and one of {'Industry','Sector','GICS'}.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        return pd.Series(dtype=object)

    xl = pd.ExcelFile(xlsx_path)
    sh = _find_sheet(
    xl,
    [
        "Industry", "Industries", "Sector", "Sectors", "GICS", "Industry_Map",
        "Ticker_Map", "Ticker Map", "TickerMap", "ticker_map"
    ],
    )

    if sh is None:
        return pd.Series(dtype=object)

    df = xl.parse(sh)
    df = _drop_all_na_rows_cols(df)
    df = _to_string_columns(df)

    # prefer explicit columns
    cols_lower = {c.lower(): c for c in df.columns}
    tcol = cols_lower.get("ticker", None)

    icol = None
    for cand in ["industry", "sector", "gics", "group"]:
        if cand in cols_lower:
            icol = cols_lower[cand]
            break

    if tcol is not None and icol is not None:
        m = df[[tcol, icol]].dropna()
        m[tcol] = m[tcol].astype(str).str.strip()
        m[icol] = m[icol].astype(str).str.strip()
        # deduplicate by last occurrence
        m = m.drop_duplicates(subset=[tcol], keep="last")
        out = pd.Series(m[icol].values, index=m[tcol].values)
        return out

    # fallback: two-column generic
    if df.shape[1] >= 2:
        tick = df.iloc[:, 0].astype(str).str.strip()
        grp = df.iloc[:, 1].astype(str).str.strip()
        m = pd.Series(grp.values, index=tick.values)
        m = m[~m.index.duplicated(keep="last")]
        return m

    return pd.Series(dtype=object)
