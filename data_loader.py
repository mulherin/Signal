# data_loader.py
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

def load_trend_input(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    prices = pd.read_excel(path, sheet_name="Prices")
    # Normalize date column in Prices
    prices.rename(columns={prices.columns[0]: "Date"}, inplace=True)
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    prices = prices.dropna(subset=["Date"])
    prices.set_index("Date", inplace=True)
    prices.sort_index(inplace=True)

    tm = pd.read_excel(path, sheet_name="Ticker_Map")
    tm.columns = [str(c).strip() for c in tm.columns]
    assert "Ticker" in tm.columns and "Beta" in tm.columns, "Ticker_Map must have Ticker and Beta columns"
    tm = tm.dropna(subset=["Ticker"])
    tm["Ticker"] = tm["Ticker"].astype(str)

    tickers = [c for c in prices.columns if c in set(tm["Ticker"])]
    prices = prices[tickers]
    betas = tm.set_index("Ticker")["Beta"].astype(float).reindex(tickers)

    return prices, betas

def load_industry_map(path: Path) -> Optional[pd.Series]:
    """Optional: returns ticker->industry if 'Industry' exists; else None."""
    try:
        tm = pd.read_excel(path, sheet_name="Ticker_Map")
    except Exception:
        return None
    tm.columns = [str(c).strip() for c in tm.columns]
    if "Ticker" not in tm.columns or "Industry" not in tm.columns:
        return None
    tm = tm.dropna(subset=["Ticker"])
    tm["Ticker"] = tm["Ticker"].astype(str)
    ind = tm.set_index("Ticker")["Industry"].astype(str)
    return ind

# -------- robust date-column normalizer (used by valuation loader)
def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or df.shape[1] == 0:
        return df
    cols = list(df.columns)

    # Map any column whose name equals/contains date-ish text to 'date'
    def _is_date_like(name: str) -> bool:
        s = str(name).strip().lower()
        return (
            s == "date" or s == "dates" or s == "as of" or s == "asof" or
            " date" in s or s.startswith("date") or s.endswith("date")
        )

    # First pass: case-insensitive exact/contains match
    date_candidates = [c for c in cols if _is_date_like(c)]
    if date_candidates:
        df = df.rename(columns={date_candidates[0]: "date"})
    else:
        # Fallback: rename the first column to 'date'
        df = df.rename(columns={cols[0]: "date"})

    # Ensure datetime and drop NaT rows
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df

def load_valuation(workbook: Optional[Path],
                   sheet_name: Optional[str],
                   price_index: pd.DatetimeIndex,
                   tickers: list) -> pd.DataFrame:
    """
    Returns a DataFrame of valuation percentiles aligned to price_index & tickers.
    If workbook/sheet missing or unreadable, returns an empty (NaN) frame.
    """
    if workbook is None or sheet_name is None:
        return pd.DataFrame(index=price_index, columns=tickers, dtype=float)

    try:
        df = pd.read_excel(workbook, sheet_name=sheet_name)
    except Exception:
        # Graceful fallback: no valuation available
        return pd.DataFrame(index=price_index, columns=tickers, dtype=float)

    # Normalize date column robustly
    df = _normalize_date_column(df)

    # Set index and align
    df = df.set_index("date").sort_index()

    # Keep only known tickers; forward-fill then reindex to prices
    cols = [c for c in df.columns if c in tickers]
    if not cols:
        # No overlapping columns → return empty frame aligned to price index
        return pd.DataFrame(index=price_index, columns=tickers, dtype=float)

    df = df[cols].reindex(price_index).ffill()

    # Reinsert any missing tickers as all-NaN columns to keep shape consistent
    missing = [t for t in tickers if t not in df.columns]
    for t in missing:
        df[t] = np.nan

    # Order columns to match tickers
    df = df.reindex(columns=tickers)
    return df
