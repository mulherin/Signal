# data_loader.py
# Copy-paste ready

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

def load_trend_input(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      prices: DataFrame indexed by Date with ticker columns
      betas: Series indexed by ticker with float beta
    """
    prices = pd.read_excel(path, sheet_name="Prices")
    prices.rename(columns={prices.columns[0]: "Date"}, inplace=True)
    prices["Date"] = pd.to_datetime(prices["Date"])
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

def load_valuation(workbook: Optional[Path], sheet_name: Optional[str], price_index: pd.DatetimeIndex, tickers: list) -> pd.DataFrame:
    """
    Reads EV metric by ticker with Date column header 'date' in A1.
    Returns DataFrame aligned to price_index with same columns, forward filled.
    """
    if workbook is None or sheet_name is None:
        return pd.DataFrame(index=price_index, columns=tickers, dtype=float)

    df = pd.read_excel(workbook, sheet_name=sheet_name)
    # Expect column A labeled 'date'
    first_col = str(df.columns[0]).strip().lower()
    if first_col != "date":
        # Try to find a column named 'date'
        date_cols = [c for c in df.columns if str(c).strip().lower() == "date"]
        if date_cols:
            df.rename(columns={date_cols[0]: "date"}, inplace=True)
        else:
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
    else:
        df.rename(columns={df.columns[0]: "date"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    cols = [c for c in df.columns if c in tickers]
    df = df[cols]
    df = df.reindex(price_index).ffill()
    return df
