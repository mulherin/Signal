from typing import Dict
import pandas as pd
from config import Config
from trend_signal import backtest_trend
from emergent_signal import backtest_emergent
from stars_signal import backtest_stars

def run_all_backtests(features: Dict[str, pd.DataFrame],
                      trend_ts: Dict[str, pd.DataFrame],
                      emergent_ts: Dict[str, pd.DataFrame],
                      star_ts: pd.DataFrame,
                      cfg: Config) -> Dict[str, Dict[str, object]]:

    resid = features["Resid"]

    # Trend (minimal)
    eq_trend, st_trend = backtest_trend(resid, trend_ts, cfg)

    # Emergent & Stars (unchanged)
    eq_emergent, st_emergent = backtest_emergent(resid, emergent_ts, cfg)
    eq_stars, st_stars = backtest_stars(resid, star_ts)

    return {
        "Trend": {"Equity": eq_trend, "Stats": st_trend},
        "Emergent": {"Equity": eq_emergent, "Stats": st_emergent},
        "Stars": {"Equity": eq_stars, "Stats": st_stars},
    }
