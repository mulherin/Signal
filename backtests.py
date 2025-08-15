from typing import Dict
import pandas as pd
from config import Config
from trend_signal import backtest_trend
from emergent_signal import backtest_emergent
from stars_signal import backtest_stars

# Enforce stop overlay inside backtests so it cannot be bypassed
from emergent_stop import apply_emergent_stop_overlays


def run_all_backtests(features: Dict[str, pd.DataFrame],
                      trend_ts: Dict[str, pd.DataFrame],
                      emergent_ts: Dict[str, pd.DataFrame],
                      star_ts: pd.DataFrame,
                      cfg: Config) -> Dict[str, Dict[str, object]]:

    resid = features["Resid"]

    # Trend
    eq_trend, st_trend = backtest_trend(resid, trend_ts, cfg)

    # Emergent - defensively apply stop overlay here (idempotent)
    emergent_ts_bt = apply_emergent_stop_overlays(emergent_ts, features, trend_ts, cfg)

    # Hard guard: do not allow backtests to proceed if overlay did not apply
    if not isinstance(emergent_ts_bt, dict) or not emergent_ts_bt.get("Meta", {}).get("StopOverlayApplied", False):
        raise RuntimeError("Emergent stop overlay did not apply to the Emergent time series. Aborting backtests to avoid stale results")

    eq_emergent, st_emergent = backtest_emergent(resid, emergent_ts_bt, cfg)

    # Stars
    eq_stars, st_stars = backtest_stars(resid, star_ts)

    return {
        "Trend": {"Equity": eq_trend, "Stats": st_trend},
        "Emergent": {"Equity": eq_emergent, "Stats": st_emergent},
        "Stars": {"Equity": eq_stars, "Stats": st_stars},
    }
