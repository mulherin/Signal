from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# ---- helpers ----
def _align_three(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = a.index.intersection(b.index).intersection(c.index)
    cols = a.columns.intersection(b.columns).intersection(c.columns)
    return a.loc[idx, cols], b.loc[idx, cols], c.loc[idx, cols]

def _kday_shock_z(resid: pd.DataFrame, k: int, vol_win: int) -> pd.DataFrame:
    # k-day residual sum standardized by the rolling std of the k-sum
    ksum = resid.rolling(int(k)).sum()
    vol  = ksum.rolling(int(vol_win)).std(ddof=0).replace(0.0, np.nan)
    return ksum.divide(vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# ---- core ----
from utils import cross_sectional_percentile  # add this import at top if not present

# add at the top of the file if not already present
from utils import cross_sectional_percentile

def compute_emergent_time_series(RS_med_pct: pd.DataFrame,
                                 resid: pd.DataFrame,
                                 pseudo: pd.DataFrame,
                                 industry_map: Optional[pd.Series],
                                 cfg) -> Dict[str, pd.DataFrame]:
    """
    Minimal Emergent (3 knobs):
      X = EMERGENT_PULSE_LOOKBACK_D
      Y = EMERGENT_FADE_TOP_PCT
      Z = EMERGENT_BUY_BOT_PCT

      - k-day residual move using Pseudo: kret = pseudo / pseudo.shift(k) - 1
      - Cross-sectional percentile each day (0..1)
      - Fire on *crossings* only:
          FadeRally  when pct crosses up >= Y and kret > 0
          BuyDip     when pct crosses down <= Z and kret < 0
      - Trend gating remains external (main.py/backtests.py).
    """
    # Align
    RS_med_pct, resid, pseudo = _align_three(RS_med_pct.astype(float), resid.astype(float), pseudo.astype(float))
    dates, names = resid.index, resid.columns

    # knobs
    k_look   = int(getattr(cfg, "EMERGENT_PULSE_LOOKBACK_D", 5))   # X
    fade_top = float(getattr(cfg, "EMERGENT_FADE_TOP_PCT", 0.90))  # Y
    buy_bot  = float(getattr(cfg, "EMERGENT_BUY_BOT_PCT", 0.10))   # Z

    # k-day move (residual pseudo-price)
    kret = (pseudo / pseudo.shift(k_look)) - 1.0

    # cross-section percentile + crossings
    pct = cross_sectional_percentile(kret).fillna(0.5)
    pct_prev = pct.shift(1)

    cross_up   = (pct >= fade_top) & ~(pct_prev >= fade_top)
    cross_down = (pct <= buy_bot)  & ~(pct_prev <= buy_bot)

    rally = cross_up   & (kret > 0)
    dip   = cross_down & (kret < 0)

    State = pd.DataFrame("", index=dates, columns=names, dtype=object)
    State[rally] = "FadeRally"
    State[dip]   = "BuyDip"

    TTL_Rem = pd.DataFrame(0, index=dates, columns=names, dtype=int)  # carry handled later
    return {"State": State, "TTL_Rem": TTL_Rem}
