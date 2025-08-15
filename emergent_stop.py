from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------- Logging helpers ----------------
def _dbg(cfg) -> bool:
    try:
        return bool(getattr(cfg, "EMERGENT_STOP_LOG", False))
    except Exception:
        return False

def _log(cfg, msg: str) -> None:
    if _dbg(cfg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] [emergent_stop] {msg}", flush=True)

# ---------------- Parameters ----------------
def _params_from_cfg(cfg) -> Dict[str, float]:
    return {
        "max_age_days": int(getattr(cfg, "EMERGENT_STOP_MAX_AGE_D", 15)),
        "min_lift": float(getattr(cfg, "EMERGENT_STOP_MIN_LIFT", 0.02)),
        "max_dd": float(getattr(cfg, "EMERGENT_STOP_MAX_DD", 0.03)),
        "cooldown_days": int(getattr(cfg, "EMERGENT_STOP_COOLDOWN_D", 10)),
    }

# ---------------- Core stop logic (per column) ----------------
def _apply_stoploss_with_cooldown(
    raw_flag: pd.Series,
    pseudo: pd.Series,
    max_age_days: int,
    min_lift: float,
    max_dd: float,
    cooldown_days: int,
) -> pd.Series:
    idx = raw_flag.index
    out = pd.Series(False, index=idx, dtype=bool)

    anchor: Optional[float] = None
    age = 0
    cooldown = 0

    for t in idx:
        if cooldown > 0:
            out.at[t] = False
            cooldown -= 1
            continue

        if not bool(raw_flag.get(t, False)):
            out.at[t] = False
            anchor = None
            age = 0
            continue

        p = pseudo.get(t, np.nan)
        if pd.isna(p):
            out.at[t] = anchor is not None
            continue

        if anchor is None:
            anchor = float(p)
            age = 1
            out.at[t] = True
            continue

        age += 1
        perf = float(p / anchor - 1.0)

        if perf <= -max_dd:
            out.at[t] = False
            anchor = None
            age = 0
            cooldown = cooldown_days
            continue

        if age >= max_age_days and perf < min_lift:
            out.at[t] = False
            anchor = None
            age = 0
            cooldown = cooldown_days
            continue

        out.at[t] = True

    return out

# ---------------- Utilities ----------------
def _validate_frame(df: Any, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"Stop overlay requires a non-empty DataFrame for {name}")
    return df

# ---------------- Public API ----------------
def apply_emergent_stop_overlays(
    emergent_ts: Any,
    features: Dict[str, pd.DataFrame],
    trend_ts: Any,  # kept for signature stability
    cfg: Any,
) -> Any:
    """
    Overlay stop and cooldown on top of BuyDip/FadeRally.
    Accepts the new strings and the legacy strings for safety.
    """
    # Idempotence guard
    if isinstance(emergent_ts, dict):
        meta = emergent_ts.get("Meta", {})
        if isinstance(meta, dict) and bool(meta.get("StopOverlayApplied", False)):
            _log(cfg, "overlay already applied, returning unchanged")
            return emergent_ts

    # Retrieve pseudo price
    pseudo = features.get("Pseudo", None)
    if pseudo is None:
        pseudo = features.get("pseudo", None)
    pseudo = _validate_frame(pseudo, "features['Pseudo']").astype(float).clip(lower=1e-12)
    pseudo = _validate_frame(pseudo, "features['Pseudo']").astype(float).clip(lower=1e-12)

    # Dict shape path
    if isinstance(emergent_ts, dict) and "State" in emergent_ts and "TTL_Rem" in emergent_ts:
        state0: pd.DataFrame = _validate_frame(emergent_ts["State"], "emergent_ts['State']")
        ttl0: pd.DataFrame = _validate_frame(emergent_ts["TTL_Rem"], "emergent_ts['TTL_Rem']").reindex(state0.index)

        # Align
        cols = [c for c in state0.columns if c in pseudo.columns]
        if not cols:
            raise ValueError("No overlapping tickers between emergent State and features['Pseudo']")

        state0 = state0[cols]
        ttl0 = ttl0[cols]
        P = pseudo[cols].reindex(state0.index).astype(float).clip(lower=1e-12)

        idx = state0.index
        new_state = pd.DataFrame("", index=idx, columns=cols, dtype=object)
        new_ttl = pd.DataFrame(0, index=idx, columns=cols, dtype=int)

        params = _params_from_cfg(cfg)

        raw_cnt_total = 0
        alive_cnt_total = 0

        for c in cols:
            s = state0[c].fillna("")
            # New strings
            raw_L = s.eq("BuyDip") | s.eq("Inflection")
            raw_S = s.eq("FadeRally") | s.eq("Breakdown")
            raw_cnt_total += int(raw_L.sum() + raw_S.sum())

            alive_L = _apply_stoploss_with_cooldown(
                raw_L, P[c],
                params["max_age_days"], params["min_lift"], params["max_dd"], params["cooldown_days"]
            )

            pseudo_short = (1.0 / P[c]).clip(lower=1e-12)
            alive_S = _apply_stoploss_with_cooldown(
                raw_S, pseudo_short,
                params["max_age_days"], params["min_lift"], params["max_dd"], params["cooldown_days"]
            )

            if alive_S.any():
                new_state.loc[alive_S, c] = "FadeRally"
            if alive_L.any():
                new_state.loc[alive_L, c] = "BuyDip"

            alive_any = (alive_L | alive_S)
            alive_cnt_total += int(alive_any.sum())
            if alive_any.any():
                new_ttl.loc[alive_any, c] = ttl0.loc[alive_any, c].astype(int)

        if _dbg(cfg):
            every = max(1, int(getattr(cfg, "EMERGENT_STOP_LOG_EVERY_D", 21)))
            for i, t in enumerate(idx):
                if (i % every) != 0:
                    continue
                raw_cnt = int(state0.iloc[i].isin(["BuyDip", "FadeRally", "Inflection", "Breakdown"]).sum())
                alive_cnt = int(new_state.iloc[i].isin(["BuyDip", "FadeRally"]).sum())
                _log(cfg, f"{t.date()} raw={raw_cnt} alive={alive_cnt} killed={raw_cnt - alive_cnt} params={params}")

        return {
            "State": new_state,
            "TTL_Rem": new_ttl,
            "Meta": {
                "StopOverlayApplied": True,
                "Params": params,
                "Summary": {
                    "raw_total": int(raw_cnt_total),
                    "alive_total": int(alive_cnt_total),
                    "killed_total": int(raw_cnt_total - alive_cnt_total),
                },
            },
        }

    # Backward compat: bare DataFrame path
    if isinstance(emergent_ts, pd.DataFrame):
        labels = _validate_frame(emergent_ts, "emergent labels DataFrame")
        cols = [c for c in labels.columns if c in pseudo.columns]
        if not cols:
            raise ValueError("No overlapping tickers between emergent label DataFrame and features['Pseudo']")
        params = _params_from_cfg(cfg)
        out = pd.DataFrame("", index=labels.index, columns=labels.columns, dtype=object)
        for c in cols:
            s = labels[c].astype(str)
            raw = s.str.lower().isin(["buydip", "inflection"])
            alive = _apply_stoploss_with_cooldown(
                raw.fillna(False),
                pseudo[c].astype(float).clip(lower=1e-12),
                params["max_age_days"], params["min_lift"], params["max_dd"], params["cooldown_days"],
            )
            out.loc[alive, c] = "BuyDip"
        return {
            "State": out,
            "TTL_Rem": pd.DataFrame(0, index=labels.index, columns=labels.columns, dtype=int),
            "Meta": {"StopOverlayApplied": True, "Params": params},
        }

    raise TypeError("Unsupported emergent_ts shape for stop overlay")
