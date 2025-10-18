# scripts/phase5_backtest.py
# ---------------------------------------------------------------------
# Rules-only backtest with robust, tz-aware timestamp handling.
# Accepts overrides for filter toggles and spread caps so sweep scripts
# can call: run_backtest(df, USE_SESSION=True, SPREAD_TO_ATR_CAP=10, ...)
# ---------------------------------------------------------------------

import json
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd


# =========================
# Paths & basic setup
# =========================
DATA_DIR     = Path("data")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

FEAT_PATH    = DATA_DIR / "m15_features.parquet"
OUT_TRADES   = RESULTS_DIR / "trades.csv"
OUT_SUMMARY  = RESULTS_DIR / "backtest_summary.json"

# =========================
# Column map, adjust if needed
# =========================
COL = {
    "ts":            "ts",
    "m15_close":     "m15_close",
    "m15_ema20":     "m15_ema20",
    "m15_atr14":     "m15_atr14",
    "spread_to_atr": "m15_spread_to_atr",
    "spread_pts":    "spread_points",
    "h1_close":      "h1_close",
    "h1_sma20":      "h1_sma20",
    "h1_sma200":     "h1_sma200",
    "h1_adx14":      "h1_adx14",
    "h1_atr_pct":    "h1_atr_pct",
    "h1_obv":        "h1_obv",
    "h1_obv_slope":  "h1_obv_slope_6",
}

# =========================
# Default filter toggles
# =========================
USE_SESSION      = True
USE_REGIME       = False
USE_RANGE_SKIP   = True
USE_ADX_FILTER   = True
USE_OBV_PCT      = True
USE_OBV_CONFIRM  = True

# =========================
# Default spread constraints
# =========================
MAX_SPREAD_POINTS  = 30
SPREAD_TO_ATR_CAP  = 20

# =========================
# Session, Malaysia time
# =========================
TZ_OFFSET_HOURS             = +8
SESSION_START_H             = 9
SESSION_END_H               = 3
SHORT_BLACKOUT_FROM_START_H = 2
SHORT_BLACKOUT_BEFORE_END_H = 1
LONG_BLOCK_START_H          = 21
LONG_BLOCK_END_H            = 24

# =========================
# Regime and Range thresholds
# =========================
RANGE_SEP_ATR_LONG   = 0.35
RANGE_SEP_ATR_SHORT  = 0.45

# =========================
# ADX thresholds
# =========================
ADX_LONG_MIN  = 22
ADX_SHORT_MIN = 30

# =========================
# Outcome proxy
# =========================
FUTURE_BARS_HORIZON = 4
R_CLIP              = 3.0


# =====================================================================
# Helpers
# =====================================================================

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    if COL["spread_to_atr"] not in df.columns:
        warnings.warn(f"'{COL['spread_to_atr']}' missing, deriving as zeros.")
        df[COL["spread_to_atr"]] = 0.0

    if COL["spread_pts"] not in df.columns:
        warnings.warn(f"'{COL['spread_pts']}' missing, deriving as zeros.")
        df[COL["spread_pts"]] = 0.0

    if COL["m15_atr14"] not in df.columns:
        warnings.warn(f"'{COL['m15_atr14']}' missing, deriving as tiny epsilon.")
        df[COL["m15_atr14"]] = 1e-6

    for k in ["m15_close", "m15_ema20", "h1_sma20", "h1_sma200"]:
        c = COL[k]
        if c not in df.columns:
            warnings.warn(f"'{c}' missing, deriving as zeros.")
            df[c] = 0.0

    for k in ["h1_adx14", "h1_atr_pct", "h1_obv", "h1_obv_slope"]:
        c = COL[k]
        if c not in df.columns:
            warnings.warn(f"Optional column '{c}' not found, related filters will be relaxed.")
    return df


def _get_ts_utc(df: pd.DataFrame, ts_col: str, bar_minutes: int = 15) -> pd.Series:
    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
    else:
        warnings.warn(f"'{ts_col}' missing, generating synthetic timestamps assuming {bar_minutes}-min bars.")
        ts = pd.date_range(
            "2000-01-01 00:00:00",
            periods=len(df),
            freq=f"{bar_minutes}min",
            tz="UTC",
        )
        ts = pd.Series(ts, index=df.index)
    return ts


def _session_masks(df: pd.DataFrame):
    ts = _get_ts_utc(df, COL["ts"], bar_minutes=15)
    myt = ts + pd.Timedelta(hours=TZ_OFFSET_HOURS)
    h = myt.dt.hour

    if SESSION_START_H <= SESSION_END_H:
        session_ok = (h >= SESSION_START_H) & (h < SESSION_END_H)
    else:
        session_ok = (h >= SESSION_START_H) | (h < SESSION_END_H)

    long_block  = (h >= LONG_BLOCK_START_H) & (h < LONG_BLOCK_END_H)
    short_block = (h >= SHORT_BLACKOUT_FROM_START_H) & (h < SHORT_BLACKOUT_BEFORE_END_H)

    return session_ok.fillna(False), long_block.fillna(False), short_block.fillna(False)


def _range_separation(df: pd.DataFrame):
    if COL["h1_atr_pct"] in df.columns:
        sep = pd.to_numeric(df[COL["h1_atr_pct"]], errors="coerce")
    else:
        base = pd.to_numeric(df.get(COL["h1_close"], df[COL["m15_close"]]), errors="coerce").replace(0, np.nan)
        sep = (pd.to_numeric(df[COL["m15_atr14"]], errors="coerce") / base).abs()

    rng_long  = (sep >= RANGE_SEP_ATR_LONG).fillna(False)
    rng_short = (sep >= RANGE_SEP_ATR_SHORT).fillna(False)
    return rng_long, rng_short


def _obv_confirms(df: pd.DataFrame):
    if COL["h1_obv_slope"] not in df.columns:
        return pd.Series(True, index=df.index), pd.Series(True, index=df.index)
    slope = pd.to_numeric(df[COL["h1_obv_slope"]], errors="coerce")
    long_ok  = (slope > 0).fillna(False)
    short_ok = (slope < 0).fillna(False)
    return long_ok, short_ok


# =====================================================================
# Parameter normalization
# =====================================================================

def _normalize_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    norm = {}
    norm["use_session"] = bool(overrides.get("USE_SESSION", USE_SESSION))
    norm["use_regime"]  = bool(overrides.get("USE_REGIME_FILTER", overrides.get("USE_REGIME", USE_REGIME)))
    norm["use_range"]   = bool(overrides.get("USE_RANGE_SKIP", USE_RANGE_SKIP))
    norm["use_adx"]     = bool(overrides.get("USE_ADX_FILTER", USE_ADX_FILTER))
    norm["use_obv_pct"] = bool(overrides.get("USE_OBV_PCT", USE_OBV_PCT))
    norm["use_obv_cfm"] = bool(overrides.get("USE_OBV_CONFIRM", USE_OBV_CONFIRM))
    norm["cap_to_atr"]  = float(overrides.get("SPREAD_TO_ATR_CAP", SPREAD_TO_ATR_CAP))
    norm["max_pts"]     = int(overrides.get("MAX_SPREAD_POINTS", MAX_SPREAD_POINTS))
    return norm


# =====================================================================
# Backtest core
# =====================================================================

def run_backtest(df: pd.DataFrame, **overrides) -> Tuple[pd.DataFrame, dict]:
    P = _normalize_overrides(overrides)
    df = _ensure_cols(df)

    spread = pd.to_numeric(df[COL["spread_to_atr"]], errors="coerce").fillna(np.inf)
    pts    = pd.to_numeric(df[COL["spread_pts"]],    errors="coerce").fillna(np.inf)

    print("\n=== Spread/ATR diagnostics, all bars ===")
    print(spread.describe(percentiles=[0.10, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())

    cap_keep = spread <= P["cap_to_atr"]
    kept     = int(cap_keep.sum())
    rem      = len(df) - kept
    pct      = (rem / len(df) * 100.0)
    print(f"Cap = {P['cap_to_atr']:g} -> kept {kept:,}, removed {rem:,} ({pct:.1f}% removed)")

    pts_keep = pts <= P["max_pts"]
    kept2    = int(pts_keep.sum())
    rem2     = len(df) - kept2
    pct2     = (rem2 / len(df) * 100.0)
    print(f"MAX_SPREAD_POINTS = {P['max_pts']} -> kept {kept2:,}, removed {rem2:,} ({pct2:.1f}% removed)")

    spread_ok = cap_keep & pts_keep

    if P["use_session"]:
        session_ok, long_blk, short_blk = _session_masks(df)
    else:
        session_ok = pd.Series(True, index=df.index)
        long_blk   = pd.Series(False, index=df.index)
        short_blk  = pd.Series(False, index=df.index)

    if P["use_regime"]:
        f  = pd.to_numeric(df[COL["h1_sma20"]],  errors="coerce")
        s  = pd.to_numeric(df[COL["h1_sma200"]], errors="coerce")
        reg_long  = (f > s)
        reg_short = (f < s)
    else:
        reg_long = reg_short = pd.Series(True, index=df.index)

    if P["use_adx"] and (COL["h1_adx14"] in df.columns):
        adx = pd.to_numeric(df[COL["h1_adx14"]], errors="coerce")
        adx_long_ok  = (adx >= ADX_LONG_MIN)
        adx_short_ok = (adx >= ADX_SHORT_MIN)
    else:
        adx_long_ok = adx_short_ok = pd.Series(True, index=df.index)

    if P["use_range"]:
        rng_long_ok, rng_short_ok = _range_separation(df)
    else:
        rng_long_ok = rng_short_ok = pd.Series(True, index=df.index)

    if P["use_obv_cfm"]:
        obv_long_ok, obv_short_ok = _obv_confirms(df)
    else:
        obv_long_ok = obv_short_ok = pd.Series(True, index=df.index)

    long_ok = session_ok & spread_ok & (~long_blk) & reg_long & adx_long_ok & rng_long_ok & obv_long_ok
    short_ok = session_ok & spread_ok & (~short_blk) & reg_short & adx_short_ok & rng_short_ok & obv_short_ok

    close = pd.to_numeric(df[COL["m15_close"]], errors="coerce")
    ema20 = pd.to_numeric(df[COL["m15_ema20"]], errors="coerce")

    mom_long  = (close > ema20)
    mom_short = (close < ema20)

    take_long  = (mom_long  & long_ok)
    take_short = (mom_short & short_ok)

    side = pd.Series(0, index=df.index, dtype=int)
    side[take_long]  =  1
    side[take_short] = -1

    entries = (side != 0) & (side.shift(1).fillna(0) == 0)
    entry_side = side.loc[entries].astype(int).values

    atr = pd.to_numeric(df[COL["m15_atr14"]], errors="coerce").replace(0, np.nan)
    future_close = close.shift(-FUTURE_BARS_HORIZON)
    fut_move = (future_close - close) / atr
    fut_move = fut_move.clip(lower=-R_CLIP, upper=R_CLIP).fillna(0.0)
    fut_R_at_entry = fut_move.loc[entries].values * entry_side

    ts = _get_ts_utc(df, COL["ts"], bar_minutes=15)
    trades = pd.DataFrame({
        "timestamp": ts.loc[entries].values,
        "side":      entry_side,
        "R":         fut_R_at_entry,
    })

    print("\n=== Backtest diagnostics ===")
    print(f"pred_rows: {len(df):>7d}")
    print(f"take_long: {int(take_long.sum()):>7d}")
    print(f"take_short:{int(take_short.sum()):>7d}")

    total_trades = len(trades)
    if total_trades > 0:
        pos = trades.loc[trades["R"] > 0, "R"].sum()
        neg = -trades.loc[trades["R"] < 0, "R"].sum()
        pf  = (pos / neg) if neg > 0 else np.inf
        win_rate = (trades["R"] > 0).mean()
        avg_R = trades["R"].mean()
        max_dd_R = float((-trades["R"].cumsum()).max())

        # Expected Value, in R
        avg_win_R  = trades.loc[trades["R"] > 0, "R"].mean()
        avg_loss_R = trades.loc[trades["R"] < 0, "R"].mean()
        EV_R = (win_rate * avg_win_R) + ((1.0 - win_rate) * avg_loss_R)
        print(f"Expected Value: {EV_R:>6.3f} R")
    else:
        pf = 0.0
        win_rate = 0.0
        avg_R = 0.0
        max_dd_R = 0.0
        EV_R = 0.0

    print("\nBacktest completed, rules only")
    print(f"Total trades : {total_trades:>7d}")
    print(f"Win rate     : {win_rate*100:>6.2f}%")
    print(f"Profit factor: {pf:>5.2f}")
    print(f"Max drawdown : {max_dd_R:>5.2f} R")
    print(f"Average R    : {avg_R:>5.2f}")
    print(f"Expected Val.: {EV_R:>5.3f} R")

    if total_trades > 0:
        trades.to_csv(OUT_TRADES, index=False)

    summary = {
        "total_trades": total_trades,
        "win_rate": float(win_rate),
        "profit_factor": float(pf),
        "average_R": float(avg_R),
        "max_drawdown_R": float(max_dd_R),
        "expected_value_R": float(EV_R),  # include in JSON
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    return trades, summary

if __name__ == "__main__":
    print(">>> phase5_backtest starting...")
    df = pd.read_parquet(FEAT_PATH)
    print(f"Loaded features OK: shape={df.shape}")
    trades, summary = run_backtest(df)

    print("\nBacktest completed, rules only")
    print(f"Total trades : {summary['total_trades']:>7d}")
    print(f"Win rate     : {summary['win_rate']*100:>6.2f}%")
    print(f"Profit factor: {summary['profit_factor']:>5.2f}")
    print(f"Max drawdown : {summary['max_drawdown_R']:>5.2f} R")
    print(f"Average R    : {summary['average_R']:>5.2f}")
    print(f"Expected Val.: {summary['expected_value_R']:>5.3f} R")
    print(">>> phase5_backtest done.")
