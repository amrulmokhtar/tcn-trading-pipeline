# scripts/phase5_backtest.py
# Rules-only backtest (no TCN) with spread-to-ATR diagnostics
# Works on: data/m15_features.parquet
# Output: prints summary; optional trades.csv + backtest_summary.json

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

# =============== File paths ===============
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

FEAT_PATH   = DATA_DIR / "m15_features.parquet"
OUT_TRADES  = RESULTS_DIR / "trades.csv"
OUT_SUMMARY = RESULTS_DIR / "backtest_summary.json"

# =============== Column names ===============
COL = {
    "ts":        "timestamp",
    "close":     "m15_close",
    "atr":       "m15_atr14",
    # we only have normalized spread (spread/ATR)
    "spread_atr": "m15_spread_to_atr",
    "spread_pts": "m15_spread_to_atr",  # reuse same column
    "commission": "commission_roundtrip_usd",  # optional
    "sma_fast":  "h1_sma20",
    "sma_slow":  "h1_sma200",
    "adx":       "h1_adx14",
    "obv_slope": "h1_obv_slope_6",
    "asia":      "asia_session",
    "london":    "london_session",
    "ny":        "ny_session",
}

# =============== Filter toggles ===============
USE_SESSION      = True
USE_REGIME_FILTER= True
USE_RANGE_SKIP   = True
USE_ADX_FILTER   = True
USE_OBV_PCT      = False   # keep False unless you have a pct column
USE_OBV_CONFIRM  = False  # True if you want an OBV slope confirm

# =============== Spread constraints ===============
MAX_SPREAD_POINTS = 30     # absolute points cap (tighten later)
SPREAD_TO_ATR_CAP = 10     # <-- sweep 6..12 to find elbow; 10 is a good start

# =============== Session (Malaysia time) ===============
TZ_OFFSET_H     = +8
SESSION_START_H = 9        # 09:00 MYT
SESSION_END_H   = 3        # 03:00 next day (wrap)
SHORT_BLACKOUT_FROM_START_H = 2  # 09:00 -> 11:00: block shorts
SHORT_BLACKOUT_BEFORE_END_H = 1  # last hour before 03:00: block shorts
LONG_BLOCK_START_H = 21          # 21:00 -> 24:00: block longs
LONG_BLOCK_END_H   = 24

# =============== Regime / Range skip ===============
RANGE_SEP_ATR_LONG  = 0.35
RANGE_SEP_ATR_SHORT = 0.45

# =============== ADX thresholds ===============
ADX_LONG_MIN  = 22
ADX_SHORT_MIN = 30

# =============== Trade management (R is ATR at entry) ===============
RISK_R           = 1.0     # SL distance in ATR multiples
TP_R             = 2.0     # single TP at +2R
MOVE_BE_AT_R     = 1.0     # move to BE at +1R
TRAIL_START_R    = 3.0     # start trailing after 3R (disabled if <= TP_R)
TRAIL_ATR_MULT   = 2.0     # trail = entry +/- ATR * mult

# =============== Utilities ===============

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    # basic presence
    need_cols = [COL["close"], COL["atr"], COL["spread_atr"], COL["spread_pts"],
                 COL["sma_fast"], COL["sma_slow"], COL["adx"],
                 COL["asia"], COL["london"], COL["ny"]]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'.")

    # commission optional
    if COL["commission"] not in df.columns:
        df[COL["commission"]] = 0.0

    # timestamp optional -> synthetic
    if COL["ts"] not in df.columns:
        # Create a synthetic 15-min timeline
        n = len(df)
        ts0 = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")  # arbitrary start
        df[COL["ts"]] = pd.date_range(ts0, periods=n, freq="15min")
        import warnings
        warnings.warn(f"'{COL['ts']}' missing; generating synthetic timestamps assuming 15-min bars.")

    return df

def in_my_session(ts_utc: pd.Timestamp) -> Tuple[bool, bool, bool, int]:
    """
    Return: (session_ok, long_ok, short_ok, hour_myt)
    Session window: 09:00 MYT -> 03:00 MYT (wrap).
    Blackouts: shorts blocked at start window (first X hours) and final Y hours,
               longs blocked nightly 21:00 -> 24:00 MYT.
    """
    myt = ts_utc.tz_convert("UTC").tz_convert("Etc/GMT-0") + pd.Timedelta(hours=TZ_OFFSET_H)
    h = int(myt.hour)

    # allowed 09 -> 03 next day
    def _in_window(hh: int) -> bool:
        if SESSION_START_H < SESSION_END_H:  # non-wrap (not our case)
            return SESSION_START_H <= hh < SESSION_END_H
        # wrap case
        return (hh >= SESSION_START_H) or (hh < SESSION_END_H)

    ok = _in_window(h)
    long_ok  = ok
    short_ok = ok

    # longs blocked at night
    if LONG_BLOCK_START_H <= h < LONG_BLOCK_END_H:
        long_ok = False

    # shorts blackout at start
    if ok:
        # hours since session start (wrap)
        if h >= SESSION_START_H:
            since_start = h - SESSION_START_H
        else:
            since_start = (24 - SESSION_START_H) + h
        # hours before end (wrap)
        if h >= SESSION_END_H:
            before_end = (24 - h) + SESSION_END_H
        else:
            before_end = SESSION_END_H - h

        if since_start < SHORT_BLACKOUT_FROM_START_H:
            short_ok = False
        if before_end <= SHORT_BLACKOUT_BEFORE_END_H:
            short_ok = False

    return ok, long_ok, short_ok, h

def _spread_filters(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return (mask_pts, mask_ratio) for diagnostics and filtering."""
    m_pts   = (df[COL["spread_pts"]].astype(float) <= MAX_SPREAD_POINTS)
    # guard against zeros/NaNs
    atr     = df[COL["atr"]].astype(float).replace(0.0, np.nan)
    ratio   = (df[COL["spread_pts"]].astype(float) / atr).replace([np.inf, -np.inf], np.nan).fillna(np.inf)
    m_ratio = (ratio <= SPREAD_TO_ATR_CAP)
    return m_pts, m_ratio

def _regime_masks(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    fast = df[COL["sma_fast"]].astype(float)
    slow = df[COL["sma_slow"]].astype(float)
    long_ok  = fast > slow
    short_ok = fast < slow
    return long_ok, short_ok

def _range_skip_masks(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    # Use your existing m15_spread_to_atr as a crude separateness proxy.
    sep = df[COL["spread_atr"]].astype(float)
    long_ok  = sep >= RANGE_SEP_ATR_LONG
    short_ok = sep >= RANGE_SEP_ATR_SHORT
    return long_ok, short_ok

def _adx_masks(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    adx = df[COL["adx"]].astype(float)
    return (adx >= ADX_LONG_MIN), (adx >= ADX_SHORT_MIN)

def _obv_confirm(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    # Simple confirm based on OBV slope sign; disable if not used/available
    if (COL["obv_slope"] not in df.columns) or (not USE_OBV_CONFIRM):
        n = len(df)
        return pd.Series([True]*n, index=df.index), pd.Series([True]*n, index=df.index)
    slope = df[COL["obv_slope"]].astype(float)
    return (slope > 0), (slope < 0)

# =============== Trade simulation ===============
def simulate_trades(df: pd.DataFrame, side: pd.Series) -> pd.DataFrame:
    """
    Simple 1-position engine:
      - Enter on side change (0->1 for long, 0->-1 for short) at CLOSE
      - SL = ATR * RISK_R; TP = ATR * TP_R
      - Move to BE at +1R
      - Optional trailing after TRAIL_START_R at ATR*TRAIL_ATR_MULT
      - Exit on first SL/TP hit
    Returns trades dataframe.
    """
    closes = df[COL["close"]].to_numpy(dtype=float)
    atr    = df[COL["atr"]].to_numpy(dtype=float)
    ts     = df[COL["ts"]].to_numpy(dtype="datetime64[ns]")

    s      = side.to_numpy(dtype=int)
    comm   = df.get(COL["commission"], pd.Series(0.0, index=df.index)).to_numpy(dtype=float)

    trades = []
    pos = 0  # 0 none, +1 long, -1 short
    entry_idx = None
    entry = sl = tp = be_level = None
    trail_on = False

    for i in range(len(df)):
        # manage open position
        if pos != 0:
            risk = atr[entry_idx] * RISK_R
            if pos > 0:
                # trailing?
                if TRAIL_START_R > TP_R:
                    # trailing only if > TP_R to avoid conflicting with TP
                    # price move in R:
                    move_r = (closes[i] - entry) / max(risk, 1e-12)
                    if (not trail_on) and (move_r >= TRAIL_START_R):
                        trail_on = True
                    if trail_on:
                        sl = max(sl, closes[i] - atr[i]*TRAIL_ATR_MULT)

                # BE move
                if be_level is None and (closes[i] - entry) >= risk * MOVE_BE_AT_R:
                    be_level = entry
                    sl = max(sl, be_level)

                # check exits
                hit_tp = closes[i] >= tp
                hit_sl = closes[i] <= sl
                if hit_tp or hit_sl:
                    exit_px = tp if hit_tp else sl
                    r = (exit_px - entry) / max(risk, 1e-12)
                    trades.append({
                        "entry_time": ts[entry_idx], "exit_time": ts[i],
                        "side": "long", "entry": entry, "exit": exit_px,
                        "R": r, "pnl": r, "commission": comm[i]
                    })
                    pos = 0; entry_idx = None; trail_on = False; be_level = None
                    continue

            else:  # short
                if TRAIL_START_R > TP_R:
                    move_r = (entry - closes[i]) / max(risk, 1e-12)
                    if (not trail_on) and (move_r >= TRAIL_START_R):
                        trail_on = True
                    if trail_on:
                        sl = min(sl, closes[i] + atr[i]*TRAIL_ATR_MULT)

                if be_level is None and (entry - closes[i]) >= risk * MOVE_BE_AT_R:
                    be_level = entry
                    sl = min(sl, be_level)

                hit_tp = closes[i] <= tp
                hit_sl = closes[i] >= sl
                if hit_tp or hit_sl:
                    exit_px = tp if hit_tp else sl
                    r = (entry - exit_px) / max(risk, 1e-12)
                    trades.append({
                        "entry_time": ts[entry_idx], "exit_time": ts[i],
                        "side": "short", "entry": entry, "exit": exit_px,
                        "R": r, "pnl": r, "commission": comm[i]
                    })
                    pos = 0; entry_idx = None; trail_on = False; be_level = None
                    continue

        # open new position
        if pos == 0 and s[i] != 0:
            pos = s[i]
            entry_idx = i
            entry = closes[i]
            risk = atr[i] * RISK_R
            if pos > 0:
                sl = entry - risk
                tp = entry + risk * TP_R
            else:
                sl = entry + risk
                tp = entry - risk * TP_R
            be_level = None
            trail_on = False

    return pd.DataFrame(trades)

# =============== Main backtest ===============
def run_backtest(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    df = _ensure_cols(df)

    # Spread filters (and diagnostics)
    m_pts, m_ratio = _spread_filters(df)
    ratio = (df[COL["spread_pts"]].astype(float) /
             df[COL["atr"]].astype(float).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

    print("\n=== Spread/ATR diagnostics (all bars) ===")
    print(ratio.dropna().describe(percentiles=[.1,.25,.5,.75,.9,.95]).to_string())
    removed_ratio = (ratio > SPREAD_TO_ATR_CAP).sum()
    kept_ratio    = (ratio <= SPREAD_TO_ATR_CAP).sum()
    print(f"Cap = {SPREAD_TO_ATR_CAP} → kept {kept_ratio:,}, removed {removed_ratio:,} "
          f"({removed_ratio/(kept_ratio+removed_ratio+1e-9):.1%} removed)")
    removed_pts = (~m_pts).sum()
    kept_pts    = (m_pts).sum()
    print(f"MAX_SPREAD_POINTS = {MAX_SPREAD_POINTS} → kept {kept_pts:,}, removed {removed_pts:,} "
          f"({removed_pts/(kept_pts+removed_pts+1e-9):.1%} removed)")

    # Base mask (always apply spread constraints)
    base = m_pts & m_ratio

    # Session mask
    if USE_SESSION:
        # If you have ready-made session booleans, use them:
        session_ok = (df[COL["asia"]] | df[COL["london"]] | df[COL["ny"]]).astype(bool)
        # refine with blackout logic by hour in MYT
        session_flags = []
        for ts in df[COL["ts"]]:
            ok, long_ok_s, short_ok_s, _ = in_my_session(ts)
            session_flags.append((ok, long_ok_s, short_ok_s))
        session_flags = np.array(session_flags, dtype=object)
        session_ok = session_ok & pd.Series([x[0] for x in session_flags], index=df.index)
        long_session_ok  = pd.Series([x[1] for x in session_flags], index=df.index)
        short_session_ok = pd.Series([x[2] for x in session_flags], index=df.index)
    else:
        n = len(df)
        session_ok      = pd.Series([True]*n, index=df.index)
        long_session_ok = pd.Series([True]*n, index=df.index)
        short_session_ok= pd.Series([True]*n, index=df.index)

    # Regime masks
    if USE_REGIME_FILTER:
        reg_long, reg_short = _regime_masks(df)
    else:
        n = len(df)
        reg_long = pd.Series([True]*n, index=df.index)
        reg_short= pd.Series([True]*n, index=df.index)

    # Range-separation masks
    if USE_RANGE_SKIP:
        rng_long, rng_short = _range_skip_masks(df)
    else:
        n = len(df)
        rng_long = pd.Series([True]*n, index=df.index)
        rng_short= pd.Series([True]*n, index=df.index)

    # ADX masks
    if USE_ADX_FILTER:
        adx_long, adx_short = _adx_masks(df)
    else:
        n = len(df)
        adx_long  = pd.Series([True]*n, index=df.index)
        adx_short = pd.Series([True]*n, index=df.index)

    # OBV confirm
    obv_long, obv_short = _obv_confirm(df)

    # Final long/short eligibility
    long_ok  = base & session_ok & long_session_ok & reg_long & rng_long & adx_long & obv_long
    short_ok = base & session_ok & short_session_ok & reg_short & rng_short & adx_short & obv_short

    # Signal: enter when eligible (no smoothing for now)
    side = pd.Series(0, index=df.index, dtype=int)
    side[long_ok]  =  1
    side[short_ok] = -1

    # Optional: ensure no simultaneous long & short (prefer none on ties)
    both = long_ok & short_ok
    side[both] = 0

    # --- Debug counts
    print("\n=== Backtest diagnostics ===")
    print(f"pred_rows: {len(df):7d}")
    print(f"session_ok: {int(session_ok.sum()):7d}")
    print(f"reg_long:   {int(reg_long.sum()):7d}   reg_short: {int(reg_short.sum()):7d}")
    print(f"rng_long:   {int(rng_long.sum()):7d}   rng_short: {int(rng_short.sum()):7d}")
    print(f"adx_ok L/S: {int(adx_long.sum()):7d} / {int(adx_short.sum()):7d}")
    print(f"spread_ok pts/ratio: {int(m_pts.sum()):7d} / {int(m_ratio.sum()):7d}")
    print(f"take_long:  {int((side== 1).sum()):7d}")
    print(f"take_short: {int((side==-1).sum()):7d}")

    # Simulate trades
    trades = simulate_trades(df, side)

    # Apply commissions in R terms (optional; here we subtract USD-equivalent R ≈ commission/(ATR*value) – unknown tick value)
    # Simpler: subtract commission as a flat R fraction of 1R risk at entry ATR; if you want exact $, keep R as-is and report PF separately.
    # For clarity we keep R as-is; PF uses R as pnl unit.

    # Summary
    wins  = (trades["R"] > 0).sum()
    loss  = (trades["R"] <= 0).sum()
    wr    = wins / max(len(trades), 1)
    pf    = trades.loc[trades["R"]>0, "R"].sum() / abs(trades.loc[trades["R"]<=0, "R"].sum() or 1e-12)
    avg_r = trades["R"].mean() if len(trades) else 0.0

    # equity and DD (in R)
    eq = trades["R"].cumsum() if len(trades) else pd.Series([], dtype=float)
    peak = eq.cummax() if len(eq) else eq
    dd = (eq - peak).min() if len(eq) else 0.0

    summary = {
        "total_trades": int(len(trades)),
        "win_rate": float(wr*100.0),
        "profit_factor": float(pf),
        "max_drawdown_R": float(abs(dd)),
        "average_R": float(avg_r),
        "params": {
            "MAX_SPREAD_POINTS": MAX_SPREAD_POINTS,
            "SPREAD_TO_ATR_CAP": SPREAD_TO_ATR_CAP,
            "USE_SESSION": USE_SESSION,
            "USE_REGIME_FILTER": USE_REGIME_FILTER,
            "USE_RANGE_SKIP": USE_RANGE_SKIP,
            "USE_ADX_FILTER": USE_ADX_FILTER,
            "USE_OBV_CONFIRM": USE_OBV_CONFIRM,
            "RISK_R": RISK_R, "TP_R": TP_R,
            "MOVE_BE_AT_R": MOVE_BE_AT_R,
            "TRAIL_START_R": TRAIL_START_R,
            "TRAIL_ATR_MULT": TRAIL_ATR_MULT,
        }
    }

    # Save outputs
    if len(trades):
        trades.to_csv(OUT_TRADES, index=False)
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    # Print nice summary
    print("\nBacktest completed (rules only)")
    print(f"Total trades : {summary['total_trades']:7d}")
    print(f"Win rate     : {summary['win_rate']:.2f}%")
    print(f"Profit factor: {summary['profit_factor']:.2f}")
    print(f"Max drawdown : {summary['max_drawdown_R']:.2f} R")
    print(f"Average R    : {summary['average_R']:.2f}")

    # --- Add this block below ---
    summary = {
        "total_trades": summary.get("total_trades", 0),
        "win_rate": summary.get("win_rate", 0.0),
        "profit_factor": summary.get("profit_factor", 0.0),
        "average_R": summary.get("average_R", 0.0),
        "max_drawdown_R": summary.get("max_drawdown_R", 0.0),
    }
    return trades, summary
    
if __name__ == "__main__":
    import sys, traceback, pandas as pd

    print("\n>>> phase5_backtest starting...", flush=True)
    try:
        print(f"FEAT_PATH = {FEAT_PATH}", flush=True)
        df = pd.read_parquet(FEAT_PATH)
        print(f"Loaded features OK: shape={df.shape}", flush=True)

        trades, summary = run_backtest(df)

        print("\nBacktest completed (rules only)", flush=True)
        print(f"Total trades : {summary.get('total_trades', 0)}", flush=True)
        wr = summary.get("win_rate", 0.0)
        print(f"Win rate     : {wr*100:.2f}%", flush=True)
        print(f"Profit factor: {summary.get('profit_factor', 0.0):.2f}", flush=True)
        print(f"Max drawdown : {summary.get('max_drawdown_R', 0.0):.2f} R", flush=True)
        print(f"Average R    : {summary.get('average_R', 0.0):.2f}", flush=True)

        print(">>> phase5_backtest done.\n", flush=True)
    except Exception as e:
        print("\n!!! phase5_backtest crashed !!!", flush=True)
        traceback.print_exc()
        sys.exit(1)
