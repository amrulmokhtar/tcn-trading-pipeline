# scripts/phase5_backtest.py
# Rule-based backtest that mirrors SMC_2_6_1_VolumeFlow_OBV_Adaptive (no TCN)

import math
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

# ------------------------- Paths -------------------------
DATA_DIR     = Path("data")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
FEAT_PATH    = DATA_DIR / "m15_features.parquet"

# Optional columns we will use if present
COL = dict(
    ts="timestamp",             # optional; if missing we synthesize 15m cadence
    close="m15_close",
    ema="m15_ema20",
    atr="m15_atr14",
    spread_to_atr="m15_spread_to_atr",
    spread_pts="spread_points",  # optional
    comm_usd="commission_roundtrip_usd",  # optional
    h1_sma_fast="h1_sma20",
    h1_sma_slow="h1_sma200",
    h1_adx="h1_adx14",
    h1_atr="h1_atr14",
    h1_atr_pct="h1_atr_pct",
    h1_vol_pct="h1_vol_pct",
    h1_obv="h1_obv",
    h1_obv_sma="h1_obv_sma50",
    h1_obv_slope="h1_obv_slope_6",
    # sessions (booleans already built in Phase-2)
    asia="asia_session",
    london="london_session",
    ny="ny_session",
)

# ------------------------- SMC defaults (from your bot) -------------------------
# Guards (spread; soft block; spread/ATR)
MAX_SPREAD_POINTS     = 9999
USE_SOFT_SPREAD_BLOCK = True
SOFT_SPREAD_FRAC      = 0.60
SPREAD_TO_ATR_CAP     = 9999

# Risk & stake
USE_RISK_PCT       = True
RISK_PCT           = 1.0
FIXED_LOTS         = 0.02

# Regime & Range skip
USE_REGIME_FILTER  = True
H1_FAST_SMA        = 20
H1_SLOW_SMA        = 200
USE_RANGE_SKIP     = True
RANGE_SEP_ATR_LONG = 0.35
RANGE_SEP_ATR_SHORT= 0.45

# ADX
USE_ADX_FILTER     = False
ADX_PERIOD         = 14  # already precomputed
ADX_LONG_MIN       = 22
ADX_SHORT_MIN      = 30

# Volume & OBV
USE_VOL_PCT        = False
VOL_PCT_MIN        = 35
VOL_PCT_MAX        = 90
USE_OBV_CONFIRM    = False
OBV_SMA            = 50  # already precomputed as h1_obv_sma50
OBV_SLOPE_BARS     = 6   # already precomputed as h1_obv_slope_6 (sign)

# Session (Malaysia)
USE_SESSION        = False
TZ_OFFSET_H        = +8
SESSION_START_H    = 9   # 09:00 MYT
SESSION_END_H      = 3   # 03:00 next day (wrap)
SHORT_BLACKOUT_FROM_START_H = 2
SHORT_BLACKOUT_BEFORE_END_H = 1
LONG_BLOCK_START_H = 21
LONG_BLOCK_END_H   = 24  # 21:00-24:00 longs blocked

# Entry controls
MAX_OPEN_POS        = 1      # single position model
MIN_BARS_BETWEEN_LONG  = 18
MIN_BARS_BETWEEN_SHORT = 20
MIN_ENTRY_DIST_ATR_LONG  = 0.7
MIN_ENTRY_DIST_ATR_SHORT = 0.8
MIN_PULLBACK_ATR_LONG    = 0.5
MIN_PULLBACK_ATR_SHORT   = 0.7

# Management (R in ATR at entry)
BE_R_LONG  = 1.25
BE_R_SHORT = 1.35
BE_ATR_BUFFER_FRAC = 0.12
PARTIAL1_R       = 1.5
PARTIAL1_PCT     = 25
TRAIL_START_R_L  = 2.2
TRAIL_START_R_S  = 2.4
ATR_MULT_L       = 1.6
ATR_MULT_S       = 1.8
MAX_HOLD_BARS    = 240

# Basic execution/valuation assumptions for backtest
PIP_SIZE   = 0.1       # XAUUSD pts->"pips" notion (tweak if you price in different units)
PIP_USD    = 1.0       # USD value per pip per 1 lot equivalent (backtest unit stake)
SL_ATR_R   = 1.0       # stop = 1.0 * ATR (at entry)
EQUITY_USD = 10_000.0  # starting equity for sizing
SLIPPAGE_PIPS = 0.0    # you can add slippage later

# ------------------------------------------------------------------------------

def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEAT_PATH)
    # optional synthetic timestamp if missing
    if COL["ts"] not in df.columns:
        # synthetically create a 15-minute cadence
        start = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
        df.insert(0, COL["ts"], pd.date_range(start, periods=len(df), freq="15min", tz="UTC"))
        import warnings
        warnings.warn(f"'{COL['ts']}' missing; generating synthetic timestamps assuming 15-min bars.")
    return df

# ---------- Guards & filters (mirror SMC) ----------

def in_session_myt(ts_utc: pd.Timestamp) -> Tuple[bool,bool,bool,int]:
    """Return session_ok, long_ok, short_ok, hour_myt."""
    if not USE_SESSION:
        return True, True, True, (ts_utc.tz_convert("UTC").hour + TZ_OFFSET_H) % 24

    myt = ts_utc.tz_convert("UTC").tz_convert("Etc/GMT-0") + pd.Timedelta(hours=TZ_OFFSET_H)
    h = int(myt.hour)

    # allowed window 09:00 → 03:00 (wrap)
    def in_window(hh, s, e):
        return (hh >= s and hh < 24) or (hh < e) if s > e else (s <= hh < e)

    session_ok = in_window(h, SESSION_START_H, SESSION_END_H)

    # long side block 21 → 24
    long_ok = session_ok and not in_window(h, LONG_BLOCK_START_H, LONG_BLOCK_END_H)
    # short blackout: first N hours after start or last N hours before end
    # crude but effective:
    hours_since_start = (h - SESSION_START_H) % 24
    # hours to end if wrap:
    hours_to_end = (SESSION_END_H - h) % 24
    short_blackout = hours_since_start < SHORT_BLACKOUT_FROM_START_H or \
                     (0 <= hours_to_end <= SHORT_BLACKOUT_BEFORE_END_H)
    short_ok = session_ok and not short_blackout

    return session_ok, long_ok, short_ok, h

def spread_ok(row) -> bool:
    pts = float(row.get(COL["spread_pts"], np.nan))
    if np.isnan(pts):
        # if no absolute points column, allow and rely on spread/ATR later
        return True
    if pts > MAX_SPREAD_POINTS:
        return False
    if USE_SOFT_SPREAD_BLOCK:
        if pts > SOFT_SPREAD_FRAC * MAX_SPREAD_POINTS:
            return False
    return True

def spread_vs_atr_ok(row) -> bool:
    atr = float(row[COL["h1_atr"]])
    if atr <= 0: 
        return True
    # if explicit points present, use that; else derive from spread_to_atr column if available
    pts = row.get(COL["spread_pts"], np.nan)
    if not np.isnan(pts):
        spread_pips = pts  # already “points/pips” unit
        ratio = (spread_pips) / (atr / PIP_SIZE)
        return ratio <= SPREAD_TO_ATR_CAP
    # else use precomputed ratio if present
    if COL["spread_to_atr"] in row.index:
        ratio = float(row[COL["spread_to_atr"]])
        return ratio <= SPREAD_TO_ATR_CAP
    return True

def trend_with_range(row) -> int:
    """+1 long, -1 short, 0 none; includes range skip by ATR distance of SMAs."""
    if not USE_REGIME_FILTER:
        return 0
    fast = float(row[COL["h1_sma_fast"]])
    slow = float(row[COL["h1_sma_slow"]])
    if np.isnan(fast) or np.isnan(slow):
        return 0
    dir_ = 1 if fast > slow else (-1 if fast < slow else 0)
    if not USE_RANGE_SKIP or dir_ == 0:
        return dir_
    atr = float(row[COL["h1_atr"]])
    if atr <= 0:
        return dir_
    sep = abs(fast - slow)
    ratio = sep / atr
    if dir_ > 0 and ratio < RANGE_SEP_ATR_LONG:
        return 0
    if dir_ < 0 and ratio < RANGE_SEP_ATR_SHORT:
        return 0
    return dir_

def adx_ok(row, dir_) -> bool:
    if not USE_ADX_FILTER or dir_ == 0:
        return True
    adx = float(row[COL["h1_adx"]])
    if np.isnan(adx) or adx < 0:
        return False
    if dir_ > 0:
        return adx >= ADX_LONG_MIN
    else:
        return adx >= ADX_SHORT_MIN

def volume_ok(row) -> bool:
    if not USE_VOL_PCT:
        return True
    pct = float(row[COL["h1_vol_pct"]])
    if np.isnan(pct):
        return False
    return (pct >= VOL_PCT_MIN) and (pct <= VOL_PCT_MAX)

def obv_ok(row, dir_) -> bool:
    if not USE_OBV_CONFIRM or dir_ == 0:
        return True
    obv = float(row[COL["h1_obv"]])
    obv_sma = float(row[COL["h1_obv_sma"]])
    slope = float(row[COL["h1_obv_slope"]])  # positive for up
    if any(np.isnan(x) for x in (obv, obv_sma, slope)):
        return False
    if dir_ > 0:
        return (obv >= obv_sma) and (slope > 0)
    else:
        return (obv <= obv_sma) and (slope < 0)

# ---------- Position model ----------
class Pos:
    def __init__(self, side: int, entry_idx: int, entry_px: float, atr: float, eq: float):
        self.side     = side              # +1 long, -1 short
        self.entry_i  = entry_idx
        self.entry_px = entry_px
        self.atr      = atr
        self.sl_pips  = (SL_ATR_R * atr) / PIP_SIZE
        # risk sizing
        if USE_RISK_PCT:
            risk_usd  = eq * (RISK_PCT/100.0)
            self.lots = max(0.01, risk_usd / max(1.0, self.sl_pips * PIP_USD))
        else:
            self.lots = FIXED_LOTS
        self.stop_px  = entry_px - self.side * self.sl_pips * PIP_SIZE
        self.take_px  = None
        self.be_done  = False
        self.part1    = False
        self.alive    = True
        self.units    = self.lots  # one “lot” unit notion

    def r_now(self, px: float) -> float:
        dist_pips = (px - self.entry_px) / PIP_SIZE * self.side
        return dist_pips / max(1e-6, self.sl_pips)

def run_backtest(df: pd.DataFrame):
    # ensure/derive columns
    need = [COL["close"], COL["ema"], COL["h1_sma_fast"], COL["h1_sma_slow"],
            COL["h1_adx"], COL["h1_atr"], COL["h1_vol_pct"], COL["h1_obv"], COL["h1_obv_sma"],
            COL["h1_obv_slope"]]
    for col in need:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # convenient numpy views
    close = df[COL["close"]].to_numpy(np.float64)
    ema   = df[COL["ema"]].to_numpy(np.float64)
    h1atr = df[COL["h1_atr"]].to_numpy(np.float64)

    ts = pd.to_datetime(df[COL["ts"]])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")


    equity = EQUITY_USD
    open_pos: Optional[Pos] = None
    last_long_i  = -10_000
    last_short_i = -10_000

    trades = []  # list of dicts

    for i in range(1, len(df)):
        row = df.iloc[i]
        px  = close[i]
        atr = h1atr[i]
        if np.isnan(px) or atr <= 0:
            # safety
            if open_pos and (i - open_pos.entry_i) >= MAX_HOLD_BARS:
                open_pos.alive = False
            continue

        # manage exit/trailing for open position
        if open_pos and open_pos.alive:
            rnow = open_pos.r_now(px)
            # BE
            be_r = BE_R_LONG if open_pos.side > 0 else BE_R_SHORT
            if (not open_pos.be_done) and rnow >= be_r:
                spread_pips = float(row.get(COL["spread_pts"], 0.0)) if COL["spread_pts"] in row.index else 0.0
                atr_pips    = atr / PIP_SIZE
                buffer_pips = spread_pips + max(1.0, BE_ATR_BUFFER_FRAC * atr_pips)
                be_px = open_pos.entry_px + open_pos.side * buffer_pips * PIP_SIZE
                # never reduce beyond entry if negative buffer math
                be_px = open_pos.entry_px + open_pos.side * max(0.0, (be_px - open_pos.entry_px) * open_pos.side)
                open_pos.stop_px = max(open_pos.stop_px, be_px) if open_pos.side>0 else min(open_pos.stop_px, be_px)
                open_pos.be_done = True

            # partial at 1.5R
            if (not open_pos.part1) and rnow >= PARTIAL1_R:
                # book partial; for PnL we’ll close that portion at current px
                trades.append(dict(
                    kind="partial", side=open_pos.side, entry_px=open_pos.entry_px, exit_px=px,
                    entry_i=open_pos.entry_i, exit_i=i, lots=open_pos.units*(PARTIAL1_PCT/100.0)))
                open_pos.units *= (1 - PARTIAL1_PCT/100.0)
                open_pos.part1 = True

            # ATR trail after start R
            startR = TRAIL_START_R_L if open_pos.side>0 else TRAIL_START_R_S
            if rnow >= startR:
                trail = atr * (ATR_MULT_L if open_pos.side>0 else ATR_MULT_S)
                new_sl = px - open_pos.side * trail
                open_pos.stop_px = max(open_pos.stop_px, new_sl) if open_pos.side>0 else min(open_pos.stop_px, new_sl)

            # max hold
            if (i - open_pos.entry_i) >= MAX_HOLD_BARS:
                # time-based exit
                trades.append(dict(kind="exit_time", side=open_pos.side, entry_px=open_pos.entry_px, exit_px=px,
                                   entry_i=open_pos.entry_i, exit_i=i, lots=open_pos.units))
                open_pos.alive = False
                open_pos = None
            else:
                # stop hit?
                if (open_pos.side>0 and px <= open_pos.stop_px) or (open_pos.side<0 and px >= open_pos.stop_px):
                    trades.append(dict(kind="stop", side=open_pos.side, entry_px=open_pos.entry_px, exit_px=open_pos.stop_px,
                                       entry_i=open_pos.entry_i, exit_i=i, lots=open_pos.units))
                    open_pos.alive = False
                    open_pos = None

        # entry logic only if flat or effectively flat (we enforce single position)
        if open_pos is None:
            session_ok, long_ok, short_ok, _ = in_session_myt(ts.iloc[i])
            if not session_ok: 
                continue
            if not spread_ok(row): 
                continue
            if not spread_vs_atr_ok(row): 
                continue

            dir_ = trend_with_range(row)
            if dir_ == 0: 
                continue
            if not adx_ok(row, dir_): 
                continue
            if not volume_ok(row):
                continue
            if not obv_ok(row, dir_):
                continue

            # M15 alignment & pullback
            align_long  = close[i] > ema[i]
            align_short = close[i] < ema[i]
            pullback_atr = abs(close[i] - ema[i]) / max(1e-8, atr)

            # min distance since last same-side entry (in ATR)
            def dist_since_last(side, last_i):
                if last_i < 0: 
                    return np.inf
                return abs(close[i] - close[last_i]) / max(1e-8, atr)

            if dir_ > 0 and long_ok and align_long:
                if (i - last_long_i) >= MIN_BARS_BETWEEN_LONG \
                   and pullback_atr >= MIN_PULLBACK_ATR_LONG \
                   and dist_since_last(+1, last_long_i) >= MIN_ENTRY_DIST_ATR_LONG:
                    # enter long
                    p = Pos(+1, i, close[i] + SLIPPAGE_PIPS*PIP_SIZE, atr, equity)
                    open_pos = p
                    last_long_i = i
            elif dir_ < 0 and short_ok and align_short:
                if (i - last_short_i) >= MIN_BARS_BETWEEN_SHORT \
                   and pullback_atr >= MIN_PULLBACK_ATR_SHORT \
                   and dist_since_last(-1, last_short_i) >= MIN_ENTRY_DIST_ATR_SHORT:
                    p = Pos(-1, i, close[i] - SLIPPAGE_PIPS*PIP_SIZE, atr, equity)
                    open_pos = p
                    last_short_i = i

    # Flush any open position at last price
    if open_pos and open_pos.alive:
        trades.append(dict(kind="final", side=open_pos.side, entry_px=open_pos.entry_px, exit_px=close[-1],
                           entry_i=open_pos.entry_i, exit_i=len(df)-1, lots=open_pos.units))
        open_pos = None

    # ---------- PnL & metrics ----------
    if not trades:
        print("No trades.")
        return

    out = []
    wins = 0
    loss = 0
    gross_p = 0.0
    gross_l = 0.0

    comm = df.get(COL["comm_usd"])
    comm_per_round = float(comm.iloc[0]) if comm is not None else 0.0

    for t in trades:
        side = t["side"]
        pips = (t["exit_px"] - t["entry_px"]) / PIP_SIZE * side
        pnl  = pips * PIP_USD * t["lots"]
        if t["kind"] in ("stop", "final", "exit_time", "partial"):
            # charge commission on full round (very rough; adjust to your broker math)
            pnl -= comm_per_round
        out.append(dict(ts_entry=str(ts.iloc[t["entry_i"]]), ts_exit=str(ts.iloc[t["exit_i"]]),
                        side="long" if side>0 else "short", kind=t["kind"], pips=pips, pnl=pnl))
        if pnl > 0: 
            wins += 1; gross_p += pnl
        else:
            loss += 1; gross_l += -pnl

    trades_df = pd.DataFrame(out)
    trades_df.to_csv(RESULTS_DIR/"trades_rules.csv", index=False)

    winrate = 100.0 * wins / max(1, wins+loss)
    pf = (gross_p / max(1e-9, gross_l)) if gross_l>0 else np.inf
    dd = compute_max_dd(trades_df["pnl"].to_numpy(np.float64))

    print("Backtest completed (rules only)")
    print(f"Total trades : {len(trades_df):,}")
    print(f"Win rate     : {winrate:.2f}%")
    print(f"Profit factor: {pf:.2f}")
    print(f"Max drawdown : {dd:,.2f} USD")
    print(f"Average R    : {trades_df['pips'].mean() / (SL_ATR_R*(df[COL['h1_atr']]/PIP_SIZE).median()):.2f}")

    summary = dict(total_trades=int(len(trades_df)), winrate_pct=winrate, profit_factor=float(pf),
                   max_drawdown_usd=float(dd))
    with open(RESULTS_DIR/"backtest_rules_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

def compute_max_dd(pnls: np.ndarray) -> float:
    """Max drawdown on cumulative PnL."""
    curve = pnls.cumsum()
    peak  = np.maximum.accumulate(curve)
    dd    = (curve - peak).min() if len(curve) else 0.0
    return -float(dd)

# ------------------------- main -------------------------
if __name__ == "__main__":
    df = load_features()
    print("Loaded features:", FEAT_PATH)
    run_backtest(df)
