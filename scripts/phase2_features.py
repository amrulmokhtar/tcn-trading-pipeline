# scripts/phase2_features.py
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
OUT_H1 = DATA_DIR / "h1_features.parquet"
OUT_M15 = DATA_DIR / "m15_features.parquet"

M15_CSV = DATA_DIR / "cleaned_15M_data.csv"
H1_CSV  = DATA_DIR / "cleaned_1H_data.csv"

# helper functions, no lookahead, use past only
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def true_range(h, l, c_prev):
    return pd.concat([
        h - l,
        (h - c_prev).abs(),
        (l - c_prev).abs()
    ], axis=1).max(axis=1)

def atr(high, low, close, period=14):
    c_prev = close.shift(1)
    tr = true_range(high, low, c_prev)
    return tr.ewm(alpha=1.0/period, adjust=False).mean()

def adx(high, low, close, period=14):
    # Wilder style, all pandas, keep index
    up = high.diff()
    down = (-low.diff())

    plus_dm  = ((up > down) & (up > 0)).astype(float) * up.fillna(0.0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.fillna(0.0)

    tr = true_range(high, low, close.shift(1))
    atr_w = tr.ewm(alpha=1.0/period, adjust=False).mean()

    plus_di  = 100.0 * plus_dm.ewm(alpha=1.0/period, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0/period, adjust=False).mean() / atr_w.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1.0/period, adjust=False).mean()
    return adx_val


def obv(close, volume):
    # classic OBV, cumulative sign of close change times volume
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.fillna(0.0)).cumsum()

def rolling_percentile(series, window=120):
    # percentile rank of current value within previous window, exclude current lookahead by shifting one
    s = series.copy()
    def pct_rank(win):
        x = win.values[:-1]
        if len(x) == 0:
            return np.nan
        return 100.0 * (x <= win.values[-1]).sum() / len(x)
    return s.shift(0).rolling(window + 1, min_periods=window + 1).apply(pct_rank, raw=False)

def make_session_flags(idx, tz_offset_hours=8, start_hour=9, end_hour=3):
    """
    Build simple session flags using a fixed hour offset.
    No tz database names needed, avoids UnknownTimeZoneError.
    """
    # ensure DatetimeIndex without tz
    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)
    else:
        idx = pd.DatetimeIndex(idx)

    # compute local hour by adding offset and wrapping 0..23
    local_hour = ((idx.hour + int(tz_offset_hours)) % 24)

    def in_window(h, s, e):
        if s == e:
            return np.ones_like(h, dtype=bool)
        if s < e:
            return (h >= s) & (h < e)
        # overnight window
        return (h >= s) | (h < e)

    asia_session = in_window(local_hour, start_hour, end_hour).astype(int)
    # rough London, New York windows in local MYT hours, adjust if you prefer
    london_session = ((local_hour >= 15) & (local_hour < 23)).astype(int)
    ny_session = ((local_hour >= 21) | (local_hour < 5)).astype(int)

    return pd.DataFrame(
        {
            "asia_session": asia_session,
            "london_session": london_session,
            "ny_session": ny_session,
        },
        index=idx,
    )

# 1) load cleaned data
m15 = pd.read_csv(M15_CSV, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
h1  = pd.read_csv(H1_CSV,  parse_dates=["timestamp"]).set_index("timestamp").sort_index()

# 2) build H1 features
h1_feat = pd.DataFrame(index=h1.index)
h1_feat["h1_close"] = h1["close"]
h1_feat["h1_sma20"] = h1["close"].rolling(20, min_periods=20).mean()
h1_feat["h1_sma200"] = h1["close"].rolling(200, min_periods=200).mean()
h1_feat["h1_adx14"] = adx(h1["high"], h1["low"], h1["close"], period=14)
h1_feat["h1_atr14"] = atr(h1["high"], h1["low"], h1["close"], period=14)
# ATR percentile over 200 lookback, exclude lookahead
h1_feat["h1_atr_pct"] = rolling_percentile(h1_feat["h1_atr14"], window=200)
# volume percentile
h1_feat["h1_vol_pct"] = rolling_percentile(h1["tick_volume"], window=120)
# OBV and slope over last N bars
h1_feat["h1_obv"] = obv(h1["close"], h1["tick_volume"])
h1_feat["h1_obv_sma50"] = h1_feat["h1_obv"].rolling(50, min_periods=50).mean()
h1_feat["h1_obv_slope_6"] = h1_feat["h1_obv"].diff(6)
# regime direction, simple
h1_feat["h1_trend_dir"] = np.sign((h1_feat["h1_sma20"] - h1_feat["h1_sma200"]).fillna(0.0))

# 3) build M15 features
m15_feat = pd.DataFrame(index=m15.index)
m15_feat["m15_close"] = m15["close"]
m15_feat["m15_ema20"] = ema(m15["close"], 20)
m15_feat["m15_atr14"] = atr(m15["high"], m15["low"], m15["close"], period=14)
# spread to ATR ratio, guard div by zero
m15_feat["m15_spread_to_atr"] = (m15["spread_points"] / m15_feat["m15_atr14"].replace(0, np.nan))
# session flags in MYT
m15_feat = m15_feat.join(make_session_flags(m15_feat.index, tz_offset_hours=8))

# 4) enforce true overlap window first
# keep only M15 rows that are within H1 feature date range
m15_overlap = m15.loc[h1_feat.index.min(): h1_feat.index.max()].copy()

# 5) align H1 features down to M15 by forward fill within each hour
h1_down = h1_feat.reindex(m15_overlap.index, method="ffill")

# 6) build M15 features on the overlap index
m15_feat = pd.DataFrame(index=m15_overlap.index)
m15_feat["m15_close"] = m15_overlap["close"]
m15_feat["m15_ema20"] = ema(m15_overlap["close"], 20)
m15_feat["m15_atr14"] = atr(m15_overlap["high"], m15_overlap["low"], m15_overlap["close"], period=14)
m15_feat["m15_spread_to_atr"] = m15_overlap["spread_points"] / m15_feat["m15_atr14"].replace(0, np.nan)
m15_feat = m15_feat.join(make_session_flags(m15_feat.index, tz_offset_hours=8))

# 7) require warmup readiness on both sides
need_h1 = ["h1_sma20","h1_sma200","h1_adx14","h1_atr14","h1_atr_pct","h1_vol_pct"]
need_m15 = ["m15_ema20","m15_atr14"]

mask_h1_ready  = h1_down[need_h1].notna().all(axis=1)
mask_m15_ready = m15_feat[need_m15].notna().all(axis=1)
mask = mask_h1_ready & mask_m15_ready

# 8) combine only ready rows
feat = pd.concat([m15_feat.loc[mask], h1_down.loc[mask]], axis=1)

# 9) sanity prints
print("Ranges")
print("  H1 features:", str(h1_feat.index.min()), "to", str(h1_feat.index.max()))
print("  M15 raw    :", str(m15.index.min()), "to", str(m15.index.max()))
print("  M15 overlap:", str(m15_overlap.index.min()), "to", str(m15_overlap.index.max()))
print("Mask counts")
print("  H1 ready  :", int(mask_h1_ready.sum()))
print("  M15 ready :", int(mask_m15_ready.sum()))
print("  Combined  :", int(mask.sum()))

# 10) save outputs
h1_feat.to_parquet(OUT_H1)
feat.to_parquet(OUT_M15)

print(f"Saved H1 features to {OUT_H1}")
print(f"Saved M15 combined features to {OUT_M15}")
print("Rows H1:", len(h1_feat), "Rows M15 features:", len(feat))
print("Range M15 features:", str(feat.index.min()), "to", str(feat.index.max()))