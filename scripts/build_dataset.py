import pandas as pd
import numpy as np
import json

M15_CSV = "15M_data.csv"
H1_CSV  = "1H_data.csv"
OUT_TRAIN = "TRAIN_TABLE.csv"
OUT_SAMPLE = "TRAIN_SAMPLE_200.csv"
OUT_SCALER = "SCALER.json"

HORIZON = 6          # predict next 6 M15 bars
ATR_LEN = 14
K_ATR   = 0.5        # label threshold in ATR units

# ---------- helpers ----------
def atr(high, low, close, n=14):
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def obv(close, volume):
    sign = np.sign(close.diff())
    return (sign.fillna(0) * volume.fillna(0)).cumsum()

def session_flags(ts):  # UTC flags. Adjust later if you prefer server time.
    h = ts.dt.hour
    asia   = ((h >= 0) & (h < 7)).astype(int)
    london = ((h >= 7) & (h < 15)).astype(int)
    ny     = ((h >= 13) & (h < 21)).astype(int)
    return asia, london, ny

def make_scaler(df, cols):
    stats = {}
    for c in cols:
        mu = float(df[c].mean())
        sd = float(df[c].std(ddof=0)) if df[c].std(ddof=0) > 0 else 1.0
        stats[c] = {"mean": mu, "std": sd}
    return stats

# ---------- load ----------
m15 = pd.read_csv(M15_CSV, parse_dates=["timestamp"])
h1  = pd.read_csv(H1_CSV,  parse_dates=["timestamp"])

# standardize column names from your logger
for df in [m15, h1]:
    df.rename(columns={
        "open": "open", "high": "high", "low": "low", "close": "close",
        "tick_volume": "tick_volume", "spread_points": "spread_points"
    }, inplace=True)

m15 = m15.sort_values("timestamp").reset_index(drop=True)
h1  = h1.sort_values("timestamp").reset_index(drop=True)

# ---------- features on M15 ----------
m15["atr14_m15"]   = atr(m15["high"], m15["low"], m15["close"], ATR_LEN)
m15["sma20_m15"]   = m15["close"].rolling(20, min_periods=20).mean()
m15["ema50_m15"]   = m15["close"].ewm(span=50, adjust=False).mean()
m15["dist_sma20"]  = (m15["close"] - m15["sma20_m15"]) / m15["atr14_m15"]
m15["dist_ema50"]  = (m15["close"] - m15["ema50_m15"]) / m15["atr14_m15"]
m15["body_over_atr"]  = (m15["close"] - m15["open"]) / m15["atr14_m15"]
m15["range_over_atr"] = (m15["high"] - m15["low"]) / m15["atr14_m15"]
m15["logret"]      = np.log(m15["close"]).diff()
m15["obv_m15"]     = obv(m15["close"], m15["tick_volume"])
m15["obv_delta_m15"] = m15["obv_m15"].diff()

# spread feature scaled by ATR. If spread_points missing, fill 0
if "spread_points" not in m15.columns:
    m15["spread_points"] = 0
m15["spread_over_atr"] = m15["spread_points"] / (m15["atr14_m15"].replace(0, np.nan))
m15["spread_over_atr"] = m15["spread_over_atr"].fillna(0.0)

# ---------- features on H1 then align ----------
h1["atr14_h1"] = atr(h1["high"], h1["low"], h1["close"], ATR_LEN)
h1["atr14p_h1"] = h1["atr14_h1"].rolling(200, min_periods=50).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
)
h1["obv_h1"] = obv(h1["close"], h1.get("tick_volume", pd.Series(index=h1.index, dtype=float)).fillna(0))
h1["obv_slope_h1"] = h1["obv_h1"].diff(5)

h1_small = h1[["timestamp", "atr14p_h1", "obv_slope_h1"]].copy()
m15 = pd.merge_asof(
    m15, h1_small.sort_values("timestamp"),
    on="timestamp", direction="backward"
)

# ---------- session flags ----------
asia, london, ny = session_flags(m15["timestamp"])
m15["sess_asia"]   = asia
m15["sess_london"] = london
m15["sess_ny"]     = ny

# ---------- label ----------
fwd_close = m15["close"].shift(-HORIZON)
fwd_ret   = (fwd_close - m15["close"]) / m15["atr14_m15"]
target = np.where(fwd_ret >= K_ATR, "Long",
          np.where(fwd_ret <= -K_ATR, "Short", "None"))
m15["target_class"] = target

# ---------- clean and save ----------
cols = [
    "timestamp","open","high","low","close","tick_volume",
    "atr14_m15","sma20_m15","ema50_m15","dist_sma20","dist_ema50",
    "body_over_atr","range_over_atr","logret","obv_delta_m15",
    "atr14p_h1","obv_slope_h1","spread_over_atr",
    "sess_asia","sess_london","sess_ny","target_class"
]
df = m15[cols].dropna().reset_index(drop=True)
df.to_csv(OUT_TRAIN, index=False)

# sample
mid = len(df) // 2
sample = df.iloc[max(0, mid-100):mid+100].copy()
sample.to_csv(OUT_SAMPLE, index=False)

# scaler for numeric features
num_cols = [
    "open","high","low","close","tick_volume",
    "atr14_m15","sma20_m15","ema50_m15","dist_sma20","dist_ema50",
    "body_over_atr","range_over_atr","logret","obv_delta_m15",
    "atr14p_h1","obv_slope_h1","spread_over_atr",
    "sess_asia","sess_london","sess_ny"
]
scaler = make_scaler(df, num_cols)
with open(OUT_SCALER, "w") as f:
    json.dump(scaler, f, indent=2)

# report
cls_counts = df["target_class"].value_counts().to_dict()
print(f"Saved {OUT_TRAIN} with {len(df)} rows")
print("Class counts:", cls_counts)
print(f"Saved sample to {OUT_SAMPLE}")
print(f"Saved scaler to {OUT_SCALER}")
