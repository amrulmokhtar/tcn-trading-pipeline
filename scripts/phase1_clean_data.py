import pandas as pd
from pathlib import Path

raw_15m_path = Path("C:/Users/amrul/Documents/tcn_1.0/data/15M_data.csv")
raw_1h_path  = Path("C:/Users/amrul/Documents/tcn_1.0/data/1H_data.csv")

clean_15m_path = Path("C:/Users/amrul/Documents/tcn_1.0/data/cleaned_15M_data.csv")
clean_1h_path  = Path("C:/Users/amrul/Documents/tcn_1.0/data/cleaned_1H_data.csv")

EXPECTED = ["timestamp","open","high","low","close","tick_volume","spread_points","commission_roundtrip_usd"]

def load_df(path, label):
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns, {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp")
    for c in EXPECTED[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def reindex_fill(df, freq, spread=20.0, commission=7.0):
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    missing_idx = full.difference(df.index)
    out = df.reindex(full)
    out["synthetic_bar"] = out.index.isin(missing_idx)

    close_ffill = out["close"].ffill()
    for c in ["open","high","low","close"]:
        out[c] = out[c].where(~out[c].isna(), close_ffill)
    out["tick_volume"] = out["tick_volume"].fillna(0.0)
    out.loc[out["synthetic_bar"], "tick_volume"] = 0.0

    out["spread_points"] = float(spread)
    out["commission_roundtrip_usd"] = float(commission)

    out.index.name = "timestamp"
    return out[["open","high","low","close","tick_volume","spread_points","commission_roundtrip_usd","synthetic_bar"]], len(missing_idx)

# Run cleaning
m15_raw = load_df(raw_15m_path, "M15")
m15_clean, miss15 = reindex_fill(m15_raw, "15min")
m15_clean.to_csv(clean_15m_path)

h1_raw = load_df(raw_1h_path, "H1")
h1_clean, miss1h = reindex_fill(h1_raw, "H")
h1_clean.to_csv(clean_1h_path)

summary = {
    "M15": {
        "start": str(m15_raw.index.min()), "end": str(m15_raw.index.max()),
        "rows_raw": int(len(m15_raw)), "rows_clean": int(len(m15_clean)),
        "missing_filled": int(miss15)
    },
    "H1": {
        "start": str(h1_raw.index.min()), "end": str(h1_raw.index.max()),
        "rows_raw": int(len(h1_raw)), "rows_clean": int(len(h1_clean)),
        "missing_filled": int(miss1h)
    }
}
print(summary)
print("Saved:", str(clean_15m_path), str(clean_1h_path))
