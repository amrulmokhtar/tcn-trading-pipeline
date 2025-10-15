# scripts/phase3_dataset.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEAT_PATH = DATA_DIR / "m15_features.parquet"
OUT_NPZ   = DATA_DIR / "dataset.npz"
OUT_SPLIT = DATA_DIR / "splits.json"
OUT_SCAL  = MODEL_DIR / "scaler.pkl"

# ----------------------
# Config, tweak as needed
# ----------------------
LOOKBACK = 128          # past bars fed to the model
HORIZON  = 4            # predict next 4 bar move
RET_COL  = "m15_close"  # base price column for returns
RET_THR  = 0.5          # in ATR units, threshold for long or short vs flat
VAL_PCT  = 0.15
TEST_PCT = 0.15
MIN_ROWS = 5000         # guard against tiny dataset

# Feature columns, exclude targets or pure identifiers
# Keep all engineered features, you can refine later
EXCLUDE_PREFIXES = []   # example, ["asia_session"] if you want to drop
# ----------------------

def sanitize_features(df, clip_q=0.999):
    # keep only numeric columns
    df = df.select_dtypes(include=[np.number]).copy()

    # replace inf with NaN, then forward fill a little, then drop leftover NaN rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # small forward fill and back fill to handle tiny gaps
    df = df.ffill(limit=2).bfill(limit=2)

    # guard against zero ATR in divisions
    if "m15_atr14" in df.columns:
        df.loc[df["m15_atr14"] <= 0, "m15_atr14"] = np.nan

    # clip heavy tailed features, especially ratios like spread_to_atr
    for col in df.columns:
        s = df[col]
        if not np.issubdtype(s.dtype, np.number):
            continue
        q_low = s.quantile(1.0 - clip_q)
        q_hi  = s.quantile(clip_q)
        # skip if quantiles are NaN
        if pd.isna(q_low) or pd.isna(q_hi):
            continue
        # widen a bit to avoid over clipping
        lo = s.quantile(0.001)
        hi = s.quantile(0.999)
        if pd.notna(lo) and pd.notna(hi) and lo < hi:
            df[col] = s.clip(lo, hi)

    # finally, drop any row that still has NaN after fills
    df = df.dropna()
    return df

def make_supervised(df, lookback, horizon, ret_thr):
    # future return in ATR units, protect against zero or NaN ATR
    atr = df["m15_atr14"].replace([0, np.inf, -np.inf], np.nan).ffill().bfill()
    fwd = df[RET_COL].shift(-horizon)
    ret_abs = (fwd - df[RET_COL])
    ret_atr = ret_abs / atr.replace(0, np.nan)

    # targets
    tgt = pd.Series(0, index=df.index, dtype=int)
    tgt[ret_atr >  ret_thr] = 1
    tgt[ret_atr < -ret_thr] = -1

    # drop tail that lacks horizon and any row with NaN in needed pieces
    valid = tgt.index[:-horizon]
    df = df.loc[valid]
    tgt = tgt.loc[valid]

    # feature columns, numeric only
    feat_cols = [c for c in df.columns if c != RET_COL and np.issubdtype(df[c].dtype, np.number)]
    X_list, y_list = [], []
    idx_end = df.index.to_numpy()
    vals = df[feat_cols].to_numpy(dtype=np.float32)

    for i in range(lookback, len(df)):
        win = vals[i - lookback:i, :]
        if not np.isfinite(win).all():
            continue
        X_list.append(win)
        y_list.append(tgt.iloc[i])

    X = np.stack(X_list, axis=0) if X_list else np.empty((0, lookback, len(feat_cols)), dtype=np.float32)
    y = np.array(y_list, dtype=np.int8)
    end_times = idx_end[lookback:lookback + len(y)]
    return X, y, feat_cols, end_times


def time_splits(n_rows, val_pct, test_pct):
    n_test = int(round(n_rows * test_pct))
    n_val  = int(round(n_rows * val_pct))
    n_train = n_rows - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough rows to split")
    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    test_idx  = np.arange(n_train + n_val, n_rows)
    return train_idx, val_idx, test_idx

def scale_3d(X, train_idx):
    if X.size == 0:
        raise ValueError("No samples after sanitization, check feature generation.")
    N, L, F = X.shape
    X2 = X.reshape(N * L, F)

    # all finite check, in case any slipped through
    if not np.isfinite(X2).all():
        bad = np.where(~np.isfinite(X2))
        raise ValueError(f"Non-finite values remain in features at positions {bad[0][:5]}, {bad[1][:5]}")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # fit only on train rows
    train_rows = np.hstack([np.arange(i * L, i * L + L) for i in train_idx])
    scaler.fit(X2[train_rows])

    X_scaled = scaler.transform(X2).reshape(N, L, F)
    return X_scaled, scaler

def main():
    df = pd.read_parquet(FEAT_PATH)
    # strong cleanup to avoid NaN or inf or extreme spikes
    df = sanitize_features(df, clip_q=0.999)

    if len(df) < MIN_ROWS:
        raise ValueError(f"Dataset too small, got {len(df)} rows")

    X, y, feat_cols, times = make_supervised(df, LOOKBACK, HORIZON, RET_THR)

    # time splits
    tr_idx, va_idx, te_idx = time_splits(len(X), VAL_PCT, TEST_PCT)

    # scale using train only
    Xs, scaler = scale_3d(X, tr_idx)
    joblib.dump(scaler, OUT_SCAL)

    # save arrays, keep splits and metadata
    np.savez_compressed(
        OUT_NPZ,
        X=Xs.astype(np.float32),
        y=y.astype(np.int8),
        times=times.astype("datetime64[ns]"),
        tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx,
        lookback=np.array([LOOKBACK], dtype=np.int32),
        horizon=np.array([HORIZON], dtype=np.int32),
        feat_cols=np.array(feat_cols, dtype=object)
    )

    meta = {
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "ret_threshold_atr": RET_THR,
        "val_pct": VAL_PCT,
        "test_pct": TEST_PCT,
        "n_samples": int(len(X)),
        "n_features": int(len(feat_cols)),
        "train_samples": int(len(tr_idx)),
        "val_samples": int(len(va_idx)),
        "test_samples": int(len(te_idx)),
        "timespan_first": str(pd.to_datetime(times[0])),
        "timespan_last": str(pd.to_datetime(times[-1])),
        "feature_cols": feat_cols,
    }
    with open(OUT_SPLIT, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved", OUT_NPZ)
    print("Saved", OUT_SPLIT)
    print("Saved", OUT_SCAL)
    print("Shapes, X:", Xs.shape, "y:", y.shape)
    print("Splits, train or val or test:", len(tr_idx), len(va_idx), len(te_idx))

if __name__ == "__main__":
    main()
