import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix

# -------- config --------
CSV = "TRAIN_TABLE.csv"             # input table you built earlier
MODEL = "tcn_signal.onnx"
SPEC = "feature_spec.json"
OUT_CSV = "predictions.csv"

HORIZON = 6                         # must match how you labeled, change if needed
THRESH_LONG = 0.60                  # example thresholds, only used for PnL filtering summary
THRESH_SHORT = 0.60

# -------- load spec, data, model --------
with open(SPEC, "r") as f:
    spec = json.load(f)

features = spec["features"]
scaler = spec["scaler"]
classes = spec["class_order"]
win = int(spec["window"])
nfeat = len(features)

df = pd.read_csv(CSV, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

# ensure required columns exist
missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns in CSV: {missing}")

# enforce dtype
for c in features:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=features + ["target_class"]).reset_index(drop=True)

# normalize with saved scaler
X = df[features].to_numpy(dtype=np.float32)
for j, c in enumerate(features):
    m = float(scaler[c]["mean"])
    s = float(scaler[c]["std"])
    s = s if s != 0.0 else 1.0
    X[:, j] = (X[:, j] - m) / s

y_true = df["target_class"].astype("category").cat.set_categories(classes).cat.codes.to_numpy()
ts = df["timestamp"].to_numpy()
open_px = df["open"].to_numpy() if "open" in df.columns else None
close_px = df["close"].to_numpy() if "close" in df.columns else None

# load onnx
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
out = sess.get_outputs()[0].name

# -------- sliding inference --------
# predictions only where we have a full window and room for horizon exit
start = win
end = len(df) - HORIZON - 1
N = max(0, end - start + 1)

pred_idx = np.zeros(N, dtype=np.int64)
probs = np.zeros((N, len(classes)), dtype=np.float32)
rows = []

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

for k, i in enumerate(range(start, end + 1)):
    # window covers rows [i-win, i)
    xw = X[i - win:i, :]                               # [win, feat]
    xw = xw[np.newaxis, :, :].astype(np.float32)       # [1, win, feat]
    logits = sess.run([out], {inp: xw})[0]             # [1, 3]
    p = softmax(logits).astype(np.float32)[0]
    probs[k] = p
    pred_idx[k] = int(np.argmax(p))

    # simple horizon PnL, enter at next bar open, exit at i + HORIZON close
    pnl = np.nan
    if open_px is not None and close_px is not None:
        entry_open = open_px[i + 1]
        exit_close = close_px[i + HORIZON]
        if pred_idx[k] == classes.index("Long"):
            pnl = exit_close - entry_open
        elif pred_idx[k] == classes.index("Short"):
            pnl = entry_open - exit_close

    rows.append({
        "timestamp": ts[i],                      # time at end of input window
        "y_true": int(y_true[i]),
        "y_pred": int(pred_idx[k]),
        "p_long": float(probs[k, classes.index("Long")]),
        "p_short": float(probs[k, classes.index("Short")]),
        "p_none": float(probs[k, classes.index("None")]),
        "pnl_h{0}".format(HORIZON): pnl
    })

pred_df = pd.DataFrame(rows)
pred_df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(pred_df)} rows to {OUT_CSV}")

# -------- metrics --------
true_slice = y_true[start:end + 1]
print(confusion_matrix(true_slice, pred_idx, labels=list(range(len(classes)))))
print(classification_report(true_slice, pred_idx, labels=list(range(len(classes))), target_names=classes, digits=3, zero_division=0))

# filtered PnL summary, only trades with prob above thresholds
if "pnl_h{0}".format(HORIZON) in pred_df.columns:
    mask_long = (pred_df["y_pred"] == classes.index("Long")) & (pred_df["p_long"] >= THRESH_LONG)
    mask_short = (pred_df["y_pred"] == classes.index("Short")) & (pred_df["p_short"] >= THRESH_SHORT)
    mask = mask_long | mask_short
    trades = pred_df[mask].copy()
    wins = trades["pnl_h{0}".format(HORIZON)] > 0
    print("\nFiltered trade summary")
    print(f"trades: {len(trades)}")
    if len(trades) > 0:
        print(f"win rate: {wins.mean():.3f}")
        print(f"avg pnl: {trades['pnl_h{0}'.format(HORIZON)].mean():.6f}")
        print(f"sum pnl: {trades['pnl_h{0}'.format(HORIZON)].sum():.6f}")
