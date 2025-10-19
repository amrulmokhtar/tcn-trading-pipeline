import pandas as pd
import numpy as np
import joblib
import torch
from pathlib import Path
from your_model_file import TCNModel   # 👈 import your model class here

FEATURES = "data/m15_features.parquet"
SCALER = "models/scaler_train_2021_2023.pkl"
MODEL = "models/tcn_model.pt"
OUT = "results/phase7_predictions_2024.csv"

FEAT_COLS = ["m15_close", "m15_atr14"]

# Load and filter features
df = pd.read_parquet(FEATURES)
ts_col = next((c for c in ["timestamp_utc", "ts", "timestamp", "datetime"] if c in df.columns), None)
df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
mask = (df[ts_col] >= "2024-01-01") & (df[ts_col] <= "2024-12-31")
df = df.loc[mask].copy()

# Scale inputs
scaler = joblib.load(SCALER)
X = scaler.transform(df[FEAT_COLS].ffill().bfill().to_numpy())

# Load model and weights
device = torch.device("cpu")
model = TCNModel(...)             # 👈 create same architecture used in training
state_dict = torch.load(MODEL, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Inference
with torch.no_grad():
    X_tensor = torch.from_numpy(X).float().to(device)
    logits = model(X_tensor)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()

ml_side = np.where(prob >= 0.5, 1, 0)
out = pd.DataFrame({
    "timestamp_utc": df[ts_col],
    "action_raw": np.where(ml_side == 1, "BUY", "HOLD"),
    "max_prob": prob,
    "ml_side": ml_side
})
Path("results").mkdir(exist_ok=True)
out.to_csv(OUT, index=False)
print(f"✅ Saved predictions to {OUT}")
