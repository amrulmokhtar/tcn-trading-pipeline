# train_export_tcn.py  self contained, no external tcn package needed
import os, json, random, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CSV = "TRAIN_TABLE.csv"
SCALER_JSON = "SCALER.json"
ONNX_OUT = "tcn_signal.onnx"
PT_OUT = "tcn_signal.pt"
FEAT_SPEC = "feature_spec.json"

WIN = 32
HORIZON = 6
BATCH = 256
EPOCHS = 15
LR = 1e-3
VAL_SIZE = 0.15

FEATURES = [
    "open","high","low","close","tick_volume",
    "atr14_m15","sma20_m15","ema50_m15","dist_sma20","dist_ema50",
    "body_over_atr","range_over_atr","logret","obv_delta_m15",
    "atr14p_h1","obv_slope_h1","spread_over_atr",
    "sess_asia","sess_london","sess_ny"
]
CLASSES = ["Long","Short","None"]
CLASS_TO_ID = {c:i for i,c in enumerate(CLASSES)}

# ---------------- dataset ----------------
class SeqDS(Dataset):
    def __init__(self, df, scalers):
        x = df[FEATURES].to_numpy(dtype=np.float32)
        y = df["target_class"].map(CLASS_TO_ID).to_numpy(dtype=np.int64)
        for j, col in enumerate(FEATURES):
            mu = scalers[col]["mean"]; sd = max(1e-8, scalers[col]["std"])
            x[:, j] = (x[:, j] - mu) / sd
        Xw, Yw = [], []
        for i in range(WIN, len(x)):
            Xw.append(x[i-WIN:i, :])     # [WIN, F]
            Yw.append(y[i])
        self.X = np.stack(Xw, axis=0)    # [N, WIN, F]
        self.Y = np.array(Yw)
    def __len__(self): return len(self.Y)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.Y[idx])

# ---------------- model: minimal TCN in PyTorch ----------------
class Chomp1d(nn.Module):
    def __init__(self, chomp): 
        super().__init__(); self.chomp = chomp
    def forward(self, x):                 # x [N, C, L+pad]
        return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, dropout=0.1):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):                 # x [N, C, L]
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_feat, channels=(64,64,64,64), k=3, dropout=0.1, n_classes=3):
        super().__init__()
        layers = []
        c_in = in_feat
        for i, c in enumerate(channels):
            d = 2 ** i
            layers.append(TemporalBlock(c_in, c, k=k, d=d, dropout=dropout))
            c_in = c
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_in, n_classes)
    def forward(self, x):                 # x [N, L, F]
        x = x.transpose(1, 2).contiguous()  # [N, F, L]
        z = self.tcn(x)                     # [N, C, L]
        h = self.pool(z).squeeze(-1)        # [N, C]
        return self.fc(h)                   # [N, n_classes]

# ---------------- load data ----------------
df = pd.read_csv(CSV, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
with open(SCALER_JSON, "r") as f: scalers = json.load(f)

# --- clean invalid labels and NaNs ---
valid_classes = {"Long", "Short", "None"}
df = df[df["target_class"].isin(valid_classes)].copy()
df = df.dropna(subset=FEATURES + ["target_class"]).reset_index(drop=True)
print(f"Cleaned dataset size: {len(df)} rows")

# make sure flags are numeric
for c in ["sess_asia","sess_london","sess_ny"]:
    df[c] = df[c].astype("float32")

# drop first WIN rows so windows form cleanly
df = df.iloc[WIN:].reset_index(drop=True)

split_ix = int(len(df) * (1.0 - VAL_SIZE))
df_tr = df.iloc[:split_ix].reset_index(drop=True)
df_va = df.iloc[split_ix:].reset_index(drop=True)

tr_ds, va_ds = SeqDS(df_tr, scalers), SeqDS(df_va, scalers)
tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, drop_last=True)
va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False)

in_feat = len(FEATURES); classes = len(CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TCN(in_feat=in_feat, channels=(64,64,64,64), k=3, dropout=0.1, n_classes=classes).to(device)

# class weights
cnt = df_tr["target_class"].value_counts().reindex(CLASSES).fillna(1).to_numpy(np.float32)
w = cnt.sum() / (cnt + 1e-6); w = w / w.sum() * classes
crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device, dtype=torch.float32))
opt = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------- train ----------------
best_val = 1e9; best_state = None
for ep in range(1, EPOCHS+1):
    model.train(); tr_loss = 0.0
    for xb, yb in tr_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward(); opt.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(tr_ds)

    model.eval(); va_loss = 0.0
    with torch.no_grad():
        for xb, yb in va_loader:
            xb, yb = xb.to(device), yb.to(device)
            va_loss += crit(model(xb), yb).item() * xb.size(0)
    va_loss /= len(va_ds)
    print(f"Epoch {ep:02d}  train {tr_loss:.4f}  val {va_loss:.4f}")
    if va_loss < best_val:
        best_val = va_loss
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

# restore best and save
model.load_state_dict(best_state)
torch.save(model.state_dict(), PT_OUT); print(f"Saved PyTorch weights to {PT_OUT}")

# ---------------- report ----------------
model.eval()
preds, gts = [], []
with torch.no_grad():
    for xb, yb in va_loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(yb.numpy())

preds = np.concatenate(preds)
gts = np.concatenate(gts)

# ✅ FIXED REPORT BLOCK
from sklearn.metrics import classification_report, confusion_matrix

labels_all = [0, 1, 2]  # Long, Short, None
print(confusion_matrix(gts, preds, labels=labels_all))
print(classification_report(
    gts, preds,
    labels=labels_all,
    target_names=CLASSES,
    digits=3,
    zero_division=0
))

# ---------------- export ONNX ----------------
dummy = torch.zeros(1, WIN, in_feat, dtype=torch.float32).to(device)
torch.onnx.export(
    model, dummy, ONNX_OUT,
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=13
)
print(f"Saved ONNX to {ONNX_OUT}")

# ---------------- feature spec for EA ----------------
spec = {
    "window": WIN,
    "features": FEATURES,
    "class_order": CLASSES,
    "scaler": {k: {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in scalers.items()}
}
with open(FEAT_SPEC, "w") as f: json.dump(spec, f, indent=2)
print(f"Saved feature spec to {FEAT_SPEC}")
