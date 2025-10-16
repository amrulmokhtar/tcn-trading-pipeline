# scripts/phase4_train_tcn.py
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# paths
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "dataset.npz"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
MODEL_OUT = MODEL_DIR / "tcn_model.pt"
METRICS_OUT = RESULTS_DIR / "training_metrics.json"

# hyperparameters, tweak later
SEED = 1337
BATCH_SIZE = 512
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.15
CHANNELS = [64, 64, 64]  # TCN stacks
PATIENCE = 5  # early stopping

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset():
    pack = np.load(DATASET_PATH, allow_pickle=True)
    X = pack["X"]              # [N, L, F], float32, already scaled
    y = pack["y"]              # [N], int8 in {-1,0,1}
    tr_idx = pack["tr_idx"]
    va_idx = pack["va_idx"]
    te_idx = pack["te_idx"]

    # remap labels to {0,1,2} for CrossEntropyLoss
    # -1 -> 0, 0 -> 1, 1 -> 2
    y_map = { -1:0, 0:1, 1:2 }
    y_remap = np.vectorize(y_map.get)(y).astype(np.int64)

    Xt, yt = X[tr_idx], y_remap[tr_idx]
    Xv, yv = X[va_idx], y_remap[va_idx]
    Xe, ye = X[te_idx], y_remap[te_idx]

    return (Xt, yt), (Xv, yv), (Xe, ye)

def to_loader(X, y, batch=BATCH_SIZE, shuffle=False):
    # TCN expects [N, C, L], where C is features, L is time
    # Our X is [N, L, F], so we permute to [N, F, L]
    X_t = torch.from_numpy(X).permute(0, 2, 1).contiguous()
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=False)

# ===== replace your TCN import and class with this self contained version =====
import torch
from torch import nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # pad only to the right, will be chomped
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                  padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                  padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class SimpleTCN(nn.Module):
    def __init__(self, in_ch, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            d = 2 ** i
            in_c = in_ch if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_c, out_ch, kernel_size, dilation=d, dropout=dropout))
        self.towers = nn.Sequential(*layers)

    def forward(self, x):
        # x, [B, C, L]
        y = self.towers(x)           # [B, C_out, L]
        return y[:, :, -1]           # take last time step

class TCNClassifier(nn.Module):
    def __init__(self, in_ch, n_classes, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.backbone = SimpleTCN(in_ch, channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Linear(channels[-1], n_classes)

    def forward(self, x):
        feats = self.backbone(x)     # [B, hidden]
        return self.head(feats)      # [B, n_classes]
# ===== end replacement =====


def class_weights(y_train, n_classes=3):
    # y_train in {0,1,2}, weight inverse to frequency
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv * (n_classes / inv.sum())
    return torch.tensor(weights, dtype=torch.float32)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        all_pred.append(pred.cpu().numpy())
        all_true.append(yb.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    return report, cm

def train():
    # data
    (Xt, yt), (Xv, yv), (Xe, ye) = load_dataset()
    in_ch = Xt.shape[2]
    n_classes = 3

    train_loader = to_loader(Xt, yt, shuffle=True)
    val_loader = to_loader(Xv, yv, shuffle=False)
    test_loader = to_loader(Xe, ye, shuffle=False)

    # model
    model = TCNClassifier(in_ch=in_ch, n_classes=n_classes, channels=CHANNELS, dropout=DROPOUT).to(DEVICE)

    # loss with class weights
    cw = class_weights(yt, n_classes=n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_f1 = -1.0
    best_state = None
    patience = PATIENCE
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(1, steps)

        # validation
        val_report, val_cm = evaluate(model, val_loader, DEVICE)
        val_acc = float(val_report["accuracy"])
        # F1 macro treats classes evenly, good for imbalance
        val_f1 = float(val_report["macro avg"]["f1-score"])
        scheduler.step(val_f1)

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_acc": val_acc,
            "val_f1_macro": val_f1,
            "val_cm": val_cm.tolist()
        })

        print(f"Epoch {epoch:02d}  loss {avg_loss:.4f}  val_acc {val_acc:.4f}  val_f1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                "model": model.state_dict(),
                "in_ch": in_ch,
                "n_classes": n_classes,
                "channels": CHANNELS,
                "dropout": DROPOUT
            }
            patience = PATIENCE
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping, no improvement")
                break

    # save best model
    if best_state is not None:
        torch.save(best_state, MODEL_OUT)
        print(f"Saved model to {MODEL_OUT}")

    # final evaluation on test set
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    test_report, test_cm = evaluate(model, test_loader, DEVICE)
    print("Test accuracy:", float(test_report["accuracy"]))
    print("Test macro F1:", float(test_report["macro avg"]["f1-score"]))
    print("Test confusion matrix, rows true, cols pred, order [-1, 0, 1] -> [0, 1, 2]:")
    print(test_cm)

    # persist metrics
    out = {
        "best_val_f1_macro": best_val_f1,
        "test_report": test_report,
        "test_confusion_matrix": test_cm.tolist(),
        "hyperparams": {
            "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "channels": CHANNELS
        }
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metrics to {METRICS_OUT}")

if __name__ == "__main__":
    train()
