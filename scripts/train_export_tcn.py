# scripts/train_export_tcn.py
# TCN model definition and a self contained training and export CLI.
# Safe to import: no CSV reads or training happen at import time.

from typing import List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------ blocks ------------------------------

class TemporalBlock(nn.Module):
    """
    Dilated causal 1D Conv block with residual connection.
    Expects input shaped [N, C, L].
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 k: int = 3,
                 d: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=d)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, dilation=d)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.relu = nn.ReLU(inplace=True)

        # kaiming init, zero biases
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, C, L]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# ------------------------------ model ------------------------------

class TCN(nn.Module):
    """
    Temporal Convolutional Network head.
    Constructor and forward are aligned to your previous file:
      __init__(self, in_feat, channels=(64,64,64,64), k=3, dropout=0.1, n_classes=3)
      forward(x) expects [N, L, F]
    Internally we transpose to [N, F, L], pass through TCN blocks,
    global average pool over time, then a Linear to n_classes.
    """
    def __init__(self,
                 in_feat: int,
                 channels: Tuple[int, ...] = (64, 64, 64, 64),
                 k: int = 3,
                 dropout: float = 0.1,
                 n_classes: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        c_in = in_feat
        for i, c in enumerate(channels):
            d = 2 ** i
            layers.append(TemporalBlock(c_in, c, k=k, d=d, dropout=dropout))
            c_in = c
        self.tcn  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(c_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, L, F] -> [N, F, L]
        x = x.transpose(1, 2).contiguous()
        z = self.tcn(x)                 # [N, C, L]
        h = self.pool(z).squeeze(-1)    # [N, C]
        return self.fc(h)               # [N, n_classes]


# ------------------------------ CLI training ------------------------------

def _infer_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _make_windows(x: torch.Tensor, y: torch.Tensor, seq_len: int):
    """
    x: [T, F], y: [T] integer class labels
    returns Xw [N, L, F], Yw [N]
    """
    T = x.shape[0]
    if T < seq_len:
        return torch.empty(0, seq_len, x.shape[1]), torch.empty(0, dtype=torch.long)
    Xw = []
    Yw = []
    for i in range(seq_len - 1, T):
        Xw.append(x[i - seq_len + 1:i + 1])
        Yw.append(y[i])
    return torch.stack(Xw, dim=0), torch.stack(Yw, dim=0)


def _class_mapping(names: List[str]):
    """
    Map string labels to indices. Default order keeps compatibility with your pipeline.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    return name_to_idx


def train_and_export(csv_path: str,
                     out_state_dict: str,
                     out_jit: Optional[str],
                     timestamp_col: str,
                     feature_cols: List[str],
                     label_col: str,
                     class_names: List[str],
                     seq_len: int = 64,
                     batch_size: int = 256,
                     lr: float = 1e-3,
                     epochs: int = 5,
                     seed: int = 1337,
                     clip_grad: Optional[float] = 1.0):
    """
    Minimal training loop to produce a state_dict compatible with TCN above.
    """
    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = _infer_device()

    # load data
    df = pd.read_csv(csv_path)
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.sort_values(timestamp_col).reset_index(drop=True)

    # build features and labels
    assert all(c in df.columns for c in feature_cols), f"Missing feature columns, need {feature_cols}"
    assert label_col in df.columns, f"Missing label column {label_col}"

    x_np = df[feature_cols].astype("float32").to_numpy()
    # label can be already numeric or string
    if df[label_col].dtype == object:
        name_to_idx = _class_mapping(class_names)
        y_np = df[label_col].map(name_to_idx).astype("int64").to_numpy()
    else:
        y_np = df[label_col].astype("int64").to_numpy()

    x = torch.from_numpy(x_np)  # [T, F]
    y = torch.from_numpy(y_np)  # [T]

    Xw, Yw = _make_windows(x, y, seq_len)
    if Xw.numel() == 0:
        raise ValueError(f"Not enough rows to form windows of length {seq_len}")

    ds = TensorDataset(Xw, Yw)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # model
    model = TCN(in_feat=x.shape[1], channels=(64, 64, 64, 64), k=3, dropout=0.1, n_classes=len(class_names))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs + 1):
        total, correct, loss_sum, n = 0, 0, 0.0, 0
        for xb, yb in dl:
            xb = xb.to(device)           # [N, L, F]
            yb = yb.to(device)           # [N]
            opt.zero_grad()
            logits = model(xb)           # [N, C]
            loss = crit(logits, yb)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                total += yb.size(0)
                correct += (pred == yb).sum().item()
                loss_sum += float(loss.item()) * yb.size(0)
                n += yb.size(0)

        acc = correct / max(1, total)
        avg_loss = loss_sum / max(1, n)
        print(f"epoch {ep:02d} loss {avg_loss:.4f} acc {acc:.4f}")

    # export
    Path(out_state_dict).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_state_dict)
    print(f"saved state_dict to {out_state_dict}")

    if out_jit:
        model.eval()
        eg = torch.zeros(1, seq_len, x.shape[1], device=device)  # [N, L, F]
        ts = torch.jit.trace(model, eg)
        Path(out_jit).parent.mkdir(parents=True, exist_ok=True)
        ts.save(out_jit)
        print(f"saved torchscript to {out_jit}")


# ------------------------------ entry point ------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train and export a TCN state_dict or TorchScript")
    ap.add_argument("--csv", default="TRAIN_TABLE.csv", help="Input CSV for quick training")
    ap.add_argument("--timestamp_col", default="timestamp")
    ap.add_argument("--features", default="m15_close,m15_atr14", help="Comma separated feature columns")
    ap.add_argument("--label_col", default="label", help="Target class column")
    ap.add_argument("--class_order", default="SELL,FLAT,BUY", help="Comma order for classes")
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_state", default="models/tcn_model.pt")
    ap.add_argument("--out_jit", default="", help="Optional, path to save TorchScript, leave blank to skip")
    args = ap.parse_args()

    feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    class_names = [c.strip().upper() for c in args.class_order.split(",") if c.strip()]

    out_jit = args.out_jit if args.out_jit else None

    train_and_export(csv_path=args.csv,
                     out_state_dict=args.out_state,
                     out_jit=out_jit,
                     timestamp_col=args.timestamp_col,
                     feature_cols=feature_cols,
                     label_col=args.label_col,
                     class_names=class_names,
                     seq_len=args.seq_len,
                     batch_size=args.batch_size,
                     lr=args.lr,
                     epochs=args.epochs)

if __name__ == "__main__":
    main()
