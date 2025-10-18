# scripts/phase6_hybrid_backtest.py
# Phase 6 Step 2, prediction scaffold with trained feature alignment and MYT timestamps
# Outputs:
#   results/phase6_predictions.csv
#   results/phase6_predictions_meta.json

from typing import Optional, List, Tuple
from pathlib import Path
import argparse
import warnings
import json

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn.functional as F

# --- import bootstrap so we can import training code ---
import sys, importlib.util
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ------------------------------------------------------

# Reuse Phase 5 paths and time settings if available
try:
    import phase5_backtest as pb  # provides FEAT_PATH, TZ_OFFSET_HOURS, COL
    FEAT_PATH_DEFAULT = pb.FEAT_PATH
    TZ_OFFSET_HOURS = pb.TZ_OFFSET_HOURS
    TS_COL = pb.COL.get("ts", "ts")
except Exception:
    FEAT_PATH_DEFAULT = Path("data") / "m15_features.parquet"
    TZ_OFFSET_HOURS = +8
    TS_COL = "ts"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Locked class mapping for Phase 6
CLASS_TO_ACTION = {0: "SELL", 1: "FLAT", 2: "BUY"}
DEFAULT_THRESHOLD = 0.60

# --------------- time helpers ---------------

def to_utc_series(ts_like: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts_like, errors="coerce", utc=True)
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    return ts

def add_time_columns(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    if ts_col in df.columns:
        ts_utc = to_utc_series(df[ts_col])
    else:
        warnings.warn(f"Timestamp column '{ts_col}' not found, synthetic timestamps will be generated.")
        ts_utc = pd.date_range("2000-01-01", periods=len(df), freq="15min", tz="UTC")
    df = df.copy()
    df["timestamp_utc"] = ts_utc
    df["timestamp_myt"] = ts_utc + pd.Timedelta(hours=TZ_OFFSET_HOURS)
    return df

# --------------- feature helpers ---------------

POSSIBLE_KEYS = (
    "feature_names", "features", "cols", "columns",
    "input_features", "model_features", "trained_features"
)

def pick_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_like = {TS_COL.lower(), "ts", "timestamp", "time", "date", "y", "label", "target"}
    keep: List[str] = []
    for c in df.columns:
        cl = str(c).lower()
        if cl in drop_like:
            continue
        if df[c].dtype.kind in ("i", "u", "f", "b"):
            keep.append(c)
    if not keep:
        raise ValueError("No numeric feature columns found after dropping time and label columns.")
    return keep

def _read_feature_list_json(p: Path) -> Optional[List[str]]:
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            cfg = json.load(f)
        if isinstance(cfg, list):
            return [str(x) for x in cfg]
        if isinstance(cfg, dict):
            for k in POSSIBLE_KEYS:
                if k in cfg and isinstance(cfg[k], list):
                    return [str(x) for x in cfg[k]]
        return None
    except Exception:
        return None

def load_trained_feature_list(scaler_path: Path) -> Tuple[Optional[List[str]], Optional[int]]:
    """
    Return (trained_cols, n_expected).
    Try SCALER.json, signal_config.json, scaler.json.
    Fall back to scaler.feature_names_in_ or scaler.n_features_in_.
    """
    parent = scaler_path.parent
    for name in ("SCALER.json", "signal_config.json", "scaler.json"):
        cols = _read_feature_list_json(parent / name)
        if cols:
            return list(cols), None

    try:
        sc = joblib.load(scaler_path)
        names = getattr(sc, "feature_names_in_", None)
        if names is not None:
            return list(names), None
        n_expected = getattr(sc, "n_features_in_", None)
        if isinstance(n_expected, int):
            return None, int(n_expected)
    except Exception:
        pass
    return None, None

def align_features(
    df_num: pd.DataFrame,
    trained_cols: Optional[List[str]],
    n_expected: Optional[int]
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Align to trained_cols when available, else trim to n_expected with a stable order.
    Returns X_aligned, used_cols, dropped_cols.
    """
    X = df_num.copy()
    used_cols: List[str] = []
    dropped_cols: List[str] = []

    if trained_cols:
        for c in trained_cols:
            if c not in X.columns:
                X[c] = 0.0
        used_cols = list(trained_cols)
        dropped_cols = [c for c in X.columns if c not in used_cols]
        X = X[used_cols]
    elif isinstance(n_expected, int):
        ordered = sorted(list(X.columns))
        used_cols = ordered[:n_expected]
        dropped_cols = [c for c in X.columns if c not in used_cols]
        X = X[used_cols]
    else:
        used_cols = list(X.columns)

    # clean numeric
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=-1e6, upper=1e6)
    X = X.fillna(0.0).astype(np.float32)
    return X, used_cols, dropped_cols

# --------------- model loader ---------------

def load_model(model_path: Path, input_dim: int = None, num_classes: int = 3):
    """
    Try TorchScript first, then rebuild TCNClassifier from your training file and load a state_dict.
    Constructor args are detected dynamically. Handles aliases like in_ch, in_channels, out_ch, n_classes.
    """
    import inspect, importlib.util, sys as _sys

    # TorchScript path
    try:
        m = torch.jit.load(str(model_path), map_location="cpu")
        m.eval()
        return m, "torchscript"
    except Exception:
        pass

    # Import TCNClassifier from training code, with file based fallback
    TCNClassifier = None
    try:
        from scripts.phase4_train_tcn import TCNClassifier as _TCN
        TCNClassifier = _TCN
    except Exception:
        spec = importlib.util.spec_from_file_location(
            "phase4_train_tcn", _ROOT / "scripts" / "phase4_train_tcn.py"
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            TCNClassifier = getattr(mod, "TCNClassifier", None)
    if TCNClassifier is None:
        raise RuntimeError("Could not import TCNClassifier from scripts.phase4_train_tcn")

    # Load checkpoint
    ckpt = torch.load(str(model_path), map_location="cpu")
    if isinstance(ckpt, torch.nn.Module):
        ckpt.eval()
        return ckpt, "raw_module"
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint type, {type(ckpt)}")

    # Inspect constructor
    sig = inspect.signature(TCNClassifier.__init__)
    params = list(sig.parameters.values())[1:]  # skip self
    names = [p.name for p in params]
    required = {p.name for p in params if p.default is inspect._empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}

    # Map aliases
    alias_values = {
        # input size
        "input_dim": input_dim, "in_dim": input_dim, "in_ch": input_dim, "in_channels": input_dim,
        "n_inputs": input_dim, "num_features": input_dim, "in_features": input_dim, "features": input_dim,
        # class count
        "num_classes": num_classes, "n_classes": num_classes, "classes": num_classes, "out_ch": num_classes, "out_channels": num_classes,
        # common hyper params
        "channels": [64, 64, 64], "dropout": 0.15,
    }

    kwargs = {}
    for n in names:
        if n in alias_values and alias_values[n] is not None:
            kwargs[n] = alias_values[n]

    # Ensure required keys are present, fill minimum set
    if "channels" in required and "channels" not in kwargs:
        kwargs["channels"] = [64, 64, 64]
    if any(k in required for k in ("in_ch", "in_channels", "input_dim", "in_dim")) and not any(k in kwargs for k in ("in_ch","in_channels","input_dim","in_dim")):
        if input_dim is None:
            raise RuntimeError("Model constructor requires input dimension but input_dim is None")
        # prefer in_ch
        if "in_ch" in names:
            kwargs["in_ch"] = input_dim
        elif "in_channels" in names:
            kwargs["in_channels"] = input_dim
        elif "input_dim" in names:
            kwargs["input_dim"] = input_dim

    if any(k in required for k in ("out_ch", "out_channels", "num_classes", "n_classes", "classes")) and not any(k in kwargs for k in ("out_ch","out_channels","num_classes","n_classes","classes")):
        if "out_ch" in names:
            kwargs["out_ch"] = num_classes
        elif "out_channels" in names:
            kwargs["out_channels"] = num_classes
        elif "num_classes" in names:
            kwargs["num_classes"] = num_classes
        elif "n_classes" in names:
            kwargs["n_classes"] = num_classes
        elif "classes" in names:
            kwargs["classes"] = num_classes

    # Build model
    try:
        model = TCNClassifier(**kwargs)
    except TypeError:
        # last resort positional, order the classic trio, in_ch, channels, out_ch if present
        pos = []
        if "in_ch" in names or "in_channels" in names or "input_dim" in names or "in_dim" in names:
            pos.append(input_dim)
        if "channels" in names:
            pos.append([64, 64, 64])
        if "out_ch" in names or "out_channels" in names or "num_classes" in names or "n_classes" in names or "classes" in names:
            pos.append(num_classes)
        model = TCNClassifier(*pos)

    # Load weights
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing keys sample:", missing[:5])
        if unexpected:
            print("  unexpected keys sample:", unexpected[:5])

    model.eval()
    return model, "state_dict"

# --------------- main ---------------

def main():
    ap = argparse.ArgumentParser(description="Phase 6 predictions with feature alignment")
    ap.add_argument("--features", type=str, default=str(FEAT_PATH_DEFAULT), help="Path to m15_features.parquet")
    ap.add_argument("--scaler", type=str, default="scaler.pkl", help="Path to scaler.pkl")
    ap.add_argument("--model", type=str, default="tcn_model.pt", help="Path to tcn_model.pt")
    ap.add_argument("--symbol", type=str, default="XAUUSD", help="Symbol for outputs")
    ap.add_argument("--prob_th", type=float, default=DEFAULT_THRESHOLD, help="Min probability to act")
    ap.add_argument("--out", type=str, default=str(RESULTS_DIR / "phase6_predictions.csv"))
    args = ap.parse_args()

    feat_path = Path(args.features)
    scaler_path = Path(args.scaler)
    model_path = Path(args.model)
    out_path = Path(args.out)

    # Load features
    df = pd.read_parquet(feat_path)
    if TS_COL not in df.columns:
        warnings.warn(f"Timestamp column '{TS_COL}' not found, synthetic timestamps will be generated.")
        df[TS_COL] = pd.NaT
    df = add_time_columns(df, TS_COL)

    # Pick numeric candidates
    feat_cols_all = pick_feature_columns(df)
    df_num = df[feat_cols_all]

    # Load scaler and determine trained feature spec
    scaler = joblib.load(scaler_path)
    trained_cols, n_expected = load_trained_feature_list(scaler_path)

    # Align to trained set and order
    X, used_cols, dropped_cols = align_features(df_num, trained_cols, n_expected)
    print(f"Using {len(used_cols)} features")
    if dropped_cols:
        print(f"Dropped {len(dropped_cols)} extras: {dropped_cols[:5]}{' ...' if len(dropped_cols) > 5 else ''}")

    # Scale
    X_scaled = scaler.transform(X)

    # Load model
    model, loader_mode = load_model(model_path, input_dim=X.shape[1], num_classes=3)

    # Inference
    with torch.no_grad():
        xt = torch.from_numpy(np.asarray(X_scaled, dtype=np.float32))
        # Convert numpy -> torch and match expected shape for TCN (B, C, L)
        xt = torch.from_numpy(np.asarray(X_scaled, dtype=np.float32))

        # If shape is (samples, features), add a dummy time dimension
        if xt.ndim == 2:
            xt = xt.unsqueeze(-1)  # (N, F, 1)
        # If shape is (N, seq_len, F), transpose to (N, F, seq_len)
        elif xt.shape[1] < xt.shape[-1]:
            xt = xt.transpose(1, 2)

        # Forward pass
        logits = model(xt)

        # Handle 1D output
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

        # Apply softmax to get probabilities
        prob = F.softmax(logits, dim=-1).cpu().numpy()

    pred_class = prob.argmax(axis=1)
    max_prob = prob.max(axis=1)
    action_raw = np.vectorize(lambda k: CLASS_TO_ACTION.get(int(k), "FLAT"))(pred_class)
    action_gated = np.where(max_prob >= float(args.prob_th), action_raw, "FLAT")

    out = pd.DataFrame({
        "timestamp_utc": df["timestamp_utc"].values,
        "timestamp_myt": df["timestamp_myt"].values,
        "symbol": args.symbol,
        "pred_class": pred_class,
        "prob_short": prob[:, 0],
        "prob_hold":  prob[:, 1],
        "prob_long":  prob[:, 2],
        "max_prob": max_prob,
        "action_raw": action_raw,
        "action_gated": action_gated,
    })
    out.to_csv(out_path, index=False)

    meta = {
        "features_path": str(feat_path),
        "scaler_path": str(scaler_path),
        "model_path": str(model_path),
        "n_rows": int(len(out)),
        "prob_threshold": float(args.prob_th),
        "class_to_action": CLASS_TO_ACTION,
        "tz_myt_offset_hours": TZ_OFFSET_HOURS,
        "used_feature_count": int(len(used_cols)),
        "used_features_preview": used_cols[:50],
        "dropped_feature_count": int(len(dropped_cols)),
        "loader_mode": loader_mode,
        "symbol": args.symbol,
        "ts_col_used": TS_COL,
    }
    with open(RESULTS_DIR / "phase6_predictions_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Saved predictions to {out_path}")
    print(f"Loader mode: {loader_mode}, threshold: {args.prob_th}")

if __name__ == "__main__":
    main()
