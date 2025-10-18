# export_torchscript.py
import json, joblib, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

STATE_DICT_PT = MODELS / "tcn_model.pt"     # your current file, change if needed
OUT_TS        = MODELS / "tcn_signal.pt"    # output TorchScript
SCALER_PKL    = MODELS / "scaler.pkl"

# 1) import your real TCN class, update this import
# example:
# from your_package.tcn_model import TCNClassifier
from your_package.tcn_model import TCNClassifier  # TODO, replace with your path

# 2) recover input feature size
def get_feature_count():
    for name in ("SCALER.json", "signal_config.json", "scaler.json"):
        p = MODELS / name
        if p.exists():
            cfg = json.loads(p.read_text())
            for k in ("feature_names","features","cols","columns",
                      "input_features","model_features","trained_features"):
                if k in cfg and isinstance(cfg[k], list):
                    return len(cfg[k])
    sc = joblib.load(SCALER_PKL)
    names = getattr(sc, "feature_names_in_", None)
    if names is not None:
        return len(names)
    n = getattr(sc, "n_features_in_", None)
    if isinstance(n, int):
        return n
    raise RuntimeError("Could not determine feature count")

n_features = get_feature_count()

# 3) build model and load weights, adjust ctor args to your codebase
# training_metrics.json shows channels [64,64,64], dropout 0.15
model = TCNClassifier(input_dim=n_features, num_classes=3, channels=[64,64,64], dropout=0.15)

state = torch.load(STATE_DICT_PT, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
if isinstance(state, dict):
    model.load_state_dict(state, strict=False)
else:
    model = state
model.eval()

# 4) script or trace
example = torch.randn(1, n_features)
try:
    scripted = torch.jit.script(model)
except Exception:
    scripted = torch.jit.trace(model, example)

OUT_TS.parent.mkdir(exist_ok=True, parents=True)
scripted.save(str(OUT_TS))
print("Saved TorchScript to:", OUT_TS)
