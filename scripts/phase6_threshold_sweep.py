import os
import subprocess
import json
import pandas as pd

# Define sweep parameters
thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
combine_mode = "ml_gate_rules"  # or "rules_gate_ml" / "both"
features = "data/m15_features.parquet"
predictions = "results/phase6_predictions.csv"
rules_trades = "results/phase5_rules_trades.csv"

# Directory for results
os.makedirs("results/sweeps", exist_ok=True)

records = []

print(f"\n=== Running ML Threshold Sweep for mode: {combine_mode} ===")

for th in thresholds:
    out_dir = f"results/sweeps/th_{th:.2f}".replace('.', '_')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n>> Threshold: {th:.2f}")
    cmd = [
        "python", "scripts/phase6_step4_hybrid_backtest.py",
        "--features", features,
        "--predictions", predictions,
        "--rules_trades", rules_trades,
        "--combine_mode", combine_mode,
        "--ml_threshold", str(th),
        "--out_dir", out_dir
    ]
    subprocess.run(cmd, check=True)

    # Load summary output (JSON)
    summary_path = os.path.join(out_dir, "phase6_hybrid_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
        data["threshold"] = th
        records.append(data)

# Consolidate all thresholds into one CSV
if records:
    df = pd.DataFrame(records)
    df.to_csv(f"results/sweeps/phase6_threshold_sweep_{combine_mode}.csv", index=False)
    print(f"\n✅ Consolidated sweep results saved to results/sweeps/phase6_threshold_sweep_{combine_mode}.csv")
else:
    print("\n⚠️ No valid summary files found, check paths or run logs.")
