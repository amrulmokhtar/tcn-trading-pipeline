# scripts/sweep_spread_cap.py
import os
import sys
import numpy as np
import pandas as pd

# Ensure we can import "scripts.phase5_backtest"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scripts.phase5_backtest as pb  # noqa: E402

def safe_get(d, k, default=np.nan):
    return (d or {}).get(k, default)

def main():
    # Load the same features file as backtest uses
    feat_path = getattr(pb, "FEAT_PATH", "data/m15_features.parquet")
    df = pd.read_parquet(feat_path)

    # Sweep values (edit as you like)
    caps = [5, 6, 7, 8, 9, 10, 12, 15, 20]
    max_spread_points = 30  # keep this fixed for the sweep

    rows = []
    for cap in caps:
        # Override runtime knobs
        pb.SPREAD_TO_ATR_CAP = cap
        pb.MAX_SPREAD_POINTS = max_spread_points

        # Some versions may return (trades, summary); guard in case
        result = pb.run_backtest(df)
        if isinstance(result, tuple) and len(result) == 2:
            trades, summary = result
        else:
            # Fallback if old code returns only summary
            trades, summary = None, result

        trades_count = 0 if trades is None else len(trades)

        # Extract metrics robustly
        win_rate = safe_get(summary, "win_rate")  # fraction (e.g. 0.5511)
        pf = safe_get(summary, "profit_factor")
        avg_r = safe_get(summary, "average_R")
        max_dd = safe_get(summary, "max_drawdown_R")

    row = {
    "spread_to_atr_cap": cap,
    "max_spread_points": max_spread_points,
    "trades": trades_count,
    "win_rate": win_rate if pd.notna(win_rate) else np.nan,  # ← remove *100
    "pf": pf,
    "avg_r": avg_r,
    "max_dd": max_dd,
    }
    rows.append(row)

    out = pd.DataFrame(rows, columns=[
        "spread_to_atr_cap", "max_spread_points", "trades",
        "win_rate", "pf", "avg_r", "max_dd"
    ])

    print("\n=== SPREAD_TO_ATR_CAP sweep results ===")
    print(out.to_string(index=False,
                        formatters={
                            "win_rate": (lambda v: f"{v:5.2f}%" if pd.notna(v) else "NaN"),
                            "pf": (lambda v: f"{v:.2f}" if pd.notna(v) else "NaN"),
                            "avg_r": (lambda v: f"{v:.2f}" if pd.notna(v) else "NaN"),
                            "max_dd": (lambda v: f"{v:.2f}" if pd.notna(v) else "NaN"),
                        }))

    os.makedirs("results", exist_ok=True)
    out.to_csv("results/spread_cap_sweep.csv", index=False)
    print("\nSaved sweep table to: results/spread_cap_sweep.csv")

if __name__ == "__main__":
    main()
