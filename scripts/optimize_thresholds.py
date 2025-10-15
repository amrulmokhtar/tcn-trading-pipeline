import pandas as pd
import numpy as np
from itertools import product

CSV = "predictions.csv"          # produced by run_sliding_inference.py
HORIZON = 6                      # must match run_sliding_inference.py
P_COL = f"pnl_h{HORIZON}"

df = pd.read_csv(CSV, parse_dates=["timestamp"])

# helper
def summarize(mask):
    sub = df.loc[mask].copy()
    if sub.empty:
        return 0, 0.0, 0.0, 0.0
    wins = sub[P_COL] > 0
    trades = len(sub)
    winrate = float(wins.mean())
    avg_pnl = float(sub[P_COL].mean())
    sum_pnl = float(sub[P_COL].sum())
    return trades, winrate, avg_pnl, sum_pnl

# search space
p_grid   = [0.50, 0.55, 0.60, 0.65, 0.70]
d_grid   = [0.05, 0.08, 0.10, 0.12, 0.15]

rows = []
for pth, dmin in product(p_grid, d_grid):
    long_mask  = (df["y_pred"] == 0) & (df["p_long"]  >= pth) & ((df["p_long"]  - df["p_short"]).abs() >= dmin)
    short_mask = (df["y_pred"] == 1) & (df["p_short"] >= pth) & ((df["p_short"] - df["p_long"]).abs()  >= dmin)
    take = long_mask | short_mask
    trades, winrate, avg_pnl, sum_pnl = summarize(take)

    rows.append({
        "p_thresh": pth,
        "delta_min": dmin,
        "trades": trades,
        "win_rate": round(winrate, 3),
        "avg_pnl": round(avg_pnl, 6),
        "sum_pnl": round(sum_pnl, 6)
    })

res = pd.DataFrame(rows).sort_values(["sum_pnl", "win_rate", "trades"], ascending=[False, False, False])
print(res.head(20).to_string(index=False))

# also print a lighter table sorted by win rate if you prefer precision
print("\nTop by win rate:")
print(res.sort_values(["win_rate","trades"], ascending=[False, False]).head(20).to_string(index=False))
