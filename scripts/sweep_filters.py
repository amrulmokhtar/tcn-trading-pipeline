import pandas as pd
import itertools
import sys
from pathlib import Path

# ensure root path ("tcn_1.0") is visible to Python
sys.path.append(str(Path(__file__).resolve().parents[1]))

import phase5_backtest as pb   # <-- notice: no "scripts."

FEAT_PATH = pb.FEAT_PATH
OUT_CSV   = Path("results/filter_sweep.csv")

DF = pd.read_parquet(FEAT_PATH)

TOGGLES = {
    "USE_SESSION":     [True, False],
    "USE_RANGE_SKIP":  [True, False],
    "USE_ADX_FILTER":  [True, False],
    # leave OBV off for now unless your features include pct/confirm columns
    "USE_OBV_PCT":     [False],
    "USE_OBV_CONFIRM": [False],
}

rows = []
for vals in itertools.product(*TOGGLES.values()):
    cfg = dict(zip(TOGGLES.keys(), vals))

    # apply toggles to the imported config (in-memory)
    pb.USE_SESSION     = cfg["USE_SESSION"]
    pb.USE_RANGE_SKIP  = cfg["USE_RANGE_SKIP"]
    pb.USE_ADX_FILTER  = cfg["USE_ADX_FILTER"]
    pb.USE_OBV_PCT     = cfg["USE_OBV_PCT"]
    pb.USE_OBV_CONFIRM = cfg["USE_OBV_CONFIRM"]

    trades, summary = pb.run_backtest(DF)

    rows.append({
        **cfg,
        "trades":       summary.get("total_trades", 0),
        "win_rate_pct": summary.get("win_rate", 0.0),             # already percent
        "pf":           summary.get("profit_factor", 0.0),
        "avg_R":        summary.get("average_R", 0.0),
        "max_dd_R":     summary.get("max_drawdown_R", 0.0),
        "tpm":          summary.get("trades_per_month", 0.0),
        "score_tpm_x_avgR_pf>=1.2":
            (summary.get("trades_per_month", 0.0) * summary.get("average_R", 0.0)
             if summary.get("profit_factor", 0.0) >= 1.2 else 0.0)
    })

out = pd.DataFrame(rows)
out.sort_values("score_tpm_x_avgR_pf>=1.2", ascending=False, inplace=True)
OUT_CSV.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(OUT_CSV, index=False)
print("\nSaved filter sweep to:", OUT_CSV)
print(out.head(10).to_string(index=False))
