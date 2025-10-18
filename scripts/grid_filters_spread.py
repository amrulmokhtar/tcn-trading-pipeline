# scripts/grid_filters_spread.py
from pathlib import Path
import sys
import itertools as it
import numpy as np
import pandas as pd

# import project root
sys.path.append(str(Path(__file__).resolve().parents[1]))
import phase5_backtest as pb  # exposes FEAT_PATH and run_backtest

# ================== sweep knobs ==================
SPREAD_TO_ATR_CAP_LIST = [5, 6, 7, 8, 9, 10, 12, 15, 20, 9999]
MAX_SPREAD_POINTS_LIST = [20, 25, 30, 40, 50]

TOGGLE_KEYS = ["USE_SESSION", "USE_REGIME_FILTER", "USE_RANGE_SKIP", "USE_ADX_FILTER"]

MAX_DD_LIMIT_R   = 30.0
MIN_TPM_LIMIT    = 5.0
MIN_TRADES_TOTAL = 800

TOP_N_PRINT = 20

def score_row(pf, avg_r, tpm):
    if pf <= 0 or avg_r <= 0 or tpm <= 0:
        return 0.0
    return float(pf * avg_r * np.sqrt(tpm))

def safe_get(d, k, default=0.0):
    try:
        v = d.get(k, default)
        return float(v) if v is not None else default
    except Exception:
        return default

def main():
    out_rows = []

    df = pd.read_parquet(pb.FEAT_PATH)
    bars = len(df)
    total_days = bars / 96.0
    months_in_sample = max(total_days / 30.0, 1e-9)

    cols = set(df.columns)
    has_obv = any(k in cols for k in ("h1_obv", "h1_obv_sma50", "h1_obv_slope_6"))
    if not has_obv:
        print("! OBV columns not found in features, OBV toggles will be fixed to False.")

    base_toggle_space = list(it.product([False, True], repeat=len(TOGGLE_KEYS)))
    obv_toggle_space  = [(False, False)] if not has_obv else list(it.product([False, True], [False, True]))

    combos = list(it.product(
        SPREAD_TO_ATR_CAP_LIST,
        MAX_SPREAD_POINTS_LIST,
        base_toggle_space,
        obv_toggle_space
    ))

    print(f"Sweeping {len(combos)} combinations...")
    for cap, max_pts, toggles, (use_obv_pct, use_obv_confirm) in combos:
        flags = dict(zip(TOGGLE_KEYS, toggles))
        flags["USE_OBV_PCT"]     = bool(use_obv_pct)
        flags["USE_OBV_CONFIRM"] = bool(use_obv_confirm)
        flags["SPREAD_TO_ATR_CAP"] = cap
        flags["MAX_SPREAD_POINTS"] = max_pts

        trades, summary = pb.run_backtest(df, **flags)
        n_trades = 0 if trades is None else len(trades)

        win_rate = safe_get(summary, "win_rate", 0.0) * 100.0
        pf       = safe_get(summary, "profit_factor", 0.0)
        avg_r    = safe_get(summary, "average_R", 0.0)
        max_dd   = safe_get(summary, "max_drawdown_R", 0.0)
        ev_r     = safe_get(summary, "expected_value_R", 0.0)  # pull EV

        tpm = n_trades / months_in_sample
        scr = score_row(pf, avg_r, tpm)

        row = {
            "spread_to_atr_cap": cap,
            "max_spread_points": max_pts,
            "USE_SESSION": flags["USE_SESSION"],
            "USE_REGIME_FILTER": flags["USE_REGIME_FILTER"],
            "USE_RANGE_SKIP": flags["USE_RANGE_SKIP"],
            "USE_ADX_FILTER": flags["USE_ADX_FILTER"],
            "USE_OBV_PCT": flags["USE_OBV_PCT"],
            "USE_OBV_CONFIRM": flags["USE_OBV_CONFIRM"],
            "trades": n_trades,
            "tpm": tpm,
            "win_rate_pct": win_rate,
            "pf": pf,
            "avg_r": avg_r,
            "max_dd_R": max_dd,
            "expected_value_R": ev_r,   # include EV in grid
            "score": scr,
        }
        out_rows.append(row)

    out = pd.DataFrame(out_rows)

    keep = (
        (out["max_dd_R"] <= MAX_DD_LIMIT_R) &
        (out["tpm"]      >= MIN_TPM_LIMIT) &
        (out["trades"]   >= MIN_TRADES_TOTAL)
    )
    filt = out[keep].copy().sort_values(["score", "pf", "tpm"], ascending=[False, False, False])

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    filt_path = out_dir / "filters_spread_grid.csv"
    raw_path  = out_dir / "filters_spread_grid_raw.csv"
    filt.to_csv(filt_path, index=False)
    out.to_csv(raw_path, index=False)

    print(f"\nSaved filtered grid to: {filt_path}")
    print(f"Saved raw grid to      : {raw_path}\n")

    cols_show = [
        "spread_to_atr_cap", "max_spread_points",
        "USE_SESSION", "USE_REGIME_FILTER", "USE_RANGE_SKIP", "USE_ADX_FILTER",
        "USE_OBV_PCT", "USE_OBV_CONFIRM",
        "trades", "tpm", "win_rate_pct", "pf", "avg_r", "max_dd_R",
        "expected_value_R",  # show EV in console
        "score"
    ]
    cols_show = [c for c in cols_show if c in filt.columns]

    print("Top candidates:")
    print(filt[cols_show].head(TOP_N_PRINT).to_string(index=False))

if __name__ == "__main__":
    main()
