# scripts/sweep_grid.py
from __future__ import annotations
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Make "scripts" parent (repo root) importable, then import the backtest module
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.phase5_backtest as pb  # noqa: E402


def bars_to_months(nbars: int, bars_per_day: int = 96) -> float:
    """
    Convert number of 15-minute bars to months (≈30 days/mo).
    """
    return nbars / (bars_per_day * 30.0)


def safe_get(d: dict, key: str, default: float = np.nan):
    v = d.get(key, default) if isinstance(d, dict) else default
    return default if v is None else v


def main():
    # --- Load once ---
    feat_path = Path(pb.FEAT_PATH) if hasattr(pb, "FEAT_PATH") else Path("data/m15_features.parquet")
    df = pd.read_parquet(feat_path)
    months = bars_to_months(len(df))

    # --- Parameter grids (tweak freely) ---
    spread_to_atr_caps = [8, 10, 12, 15]
    max_spread_points_list = [30, 40, 50]
    adx_long_list = [22, 20, 18]
    adx_short_list = [30, 28, 25, 22]

    # Optional: keep short threshold >= long threshold
    combos = []
    for cap, msp, adx_l, adx_s in itertools.product(
        spread_to_atr_caps, max_spread_points_list, adx_long_list, adx_short_list
    ):
        if adx_s >= adx_l:  # mild sanity; comment out if you want all combos
            combos.append((cap, msp, adx_l, adx_s))

    rows = []

    print("\n=== GRID SWEEP ===\n")
    print("cap  msp  adxL  adxS | trades  win%   PF   avgR  maxDD(R)  t/mo   score")
    print("-" * 75)

    for cap, msp, adx_l, adx_s in combos:
        # Apply params to the backtest module (module-level constants)
        pb.SPREAD_TO_ATR_CAP = cap
        pb.MAX_SPREAD_POINTS = msp
        pb.ADX_LONG_MIN = adx_l
        pb.ADX_SHORT_MIN = adx_s

        # Run backtest: our pb.run_backtest(df) returns (trades, summary)
        trades, summary = pb.run_backtest(df)

        total_trades = safe_get(summary, "total_trades", 0)
        win_rate = safe_get(summary, "win_rate", np.nan)   # fraction
        pf = safe_get(summary, "profit_factor", np.nan)
        avg_r = safe_get(summary, "average_R", np.nan)
        max_dd_r = safe_get(summary, "max_drawdown_R", np.nan)

        trades_per_month = total_trades / months if months > 0 else np.nan

        # Scoring rule: only count if PF >= 1.20, otherwise NaN
        score = trades_per_month * avg_r if (not np.isnan(pf) and pf >= 1.20) else np.nan

        rows.append({
            "spread_to_atr_cap": cap,
            "max_spread_points": msp,
            "adx_long_min": adx_l,
            "adx_short_min": adx_s,
            "trades": total_trades,
            "win_rate_pct": win_rate * 100.0 if pd.notna(win_rate) else np.nan,
            "pf": pf,
            "avg_r": avg_r,
            "max_dd_r": max_dd_r,
            "trades_per_month": trades_per_month,
            "score_tpm_x_avgR_pf>=1.2": score,
        })

        print(
            f"{cap:>3}  {msp:>3}  {adx_l:>4}  {adx_s:>4} | "
            f"{int(total_trades):>6}  "
            f"{(win_rate*100.0):>5.2f}%  "
            f"{pf:>4.2f}  {avg_r:>5.2f}  {max_dd_r:>8.2f}  "
            f"{trades_per_month:>5.1f}  {score if pd.notna(score) else float('nan'):>6.3f}"
        )

    out = pd.DataFrame(rows)

    # Sort best by score (desc), then by PF desc, then trades/month desc
    out_sorted = out.sort_values(
        by=["score_tpm_x_avgR_pf>=1.2", "pf", "trades_per_month"],
        ascending=[False, False, False]
    )

    # Save to results/
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "grid_sweep.csv"
    out_sorted.to_csv(out_path, index=False)

    print("\nSaved sweep table to:", out_path)
    print("\nTop 10 by score:")
    print(out_sorted.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
