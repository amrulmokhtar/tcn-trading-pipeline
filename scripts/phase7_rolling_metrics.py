# scripts/phase7_rolling_metrics.py
import pandas as pd
import numpy as np
from pathlib import Path

TS_CANDIDATES = ["exit_ts", "timestamp_utc", "timestamp", "close_ts", "exit_time", "timestamp_myt"]

def pick_ts_col(df: pd.DataFrame) -> str:
    for c in TS_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No timestamp like column found, first columns: {list(df.columns)[:12]}")

def pick_net_r(df: pd.DataFrame) -> pd.Series:
    # preferred, already net of costs
    if "R_net" in df.columns:
        return df["R_net"].astype(float)

    # else, derive from raw minus cost if available
    if "R_raw" in df.columns and "R_cost" in df.columns:
        return (df["R_raw"].astype(float) - df["R_cost"].astype(float))

    # common alternates
    for c in ["R", "r", "pnl_R", "r_multiple", "ret_r", "rr", "R_multiple"]:
        if c in df.columns:
            return df[c].astype(float)

    # last resort, try pnl_usd divided by risk_usd
    pnl_candidates  = [c for c in df.columns if c.lower() in ("pnl_usd","pnl","pl_usd","pnl_risk_usd")]
    risk_candidates = [c for c in df.columns if c.lower() in ("risk_usd","risk","risk_per_trade_usd")]
    if pnl_candidates and risk_candidates:
        pnl = df[pnl_candidates[0]].astype(float)
        risk = df[risk_candidates[0]].astype(float).replace(0, np.nan)
        derived = pnl / risk
        if derived.notna().any():
            return derived

    raise ValueError(f"Could not detect per trade R, available columns: {list(df.columns)[:25]}")

def compute_rolling_metrics(trades_path: str, out_prefix: str, window: int = 200):
    df = pd.read_csv(trades_path)
    tscol = pick_ts_col(df)
    r = pick_net_r(df)

    df[tscol] = pd.to_datetime(df[tscol], errors="coerce", utc=True)
    df = df.sort_values(tscol).reset_index(drop=True)
    r = r.loc[df.index].astype(float)

    mean_r = r.rolling(window).mean()
    std_r = r.rolling(window).std(ddof=0)
    sharpe = mean_r / std_r

    downside = r.where(r < 0, 0.0)
    std_down = downside.rolling(window).std(ddof=0).replace(0, np.nan)
    sortino = mean_r / std_down

    out = pd.DataFrame({tscol: df[tscol], "rolling_sharpe": sharpe, "rolling_sortino": sortino})
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(f"{out_prefix}_rolling.csv", index=False)
    print(f"saved {out_prefix}_rolling.csv, window={window}, trades={len(df)}")

def main():
    scenarios = {
        "spread4_2": r"results\phase7_oos_spread4_2\phase6_hybrid_trades.csv",
        "spread3_6": r"results\phase7_oos_spread3_6\phase6_hybrid_trades.csv",
        "spread3_0": r"results\phase7_oos_spread3_0\phase6_hybrid_trades.csv",
    }
    for name, path in scenarios.items():
        compute_rolling_metrics(path, f"results\\phase7_{name}\\phase7", window=200)

if __name__ == "__main__":
    main()
