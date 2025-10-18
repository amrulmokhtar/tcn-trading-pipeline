#!/usr/bin/env python3
"""
Phase 6 Step 3, Pure ML Backtest with threshold sweep, costs, auto thresholds,
and session tagging.

Inputs
  - results/phase6_predictions.csv
  - data/m15_features.parquet

Outputs
  - results/phase6_ml_trades.csv
  - results/phase6_ml_summary.json
  - results/phase6_ml_session_summary.csv
"""

import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

# ------------ config ------------
PRED_PATHS: List[Path] = [
    Path("results/phase6_predictions.csv"),
    Path("phase6_predictions.csv"),
]
FEAT_PATHS: List[Path] = [
    Path("data/m15_features.parquet"),
    Path("m15_features.parquet"),
]
OUT_TRADES   = Path("results/phase6_ml_trades.csv")
OUT_SUMMARY  = Path("results/phase6_ml_summary.json")
OUT_SESSION  = Path("results/phase6_ml_session_summary.csv")
OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)

SYMBOL = "XAUUSD"
TZ = "Asia/Kuala_Lumpur"

# Full period
DATE_START: Optional[pd.Timestamp] = None
DATE_END:   Optional[pd.Timestamp] = None

# Horizon and clipping
FUTURE_BARS_HORIZON = 20
R_CLIP = 3.0

# Costs
SPREAD_PIPS = 3.0
COMMISSION_USD_ROUND_TRIP = 3.0
LOT_SIZE_OZ = 100.0
PIP_MODE = "icm"                 # or "ftmo_nonfx_2024"
pip_size_price = 0.01 if PIP_MODE == "icm" else 1.0

# Column hints
COL = {"ts": "ts", "m15_close": "m15_close", "m15_atr14": "m15_atr14"}

# Session columns if present in features
SESSION_COLS_CANDIDATES = ["sess_asia", "sess_london", "sess_ny"]

# ------------ helpers ------------
def _find_first(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")

def _load_predictions() -> pd.DataFrame:
    p = _find_first(PRED_PATHS)
    df = pd.read_csv(p, parse_dates=["timestamp_utc", "timestamp_myt"])
    return df

def _load_features() -> pd.DataFrame:
    p = _find_first(FEAT_PATHS)
    df = pd.read_parquet(p)  # requires pyarrow or fastparquet
    if COL["ts"] in df.columns:
        ts = pd.to_datetime(df[COL["ts"]], errors="coerce", utc=True)
        if ts.isna().mean() > 0.5:
            ts = pd.date_range("2020-01-01", periods=len(df), freq="15min", tz="UTC")
    else:
        ts = pd.date_range("2020-01-01", periods=len(df), freq="15min", tz="UTC")

    keep = [COL["m15_close"], COL["m15_atr14"]]
    for c in keep:
        if c not in df.columns:
            df[c] = 1e-6 if c == COL["m15_atr14"] else np.nan

    out = pd.DataFrame({"timestamp_utc": ts})
    out[keep] = df[keep].reset_index(drop=True)

    # carry session flags if present
    for s in SESSION_COLS_CANDIDATES:
        if s in df.columns:
            out[s] = df[s].astype(int).reset_index(drop=True)

    return out

def _auto_thresholds(series: pd.Series) -> List[float]:
    vals = series.dropna().values
    if len(vals) == 0:
        return [0.50, 0.60, 0.70, 0.80]
    qs = np.quantile(vals, [0.50, 0.60, 0.70, 0.80])
    th = sorted(set([float(x) for x in qs]))
    return th

# ------------ main ------------
def main():
    pred = _load_predictions()
    feat = _load_features()

    # unify timezone type
    if "timestamp_utc" in pred.columns:
        pred["timestamp_utc"] = pd.to_datetime(pred["timestamp_utc"], utc=True)
    if "timestamp_utc" in feat.columns:
        feat["timestamp_utc"] = pd.to_datetime(feat["timestamp_utc"], utc=True)

    # merge and timestamp
    df = pred.merge(feat, on="timestamp_utc", how="left")
    df["timestamp_myt"] = pd.to_datetime(df["timestamp_myt"], utc=True).dt.tz_convert(TZ)

    # derive session flags from local time if missing
    for c in ["sess_asia", "sess_london", "sess_ny"]:
        if c not in df.columns:
            df[c] = 0
    hrs = df["timestamp_myt"].dt.hour
    # Asia 7 to 15
    df.loc[(hrs >= 7) & (hrs < 15), "sess_asia"] = 1
    # London 15 to 23
    df.loc[(hrs >= 15) & (hrs < 23), "sess_london"] = 1
    # New York 20 to 5
    df.loc[(hrs >= 20) | (hrs < 5), "sess_ny"] = 1

    # optional date filter
    if DATE_START is not None and DATE_END is not None:
        mask = (df["timestamp_myt"] >= DATE_START) & (df["timestamp_myt"] <= DATE_END)
        df = df.loc[mask].reset_index(drop=True)

    # numeric casting
    df[COL["m15_close"]] = pd.to_numeric(df[COL["m15_close"]], errors="coerce")
    df[COL["m15_atr14"]] = pd.to_numeric(df[COL["m15_atr14"]], errors="coerce")
    df = df.dropna(subset=[COL["m15_close"], COL["m15_atr14"]]).reset_index(drop=True)

    # entry detection
    side_map = {"BUY": 1, "SELL": -1, "FLAT": 0}
    df["side"] = df["action_gated"].map(side_map).fillna(0).astype(int)
    entries_mask = (df["side"] != 0) & (df["side"].shift(1).fillna(0) == 0)

    # price and ATR
    close = df[COL["m15_close"]].astype(float)
    atr = df[COL["m15_atr14"]].astype(float).clip(lower=1e-6)

    # future price and valid entries
    future_close = close.shift(-FUTURE_BARS_HORIZON)
    valid_future = future_close.notna()
    entries_mask = entries_mask & valid_future

    # raw R then clip
    move = future_close - close
    R_unclipped = (move / atr)
    R = R_unclipped.clip(lower=-R_CLIP, upper=R_CLIP)

    # R at entries and side
    R_entry = R.where(entries_mask).dropna()
    side_entry = df.loc[R_entry.index, "side"].astype(int)
    R_entry = R_entry * side_entry.values

    # costs to R
    spread_price = SPREAD_PIPS * pip_size_price
    commission_price = COMMISSION_USD_ROUND_TRIP / LOT_SIZE_OZ
    total_price_cost = spread_price + commission_price
    atr_at_entry = atr.loc[R_entry.index].astype(float).clip(lower=1e-6)
    R_cost = (total_price_cost / atr_at_entry).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # session data at entries
    sess_cols = [c for c in SESSION_COLS_CANDIDATES if c in df.columns]
    sess_lab = np.where(df.loc[R_entry.index, "sess_asia"].values == 1, "asia",
                 np.where(df.loc[R_entry.index, "sess_london"].values == 1, "london",
                 np.where(df.loc[R_entry.index, "sess_ny"].values == 1, "ny", "other")))

    # auto thresholds from entry bar probabilities
    auto_thresholds = _auto_thresholds(df.loc[R_entry.index, "max_prob"])
    THRESHOLDS = auto_thresholds

    # quick scan to suggest a single best threshold
    def evaluate_threshold(th: float):
        idx_all = R_entry.index[df.loc[R_entry.index, "max_prob"].values >= th]
        if len(idx_all) == 0:
            return {"th": float(th), "trades": 0, "pf": 0.0, "ev": 0.0}
        rnet = (R_entry.loc[idx_all] - R_cost.loc[idx_all])
        pos = rnet[rnet > 0].sum()
        neg = -rnet[rnet < 0].sum()
        pf = float(pos / neg) if neg > 0 else 0.0
        wr = float((rnet > 0).mean())
        aw = float(rnet[rnet > 0].mean()) if (rnet > 0).any() else 0.0
        al = float(rnet[rnet < 0].mean()) if (rnet < 0).any() else 0.0
        ev = wr * aw + (1 - wr) * al
        return {"th": float(th), "trades": int(len(rnet)), "pf": pf, "ev": ev}

    scores = [evaluate_threshold(t) for t in THRESHOLDS]
    best_by_pf = max(scores, key=lambda x: x["pf"]) if scores else None
    best_by_ev = max(scores, key=lambda x: x["ev"]) if scores else None

    # sweep thresholds
    records = []
    all_trades = []

    for th in THRESHOLDS:
        entry_idx = R_entry.index
        keep = df.loc[entry_idx, "max_prob"] >= th
        idx = entry_idx[keep.values]

        if len(idx) == 0:
            records.append({
                "threshold": float(th),
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_R": 0.0,
                "max_drawdown_R": 0.0,
                "expected_value_R": 0.0,
                "spread_pips": SPREAD_PIPS,
                "commission_usd_round_trip": COMMISSION_USD_ROUND_TRIP,
                "pip_mode": PIP_MODE,
                "pip_size_price": pip_size_price,
                "lot_size_oz": LOT_SIZE_OZ,
                "horizon_bars": FUTURE_BARS_HORIZON,
            })
            continue

        tr = pd.DataFrame({
            "timestamp_myt": df.loc[idx, "timestamp_myt"].values,
            "timestamp_utc": df.loc[idx, "timestamp_utc"].values,
            "symbol": SYMBOL,
            "threshold": float(th),
            "side": df.loc[idx, "side"].astype(int).values,
            "R_raw": R_entry.loc[idx].values,
            "R_cost": R_cost.loc[idx].values,
        })
        for c in sess_cols:
            tr[c] = df.loc[idx, c].astype(int).values
        tr["session"] = sess_lab[keep.values]

        tr = tr.dropna(subset=["R_raw"]).reset_index(drop=True)
        tr["R_net"] = tr["R_raw"] - tr["R_cost"]

        # metrics
        n = len(tr)
        if n > 0:
            pos = tr.loc[tr["R_net"] > 0, "R_net"].sum()
            neg = -tr.loc[tr["R_net"] < 0, "R_net"].sum()
            pf  = float(pos / neg) if neg > 0 else 0.0
            win_rate = float((tr["R_net"] > 0).mean())
            avg_R = float(tr["R_net"].mean())
            eq = tr["R_net"].cumsum()
            dd = eq.cummax() - eq
            max_dd_R = float(dd.max()) if len(dd) > 0 else 0.0
            aw  = float(tr.loc[tr["R_net"] > 0, "R_net"].mean()) if (tr["R_net"] > 0).any() else 0.0
            al  = float(tr.loc[tr["R_net"] < 0, "R_net"].mean()) if (tr["R_net"] < 0).any() else 0.0
            EV_R = win_rate * aw + (1.0 - win_rate) * al
        else:
            pf = 0.0
            win_rate = 0.0
            avg_R = 0.0
            max_dd_R = 0.0
            EV_R = 0.0

        records.append({
            "threshold": float(th),
            "total_trades": int(n),
            "win_rate": win_rate,
            "profit_factor": pf,
            "average_R": avg_R,
            "max_drawdown_R": max_dd_R,
            "expected_value_R": EV_R,
            "spread_pips": SPREAD_PIPS,
            "commission_usd_round_trip": COMMISSION_USD_ROUND_TRIP,
            "pip_mode": PIP_MODE,
            "pip_size_price": pip_size_price,
            "lot_size_oz": LOT_SIZE_OZ,
            "horizon_bars": FUTURE_BARS_HORIZON,
        })

        all_trades.append(tr)

    # write trades
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(
        columns=["timestamp_myt","timestamp_utc","symbol","threshold","side","R_raw","R_cost","R_net","session"] + sess_cols
    )
    trades_df.to_csv(OUT_TRADES, index=False)

    # write threshold summary
    with open(OUT_SUMMARY, "w") as f:
        json.dump(records, f, indent=2)

    # write session summary
    if not trades_df.empty and "session" in trades_df.columns:
        sess_summary = (
            trades_df.groupby(["threshold","session"])["R_net"]
            .agg(count="count", mean="mean", sum="sum")
            .reset_index()
            .sort_values(["threshold","session"])
        )
        sess_summary.to_csv(OUT_SESSION, index=False)

    # console prints
    print(f"Auto thresholds: {', '.join([f'{t:.4f}' for t in THRESHOLDS])}")
    if best_by_pf:
        print(f"Best by PF  threshold={best_by_pf['th']:.4f}, trades={best_by_pf['trades']}, PF={best_by_pf['pf']:.4f}, EV={best_by_pf['ev']:.5f}")
    if best_by_ev:
        print(f"Best by EV  threshold={best_by_ev['th']:.4f}, trades={best_by_ev['trades']}, PF={best_by_ev['pf']:.4f}, EV={best_by_ev['ev']:.5f}")
    print(f"Saved trades to {OUT_TRADES}")
    print(f"Saved summary to {OUT_SUMMARY}")
    if OUT_SESSION.exists():
        print(f"Saved session summary to {OUT_SESSION}")
    print("Done.")

if __name__ == "__main__":
    main()
