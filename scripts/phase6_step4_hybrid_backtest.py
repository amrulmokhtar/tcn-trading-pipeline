#!/usr/bin/env python3
"""
Phase 6 Step 4, Hybrid Backtest
Merge rule-based entries with ML predictions into a single hybrid system
using the same cost model and full-period data.

Inputs
  - results/phase6_predictions.csv             , from phase6_hybrid_backtest.py prediction scaffold
  - <RULES_TRADES_CSV>                         , rule engine entries, flexible format, see notes
  - data/m15_features.parquet                  , price and ATR columns

Outputs
  - results/phase6_hybrid_trades.csv
  - results/phase6_hybrid_summary.json
  - results/phase6_hybrid_session_summary.csv

Notes on RULES_TRADES_CSV expected columns, flexible parsing supported:
  Preferred: timestamp_utc or timestamp_myt, and side in {1,-1}, or action in {"BUY","SELL"}
  Optional: symbol
  If timestamp_myt is provided, the script converts it to UTC with Asia/Kuala_Lumpur offset.
"""

import json
from typing import Optional, List
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# ------------- config defaults -------------
SYMBOL = "XAUUSD"
TZ = "Asia/Kuala_Lumpur"

# Horizon and clipping
FUTURE_BARS_HORIZON = 20
R_CLIP = 3.0

# Costs
SPREAD_PIPS = 3.0
COMMISSION_USD_ROUND_TRIP = 3.0
LOT_SIZE_OZ = 100.0
PIP_MODE = "icm"                 # or "ftmo_nonfx_2024"
pip_size_price = 0.01 if PIP_MODE == "icm" else 1.0

# Feature column hints
COL = {"ts": "ts", "m15_close": "m15_close", "m15_atr14": "m15_atr14"}

# Candidate session flags, will auto-create if missing
SESSION_COLS_CANDIDATES = ["sess_asia", "sess_london", "sess_ny"]

# ------------- helpers -------------
def _find_first(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")

def _load_features(feat_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(feat_path)
    # timestamp synthesis if missing or mostly NaT
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

def _load_predictions(pred_path: Path, ml_threshold: float) -> pd.DataFrame:
    df = pd.read_csv(pred_path, parse_dates=["timestamp_utc", "timestamp_myt"])
    # derive ML side from raw class with threshold gate
    side_map = {"BUY": 1, "SELL": -1, "FLAT": 0}
    # prefer action_raw with threshold applied now
    if "action_raw" in df.columns and "max_prob" in df.columns:
        gated = np.where(df["max_prob"].astype(float) >= float(ml_threshold), df["action_raw"], "FLAT")
        df["ml_side"] = pd.Series(gated).map(side_map).fillna(0).astype(int)
    elif "action_gated" in df.columns:
        df["ml_side"] = df["action_gated"].map(side_map).fillna(0).astype(int)
    else:
        raise ValueError("Predictions file must contain either action_raw with max_prob, or action_gated.")

    return df[["timestamp_utc", "timestamp_myt", "ml_side"]].copy()

def _load_rules_trades(rules_path: Path) -> pd.DataFrame:
    df = pd.read_csv(rules_path)
    # parse timestamps
    ts_utc = None
    if "timestamp_utc" in df.columns:
        ts_utc = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    elif "timestamp_myt" in df.columns:
        ts_myt = pd.to_datetime(df["timestamp_myt"], errors="coerce")
        # localize to MYT if naive, then convert to UTC
        if getattr(ts_myt.dt, "tz", None) is None:
            ts_myt = ts_myt.dt.tz_localize(TZ)
        ts_utc = ts_myt.dt.tz_convert("UTC")
    else:
        raise ValueError("Rules trades must have timestamp_utc or timestamp_myt.")

    # parse side
    side = None
    if "side" in df.columns:
        side = pd.to_numeric(df["side"], errors="coerce").fillna(0).astype(int)
        side = side.clip(-1, 1)
    elif "action" in df.columns:
        side = df["action"].map({"BUY": 1, "SELL": -1, "FLAT": 0}).fillna(0).astype(int)
    else:
        # try various common columns
        for col in df.columns:
            if str(col).lower() in ("signal", "direction", "rule_side"):
                tmp = df[col]
                if tmp.dtype.kind in ("i", "u"):
                    side = pd.to_numeric(tmp, errors="coerce").fillna(0).astype(int).clip(-1, 1)
                else:
                    side = tmp.map({"BUY": 1, "SELL": -1, "FLAT": 0}).fillna(0).astype(int)
                break
        if side is None:
            raise ValueError("Could not detect side column for rules trades. Provide 'side' or 'action'.")

    out = pd.DataFrame({"timestamp_utc": ts_utc, "rules_side": side})
    out = out.dropna(subset=["timestamp_utc"]).reset_index(drop=True)
    return out

def _derive_sessions(df_ts_myt: pd.Series) -> pd.DataFrame:
    hrs = df_ts_myt.dt.hour
    sess_asia = ((hrs >= 7) & (hrs < 15)).astype(int)
    sess_london = ((hrs >= 15) & (hrs < 23)).astype(int)
    sess_ny = ((hrs >= 20) | (hrs < 5)).astype(int)
    return pd.DataFrame({"sess_asia": sess_asia, "sess_london": sess_london, "sess_ny": sess_ny})

def _cost_to_R(atr_entry: pd.Series) -> pd.Series:
    spread_price = SPREAD_PIPS * pip_size_price
    commission_price = COMMISSION_USD_ROUND_TRIP / LOT_SIZE_OZ
    total_price_cost = spread_price + commission_price
    atr_e = atr_entry.astype(float).clip(lower=1e-6)
    return (total_price_cost / atr_e).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def _hybrid_side(row, mode: str) -> int:
    rs = int(row.get("rules_side", 0))
    ms = int(row.get("ml_side", 0))
    if mode == "both":                  # trade only when both agree and same direction
        return rs if (rs != 0 and rs == ms) else 0
    if mode == "either":                # trade when either produces a signal, prefer agreement sign when both fire
        return rs if rs != 0 else ms
    if mode == "rules_gate_ml":         # ML must fire, direction from ML, but rules must be nonzero gate
        return ms if (ms != 0 and rs != 0) else 0
    if mode == "ml_gate_rules":         # Rules must fire, direction from rules, but ML must be nonzero gate
        return rs if (rs != 0 and ms != 0) else 0
    # default
    return 0

def run_hybrid(
    feat_path: Path,
    pred_path: Path,
    rules_path: Path,
    combine_mode: str = "ml_gate_rules",
    ml_threshold: float = 0.60,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    out_dir: Path = Path("results")
):
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = _load_features(feat_path)
    pred = _load_predictions(pred_path, ml_threshold=ml_threshold)
    rules = _load_rules_trades(rules_path)

    # merge
    df = feat.merge(pred, on="timestamp_utc", how="left").merge(rules, on="timestamp_utc", how="left")
    # fill missing sides with 0
    df["ml_side"] = df["ml_side"].fillna(0).astype(int)
    df["rules_side"] = df["rules_side"].fillna(0).astype(int)

    # add MYT if missing
    if "timestamp_myt" not in df.columns:
        df["timestamp_myt"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert(TZ)

    # sessions
    for c in ["sess_asia", "sess_london", "sess_ny"]:
        if c not in df.columns:
            sess = _derive_sessions(pd.to_datetime(df["timestamp_myt"], utc=True))
            df[["sess_asia","sess_london","sess_ny"]] = sess.values
            break

    # optional date filters on MYT
    if date_start:
        ds = pd.Timestamp(date_start, tz=TZ)
        df = df[df["timestamp_myt"] >= ds]
    if date_end:
        de = pd.Timestamp(date_end, tz=TZ)
        df = df[df["timestamp_myt"] <= de]

    # compute hybrid side
    df["hybrid_side"] = df.apply(lambda r: _hybrid_side(r, combine_mode), axis=1).astype(int)

    # detect entries
    entries_mask = (df["hybrid_side"] != 0) & (df["hybrid_side"].shift(1).fillna(0) == 0)

    # price and ATR
    close = pd.to_numeric(df[COL["m15_close"]], errors="coerce").astype(float)
    atr = pd.to_numeric(df[COL["m15_atr14"]], errors="coerce").astype(float).clip(lower=1e-6)

    # future horizon
    future_close = close.shift(-FUTURE_BARS_HORIZON)
    valid_future = future_close.notna()
    entries_mask = entries_mask & valid_future

    # raw R then clip
    move = future_close - close
    R_unclipped = (move / atr)
    R = R_unclipped.clip(lower=-R_CLIP, upper=R_CLIP)

    # entries
    idx = df.index[entries_mask]
    if len(idx) == 0:
        trades_df = pd.DataFrame(columns=[
            "timestamp_myt","timestamp_utc","symbol","combine_mode","ml_threshold",
            "side","R_raw","R_cost","R_net","sess_asia","sess_london","sess_ny","session"
        ])
    else:
        R_entry = R.loc[idx].astype(float)
        side_entry = df.loc[idx, "hybrid_side"].astype(int)
        R_entry = R_entry * side_entry.values

        R_cost = _cost_to_R(atr.loc[idx])
        sess_cols = ["sess_asia","sess_london","sess_ny"]
        session_label = np.where(df.loc[idx,"sess_asia"].values == 1, "asia",
                           np.where(df.loc[idx,"sess_london"].values == 1, "london",
                           np.where(df.loc[idx,"sess_ny"].values == 1, "ny", "other")))

        trades_df = pd.DataFrame({
            "timestamp_myt": df.loc[idx, "timestamp_myt"].values,
            "timestamp_utc": df.loc[idx, "timestamp_utc"].values,
            "symbol": SYMBOL,
            "combine_mode": combine_mode,
            "ml_threshold": float(ml_threshold),
            "side": side_entry.values,
            "R_raw": R_entry.values,
            "R_cost": R_cost.values,
        })
        for c in sess_cols:
            trades_df[c] = df.loc[idx, c].astype(int).values
        trades_df["session"] = session_label
        trades_df["R_net"] = trades_df["R_raw"] - trades_df["R_cost"]
        trades_df = trades_df.reset_index(drop=True)

    # metrics
    rec = {
        "combine_mode": combine_mode,
        "ml_threshold": float(ml_threshold),
        "total_trades": int(len(trades_df)),
        "win_rate": float((trades_df["R_net"] > 0).mean()) if len(trades_df) > 0 else 0.0,
        "profit_factor": 0.0,
        "average_R": float(trades_df["R_net"].mean()) if len(trades_df) > 0 else 0.0,
        "max_drawdown_R": 0.0,
        "expected_value_R": 0.0,
        "spread_pips": SPREAD_PIPS,
        "commission_usd_round_trip": COMMISSION_USD_ROUND_TRIP,
        "pip_mode": PIP_MODE,
        "pip_size_price": pip_size_price,
        "lot_size_oz": LOT_SIZE_OZ,
        "horizon_bars": FUTURE_BARS_HORIZON,
    }
    if len(trades_df) > 0:
        pos = trades_df.loc[trades_df["R_net"] > 0, "R_net"].sum()
        neg = -trades_df.loc[trades_df["R_net"] < 0, "R_net"].sum()
        rec["profit_factor"] = float(pos / neg) if neg > 0 else 0.0
        eq = trades_df["R_net"].cumsum()
        dd = eq.cummax() - eq
        rec["max_drawdown_R"] = float(dd.max()) if len(dd) > 0 else 0.0
        aw = float(trades_df.loc[trades_df["R_net"] > 0, "R_net"].mean()) if (trades_df["R_net"] > 0).any() else 0.0
        al = float(trades_df.loc[trades_df["R_net"] < 0, "R_net"].mean()) if (trades_df["R_net"] < 0).any() else 0.0
        rec["expected_value_R"] = rec["win_rate"] * aw + (1.0 - rec["win_rate"]) * al

    # session summary
    sess_summary = pd.DataFrame()
    if len(trades_df) > 0 and "session" in trades_df.columns:
        sess_summary = (
            trades_df.groupby(["combine_mode","ml_threshold","session"])["R_net"]
            .agg(count="count", mean="mean", sum="sum")
            .reset_index()
            .sort_values(["combine_mode","ml_threshold","session"])
        )

    # write outputs
    out_trades = out_dir / "phase6_hybrid_trades.csv"
    out_summary = out_dir / "phase6_hybrid_summary.json"
    out_session = out_dir / "phase6_hybrid_session_summary.csv"

    trades_df.to_csv(out_trades, index=False)
    with open(out_summary, "w") as f:
        json.dump([rec], f, indent=2)
    if not sess_summary.empty:
        sess_summary.to_csv(out_session, index=False)

    # console
    print(f"Saved trades to {out_trades}")
    print(f"Saved summary to {out_summary}")
    if out_session.exists():
        print(f"Saved session summary to {out_session}")
    print("Done.")

def parse_args():
    ap = argparse.ArgumentParser(description="Phase 6 Step 4, Hybrid Backtest")
    ap.add_argument("--features", type=str, default="data/m15_features.parquet", help="Path to m15_features.parquet")
    ap.add_argument("--predictions", type=str, default="results/phase6_predictions.csv", help="Path to ML predictions CSV")
    ap.add_argument("--rules_trades", type=str, required=True, help="Path to rules trades CSV")
    ap.add_argument("--combine_mode", type=str, default="ml_gate_rules",
                    choices=["both","either","rules_gate_ml","ml_gate_rules"],
                    help="How to combine rules and ML signals")
    ap.add_argument("--ml_threshold", type=float, default=0.60, help="Min probability to treat ML as a signal")
    ap.add_argument("--date_start", type=str, default=None, help="Filter start in MYT, like 2021-01-01")
    ap.add_argument("--date_end", type=str, default=None, help="Filter end in MYT, like 2025-10-18")
    ap.add_argument("--out_dir", type=str, default="results", help="Output directory")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_hybrid(
        feat_path=Path(args.features),
        pred_path=Path(args.predictions),
        rules_path=Path(args.rules_trades),
        combine_mode=args.combine_mode,
        ml_threshold=args.ml_threshold,
        date_start=args.date_start,
        date_end=args.date_end,
        out_dir=Path(args.out_dir)
    )
