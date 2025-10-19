# scripts/phase6_step4_hybrid_backtest.py
# Phase 6 Step 4, Hybrid Backtest with extended trade export
# Exports entry and exit prices to CSV for MT5 cross checks

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Optional, Tuple

# Defaults
SYMBOL = "XAUUSD"
COMBINE_DEFAULT = "ml_gate_rules"
ML_THRESHOLD_DEFAULT = 0.55
FUTURE_BARS_HORIZON = 20

# Costs, can be overridden via CLI in __main__
DEFAULT_SPREAD_PIPS = 3.0
DEFAULT_COMMISSION_USD_RT = 7.0
SPREAD_PIPS = DEFAULT_SPREAD_PIPS
COMMISSION_USD_ROUND_TRIP = DEFAULT_COMMISSION_USD_RT
LOT_SIZE_OZ = 100.0  # your standard

# Pip conversion for gold, 1 pip is 0.01 USD
PIP_VALUE_USD = 0.01

def parse_args():
    ap = argparse.ArgumentParser(description="Phase 6 Step 4, Hybrid Backtest")
    ap.add_argument("--features", type=str, default="data/m15_features.parquet", help="Path to m15_features.parquet")
    ap.add_argument("--predictions", type=str, default="results/phase6_predictions.csv", help="Path to ML predictions CSV")
    ap.add_argument("--rules_trades", type=str, required=True, help="Path to rules trades CSV")
    ap.add_argument("--combine_mode", type=str, default=COMBINE_DEFAULT,
                    choices=["both", "either", "rules_gate_ml", "ml_gate_rules"],
                    help="How to combine rules and ML signals")
    ap.add_argument("--ml_threshold", type=float, default=ML_THRESHOLD_DEFAULT, help="Min probability to treat ML as a signal")
    ap.add_argument("--date_start", type=str, default=None, help="Filter start in MYT, like 2024-01-01")
    ap.add_argument("--date_end", type=str, default=None, help="Filter end in MYT, like 2024-12-31")
    ap.add_argument("--out_dir", type=str, default="results", help="Output directory")

    # new cost args
    ap.add_argument("--spread_pips", type=float, default=DEFAULT_SPREAD_PIPS, help="Spread in pips, round trip")
    ap.add_argument("--commission_usd_rt", type=float, default=DEFAULT_COMMISSION_USD_RT, help="Commission USD, round trip")

    return ap.parse_args()

def _ensure_ts(df: pd.DataFrame) -> Tuple[str, str]:
    myt = "timestamp_myt" if "timestamp_myt" in df.columns else None
    utc = "timestamp_utc" if "timestamp_utc" in df.columns else None
    if utc is None:
        # synthesize if missing
        df["timestamp_utc"] = pd.date_range("2020-01-01", periods=len(df), freq="15min", tz="UTC")
        utc = "timestamp_utc"
    df[utc] = pd.to_datetime(df[utc], utc=True)
    if myt and not pd.api.types.is_datetime64_any_dtype(df[myt]):
        df[myt] = pd.to_datetime(df[myt])
    return myt or utc, utc

def _load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # require these columns for pricing and ATR
    needed = ["m15_close", "m15_atr14"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing feature columns: {miss}")
    _ensure_ts(df)
    return df

def _load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # expected columns: timestamp_utc, timestamp_myt optional, action_raw or action_gated, max_prob optional
    if "timestamp_utc" not in df.columns:
        raise ValueError("Predictions must have timestamp_utc")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    if "timestamp_myt" in df.columns:
        df["timestamp_myt"] = pd.to_datetime(df["timestamp_myt"])
    # derive ml_side if not present
    if "ml_side" not in df.columns:
        if "action_gated" in df.columns:
            df["ml_side"] = np.where(df["action_gated"].astype(str).str.upper() == "BUY", 1, 0)
        elif "action_raw" in df.columns and "max_prob" in df.columns:
            df["ml_side"] = np.where(df["action_raw"].astype(str).str.upper() == "BUY", 1, 0)
        else:
            raise ValueError("Predictions need ml_side or action_* with max_prob")
    return df[["timestamp_utc", "ml_side"]].copy()

def _load_rules_trades(csv_path: Path) -> pd.DataFrame:
    # expect rules trades CSV has timestamps and a side column
    df = pd.read_csv(csv_path)
    # accept a few common column names
    ts = None
    for c in ["timestamp_utc", "exit_ts", "timestamp", "time"]:
        if c in df.columns:
            ts = c
            break
    if ts is None:
        raise ValueError("rules_trades CSV must include a timestamp column")
    df["timestamp_utc"] = pd.to_datetime(df[ts], utc=True, errors="coerce")
    # rules side column
    side_col = None
    for c in ["side", "rule_side", "signal"]:
        if c in df.columns:
            side_col = c
            break
    if side_col is None:
        raise ValueError("rules_trades CSV must include a side column")
    # normalize side to 0 or 1 long only, your hybrid uses rules for direction filter
    rules = df[["timestamp_utc", side_col]].rename(columns={side_col: "rule_side"})
    rules["rule_side"] = np.where(rules["rule_side"] > 0, 1, 0)
    return rules

def _combine_signals(ml_side: pd.Series, rule_side: pd.Series, mode: str) -> pd.Series:
    if mode == "both":
        return ((ml_side == 1) & (rule_side == 1)).astype(int)
    if mode == "either":
        return ((ml_side == 1) | (rule_side == 1)).astype(int)
    if mode == "rules_gate_ml":
        # rules decide trade on or off, ml ignored if rules say 0
        return (rule_side == 1).astype(int)
    if mode == "ml_gate_rules":
        # ml decides whether to allow rules, your default
        return ((ml_side == 1) & (rule_side == 1)).astype(int)
    raise ValueError(f"Unknown combine mode {mode}")

def _cost_in_R(spread_pips: float, commission_usd_rt: float, atr_entry: pd.Series) -> pd.Series:
    # Convert spread in pips to USD, then to R using ATR in USD
    spread_usd = spread_pips * PIP_VALUE_USD
    # cost per trade in USD, round trip
    cost_usd = spread_usd + commission_usd_rt / LOT_SIZE_OZ
    # convert to R using ATR
    atr_usd = atr_entry.astype(float)
    # avoid division by zero
    atr_usd = atr_usd.replace(0.0, np.nan)
    cost_r = cost_usd / atr_usd
    return cost_r.fillna(0.0)

def run_hybrid(
    feat_path: Path,
    pred_path: Path,
    rules_path: Path,
    combine_mode: str,
    ml_threshold: float,
    date_start: Optional[str],
    date_end: Optional[str],
    out_dir: Path,
):
    # --- safe date filter in UTC only ---
    TZ_MYT = "Asia/Kuala_Lumpur"

    def _to_utc_bounds(start_str, end_str):
        start_utc = None
        end_utc = None
        if start_str is not None:
            start_utc = pd.Timestamp(start_str).tz_localize(TZ_MYT).tz_convert("UTC")
        if end_str is not None:
            # inclusive end, add one day then use < end_utc
            end_utc = (pd.Timestamp(end_str) + pd.Timedelta(days=1)).tz_localize(TZ_MYT).tz_convert("UTC")
        return start_utc, end_utc

    df = _load_features(feat_path)
    preds = _load_predictions(pred_path)
    rules = _load_rules_trades(rules_path)

    # merge on timestamp_utc
    data = df.merge(preds, on="timestamp_utc", how="left").merge(rules, on="timestamp_utc", how="left")

    # --- filter in UTC only, convert MYT bounds to UTC first ---
    TZ_MYT = "Asia/Kuala_Lumpur"

    def _to_utc_bounds(start_str, end_str):
        start_utc = None
        end_utc = None
        if start_str is not None:
            start_utc = pd.Timestamp(start_str).tz_localize(TZ_MYT).tz_convert("UTC")
        if end_str is not None:
            # inclusive end, add one day then use < end_utc
            end_utc = (pd.Timestamp(end_str) + pd.Timedelta(days=1)).tz_localize(TZ_MYT).tz_convert("UTC")
        return start_utc, end_utc

    start_utc, end_utc = _to_utc_bounds(date_start, date_end)

    # ensure timestamp_utc is tz-aware UTC
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True, errors="coerce")

    if start_utc is not None:
        data = data[data["timestamp_utc"] >= start_utc]
    if end_utc is not None:
        data = data[data["timestamp_utc"] < end_utc]
            
    # Inputs
    close = data["m15_close"].astype(float)
    atr = data["m15_atr14"].astype(float)

    # Future exit price per horizon
    future_close = close.shift(-FUTURE_BARS_HORIZON)

    # R raw from price move normalized by ATR, clipped to +-3R if that is your standard
    R_unclipped = (future_close - close) / atr
    R_entry = R_unclipped.clip(lower=-3.0, upper=3.0)

    # Combine signals
    ml_side = data["ml_side"].fillna(0).astype(int)
    rule_side = data["rule_side"].fillna(0).astype(int)
    side_entry = _combine_signals(ml_side, rule_side, combine_mode)

    # apply threshold if needed, current input already gated by ml_side at inference, so side_entry is final

    # cost in R per trade using ATR at entry
    R_cost = _cost_in_R(SPREAD_PIPS, COMMISSION_USD_ROUND_TRIP, atr)

    # index of valid trades where side is 1 and we have a future bar
    valid = side_entry.eq(1) & future_close.notna()

    idx = data.index[valid]
    # extended trade dataframe with prices
    trades_df = pd.DataFrame({
        "timestamp_myt": data.loc[idx, "timestamp_myt"].values if "timestamp_myt" in data.columns else pd.NaT,
        "timestamp_utc": data.loc[idx, "timestamp_utc"].values,
        "symbol": SYMBOL,
        "combine_mode": combine_mode,
        "ml_threshold": float(ml_threshold),
        "side": side_entry.loc[idx].values,                       # 1 for long
        "entry_price": close.loc[idx].values,                     # price at entry
        "exit_price": future_close.loc[idx].values,               # price at horizon exit
        "atr_entry": atr.loc[idx].values,                         # ATR at entry
        "R_raw": R_entry.loc[idx].values,                         # raw R before costs
        "R_cost": R_cost.loc[idx].values,                         # cost converted to R
    })
    trades_df["R_net"] = trades_df["R_raw"] - trades_df["R_cost"]
    trades_df["pips_diff"] = (trades_df["exit_price"] - trades_df["entry_price"]) / PIP_VALUE_USD
    
    # Add exit timestamps and duration
    exit_ts_utc_all = data["timestamp_utc"].shift(-FUTURE_BARS_HORIZON)
    exit_ts_myt_all = data["timestamp_myt"].shift(-FUTURE_BARS_HORIZON) if "timestamp_myt" in data.columns else None

    trades_df["exit_timestamp_utc"] = exit_ts_utc_all.loc[idx].values
    if exit_ts_myt_all is not None:
        trades_df["exit_timestamp_myt"] = exit_ts_myt_all.loc[idx].values

    # Duration in bars and minutes
    trades_df["duration_bars"] = FUTURE_BARS_HORIZON
    trades_df["duration_minutes"] = (
        (pd.to_datetime(trades_df["exit_timestamp_utc"]) - pd.to_datetime(trades_df["timestamp_utc"]))
        .dt.total_seconds() / 60.0
    )
    
    # Label direction (Buy/Sell)
    trades_df["direction"] = np.where(trades_df["side"] == 1, "BUY", "SELL")

    # Label win/loss based on R_net
    trades_df["result"] = np.where(trades_df["R_net"] > 0, "WIN", "LOSS")

    # session tagging if available
    for c in ["sess_asia", "sess_london", "sess_ny", "session"]:
        if c in data.columns:
            trades_df[c] = data.loc[idx, c].values

    # Save trades CSV
    trades_csv = out_dir / "phase6_hybrid_trades.csv"
    trades_df.to_csv(trades_csv, index=False)

    # Summary
    wins = (trades_df["R_net"] > 0).sum()
    losses = (trades_df["R_net"] <= 0).sum()
    gross_profit = trades_df.loc[trades_df["R_net"] > 0, "R_net"].sum()
    gross_loss = -trades_df.loc[trades_df["R_net"] <= 0, "R_net"].sum()
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf
    ev = float(trades_df["R_net"].mean()) if len(trades_df) else 0.0
    dd = float(_max_drawdown(trades_df["R_net"].to_numpy()))
    win_rate = float(wins / max(1, len(trades_df)))

    summary = {
        "symbol": SYMBOL,
        "combine_mode": combine_mode,
        "ml_threshold": ml_threshold,
        "horizon_bars": FUTURE_BARS_HORIZON,
        "spread_pips": SPREAD_PIPS,
        "commission_usd_rt": COMMISSION_USD_ROUND_TRIP,
        "trades": int(len(trades_df)),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": win_rate,
        "gross_profit_R": float(gross_profit),
        "gross_loss_R": float(gross_loss),
        "profit_factor": pf,
        "avg_R": ev,
        "max_dd_R": dd,
    }
    with open(out_dir / "phase6_hybrid_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # optional session aggregation
    sess_csv = out_dir / "phase6_hybrid_session_summary.csv"
    if "session" in trades_df.columns:
        trades_df.groupby("session")["R_net"].agg(["count", "mean", "sum"]).to_csv(sess_csv)

    print(f"Saved: {trades_csv}")
    print(f"Saved: {out_dir / 'phase6_hybrid_summary.json'}")
    if sess_csv.exists():
        print(f"Saved: {sess_csv}")

def _max_drawdown(r_series: np.ndarray) -> float:
    # simple equity curve drawdown in R
    eq = np.cumsum(r_series)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return float(np.max(dd)) if len(dd) else 0.0

if __name__ == "__main__":
    args = parse_args()

    # override module level costs from CLI
    SPREAD_PIPS = args.spread_pips
    COMMISSION_USD_ROUND_TRIP = args.commission_usd_rt

    run_hybrid(
        feat_path=Path(args.features),
        pred_path=Path(args.predictions),
        rules_path=Path(args.rules_trades),
        combine_mode=args.combine_mode,
        ml_threshold=args.ml_threshold,
        date_start=args.date_start,
        date_end=args.date_end,
        out_dir=Path(args.out_dir),
    )
