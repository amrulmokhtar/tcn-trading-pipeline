
import argparse
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_frames(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__source"] = os.path.basename(f)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")
    if not frames:
        raise RuntimeError("No readable CSV files")
    return frames

def coerce_num(s, default=0.0):
    out = pd.to_numeric(s, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(default)

def compute_atr(df: pd.DataFrame, n: int = 14):
    need = ["high","low","close"]
    if not all(c in df.columns for c in need):
        raise ValueError("CSV must contain high, low, close for ATR")
    high = coerce_num(df["high"])
    low = coerce_num(df["low"])
    close = coerce_num(df["close"])
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr = pd.Series(tr).rolling(n, min_periods=n).mean()
    return atr

def prepare(df: pd.DataFrame):
    # Ensure columns exist
    for c in ["time_utc","open","high","low","close","top_class",
              "p_long_smooth","p_short_smooth","p_none_smooth"]:
        if c not in df.columns:
            if c in ("time_utc","top_class"):
                df[c] = "None"
            else:
                df[c] = 0.0

    probs = df[["p_long_smooth","p_short_smooth","p_none_smooth"]].to_numpy(dtype=float)
    top_prob = probs.max(axis=1)
    second_best = np.partition(probs, -2, axis=1)[:, -2]
    delta_gap = top_prob - second_best
    df["__top_prob"] = top_prob
    df["__delta_gap"] = delta_gap
    return df

def simulate(df: pd.DataFrame, p: float, d: float, atr_len: int,
             sl_mult: float, tp_mult: float, max_bars_hold: int = 0):
    df = df.copy()
    df = prepare(df)
    atr = compute_atr(df, atr_len)
    df["__atr"] = atr

    # Only consider rows where ATR is ready
    mask_ready = ~df["__atr"].isna()
    df = df[mask_ready].reset_index(drop=True)

    # Acceptance mask
    acc_mask = (df["__top_prob"] >= p) & (df["__delta_gap"] >= d) & df["top_class"].isin(["Long","Short"])
    signals = df[acc_mask].copy()
    if signals.empty:
        return pd.DataFrame(), {"trades": 0, "pf": 0.0, "win_rate": 0.0, "avg_r": 0.0,
                                "max_dd": 0.0, "equity_end": 0.0}

    trades = []
    n = len(df)

    for idx in signals.index:
        entry_idx = idx + 1
        if entry_idx >= n:
            break
        direction = df.at[idx, "top_class"]
        entry_time = df.at[entry_idx, "time_utc"] if "time_utc" in df.columns else str(entry_idx)
        entry_open = float(df.at[entry_idx, "open"])
        atr_val = float(df.at[idx, "__atr"])

        # SL and TP levels
        sl = sl_mult * atr_val
        tp = tp_mult * atr_val

        if direction == "Long":
            sl_level = entry_open - sl
            tp_level = entry_open + tp
        else:
            sl_level = entry_open + sl
            tp_level = entry_open - tp

        exit_idx = None
        exit_reason = "timeout"
        r_result = 0.0
        bar_limit = n if max_bars_hold <= 0 else min(n, entry_idx + max_bars_hold)

        # Walk forward until SL or TP
        for j in range(entry_idx, bar_limit):
            high = float(df.at[j, "high"])
            low = float(df.at[j, "low"])

            hit_tp = (high >= tp_level) if direction == "Long" else (low <= tp_level)
            hit_sl = (low <= sl_level) if direction == "Long" else (high >= sl_level)

            if hit_tp and hit_sl:
                # assume worst case, SL first
                exit_idx = j
                r_result = -1.0
                exit_reason = "both_hit_SL_first"
                break
            elif hit_tp:
                exit_idx = j
                r_result = 1.0
                exit_reason = "tp"
                break
            elif hit_sl:
                exit_idx = j
                r_result = -1.0
                exit_reason = "sl"
                break

        if exit_idx is None:
            exit_idx = bar_limit - 1
            # mark to market R with close
            close_px = float(df.at[exit_idx, "close"])
            move = close_px - entry_open if direction == "Long" else entry_open - close_px
            r_result = move / sl if sl != 0 else 0.0

        trades.append({
            "entry_time": entry_time,
            "dir": direction,
            "entry_open": entry_open,
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
            "p": p,
            "d": d,
            "atr": atr_val,
            "exit_time": df.at[exit_idx, "time_utc"] if "time_utc" in df.columns else str(exit_idx),
            "exit_reason": exit_reason,
            "r": r_result
        })

    trades_df = pd.DataFrame(trades)

    # Equity stats
    equity = trades_df["r"].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = dd.min() if not dd.empty else 0.0
    wins = trades_df[trades_df["r"] > 0].shape[0]
    losses = trades_df[trades_df["r"] < 0].shape[0]
    gains = trades_df[trades_df["r"] > 0]["r"].sum()
    loss_sum = -trades_df[trades_df["r"] < 0]["r"].sum()
    pf = (gains / loss_sum) if loss_sum > 0 else float("inf") if gains > 0 else 0.0
    win_rate = wins / len(trades_df) if len(trades_df) else 0.0
    avg_r = trades_df["r"].mean() if not trades_df.empty else 0.0

    summary = {
        "trades": int(len(trades_df)),
        "pf": float(round(pf, 6)),
        "win_rate": float(round(win_rate, 6)),
        "avg_r": float(round(avg_r, 6)),
        "max_dd": float(round(abs(max_dd), 6)),
        "equity_end": float(round(equity.iloc[-1] if not equity.empty else 0.0, 6))
    }

    equity_df = pd.DataFrame({"trade": np.arange(1, len(trades_df)+1), "equity": equity})

    return trades_df, summary, equity_df

def main():
    ap = argparse.ArgumentParser(description="Option C, ATR-based SL/TP simulator for TCN predictions")
    ap.add_argument("--pattern", required=True, help="Glob for prediction CSVs, for example: predictions_csv/predictions_*.csv")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--p", type=float, default=0.70, help="p threshold for acceptance")
    ap.add_argument("--d", type=float, default=0.12, help="delta threshold for acceptance")
    ap.add_argument("--atr_len", type=float, default=14, help="ATR lookback")
    ap.add_argument("--sl_mult", type=float, default=1.0, help="SL multiple of ATR")
    ap.add_argument("--tp_mult", type=float, default=1.8, help="TP multiple of ATR")
    ap.add_argument("--max_bars_hold", type=int, default=0, help="0 means until SL or TP only")
    args = ap.parse_args()

    frames = load_frames(args.pattern)
    combined = pd.concat(frames, ignore_index=True)

    trades_df, summary, equity_df = simulate(
        combined, p=args.p, d=args.d, atr_len=int(args.atr_len),
        sl_mult=args.sl_mult, tp_mult=args.tp_mult, max_bars_hold=args.max_bars_hold
    )

    os.makedirs(args.outdir, exist_ok=True)

    trades_path = os.path.join(args.outdir, "step4_trades.csv")
    trades_df.to_csv(trades_path, index=False)

    summary_path = os.path.join(args.outdir, "step4_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    equity_path = os.path.join(args.outdir, "step4_equity.csv")
    equity_df.to_csv(equity_path, index=False)

    # equity plot
    plt.figure()
    plt.plot(equity_df["trade"], equity_df["equity"])
    plt.title("Equity Curve, Option C ATR Simulator")
    plt.xlabel("Trade #")
    plt.ylabel("Equity (R)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "step4_equity.png"), dpi=150)
    plt.close()

    print(f"[OK] Trades: {len(trades_df)}  PF: {summary['pf']:.3f}  WinRate: {summary['win_rate']:.3f}  AvgR: {summary['avg_r']:.3f}  MaxDD: {summary['max_dd']:.3f}")
    print(f"[OK] Wrote {trades_path}")
    print(f"[OK] Wrote {summary_path}")
    print(f"[OK] Wrote {equity_path}")
    print(f"[OK] Wrote step4_equity.png")

if __name__ == "__main__":
    main()
