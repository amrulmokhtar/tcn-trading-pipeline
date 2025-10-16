
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
    # Needs high, low, close
    req = ["high","low","close"]
    if not all(c in df.columns for c in req):
        return None
    high = coerce_num(df["high"])
    low = coerce_num(df["low"])
    close = coerce_num(df["close"])
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    atr = pd.Series(tr).rolling(n, min_periods=n).mean()
    return atr

def prepare(df: pd.DataFrame, k: int, eps_pct: float, eps_atr_mult: float, atr_len: int):
    # Ensure needed columns
    for c in ["top_class","p_long_smooth","p_short_smooth","p_none_smooth","open","high","low","close"]:
        if c not in df.columns:
            if c in ("top_class",):
                df[c] = "None"
            else:
                df[c] = 0.0

    # Acceptance features from smoothed probs
    probs = df[["p_long_smooth","p_short_smooth","p_none_smooth"]].to_numpy(dtype=float)
    top_prob = probs.max(axis=1)
    second_best = np.partition(probs, -2, axis=1)[:, -2]
    delta_gap = top_prob - second_best
    df["__top_prob"] = top_prob
    df["__delta_gap"] = delta_gap

    # Horizon based realized direction
    close = coerce_num(df["close"])
    future = close.shift(-k)
    ret = (future - close) / close

    realized = np.where(ret >= eps_pct, "Long",
                np.where(ret <= -eps_pct, "Short", "None"))

    # If ATR option provided, build ATR and override epsilon test when ATR available
    if eps_atr_mult is not None and eps_atr_mult > 0:
        atr = compute_atr(df, atr_len)
        if atr is not None is not False:
            thr = eps_atr_mult * atr / close.replace(0, np.nan)
            thr = thr.fillna(eps_pct)  # fallback to percent where ATR not ready
            realized = np.where(ret >= thr, "Long",
                        np.where(ret <= -thr, "Short", "None"))

    df["__realized_k"] = realized.astype(str)

    return df

def eval_grid(df: pd.DataFrame, pmin, pmax, pstep, dmin, dmax, dstep):
    ps = np.arange(pmin, pmax + 1e-9, pstep)
    ds = np.arange(dmin, dmax + 1e-9, dstep)
    rows = []
    for p in ps:
        for d in ds:
            mask = (df["__top_prob"] >= p) & (df["__delta_gap"] >= d)
            acc = df[mask]
            total = len(df)
            n = len(acc)
            acc_rate = n / total if total else 0.0
            # accuracy with top_class vs horizon realized
            correct = (acc["top_class"].astype(str) == acc["__realized_k"].astype(str)).astype(int)
            acc_on_accepted = correct.mean() if n else 0.0

            # 1:1 RRR PF on Long or Short only
            is_ls = acc["top_class"].isin(["Long","Short"])
            pnl = np.where(is_ls & (acc["top_class"] == acc["__realized_k"]), 1,
                           np.where(is_ls & (acc["top_class"] != acc["__realized_k"]), -1, 0))
            gains = pnl[pnl > 0].sum()
            losses = -pnl[pnl < 0].sum()
            if losses > 0:
                pf = gains / losses
            else:
                pf = float("inf") if gains > 0 else 0.0

            rows.append({
                "p": round(float(p), 4),
                "d": round(float(d), 4),
                "rows_total": int(total),
                "accepted_rows": int(n),
                "acceptance_rate": round(float(acc_rate), 6),
                "accuracy_on_accepted": round(float(acc_on_accepted), 6),
                "pf_rr1": round(float(pf), 6),
            })
    grid = pd.DataFrame(rows)
    return grid, ps, ds

def save_heatmap(grid: pd.DataFrame, ps, ds, out_png: str, value_col="pf_rr1"):
    pivot = grid.pivot(index="d", columns="p", values=value_col)
    plt.figure()
    im = plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(im)
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=[f"{x:.2f}" for x in pivot.columns], rotation=90)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=[f"{y:.2f}" for y in pivot.index])
    plt.title(f"Heatmap of {value_col} over thresholds")
    plt.xlabel("p threshold")
    plt.ylabel("d threshold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Option B, horizon aligned evaluation using k step ahead and epsilon threshold")
    ap.add_argument("--pattern", required=True, help="Glob, for example: predictions_csv/predictions_*.csv")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--k", type=int, default=3, help="Horizon in bars")
    ap.add_argument("--eps_pct", type=float, default=0.001, help="Epsilon as fraction of price, 0.001 = 0.1 percent")
    ap.add_argument("--eps_atr_mult", type=float, default=0.0, help="If > 0, use ATR multiple as epsilon")
    ap.add_argument("--atr_len", type=int, default=14, help="ATR lookback for epsilon when eps_atr_mult > 0")
    ap.add_argument("--pmin", type=float, default=0.50)
    ap.add_argument("--pmax", type=float, default=0.90)
    ap.add_argument("--pstep", type=float, default=0.02)
    ap.add_argument("--dmin", type=float, default=0.08)
    ap.add_argument("--dmax", type=float, default=0.20)
    ap.add_argument("--dstep", type=float, default=0.02)
    args = ap.parse_args()

    frames = load_frames(args.pattern)
    combined = pd.concat(frames, ignore_index=True)

    combined = prepare(combined, k=args.k, eps_pct=args.eps_pct, eps_atr_mult=args.eps_atr_mult, atr_len=args.atr_len)

    grid, ps, ds = eval_grid(combined, args.pmin, args.pmax, args.pstep, args.dmin, args.dmax, args.dstep)

    os.makedirs(args.outdir, exist_ok=True)
    grid_path = os.path.join(args.outdir, "step3B_threshold_grid.csv")
    grid.to_csv(grid_path, index=False)

    top5 = grid.sort_values(["pf_rr1","accepted_rows"], ascending=[False, False]).head(5)
    top5_path = os.path.join(args.outdir, "step3B_top5.csv")
    top5.to_csv(top5_path, index=False)

    heatmap_path = os.path.join(args.outdir, "step3B_pf_heatmap.png")
    save_heatmap(grid, ps, ds, heatmap_path, value_col="pf_rr1")

    # overall snapshot at your current defaults
    sel = grid.sort_values(["pf_rr1","accepted_rows"], ascending=[False, False]).head(1)
    sel.to_csv(os.path.join(args.outdir, "step3B_best_at_defaults.csv"), index=False)

    print(f"[OK] Wrote {grid_path}")
    print(f"[OK] Wrote {top5_path}")
    print(f"[OK] Wrote {heatmap_path}")
    print("[OK] Wrote step3B_best_at_defaults.csv")
    print("Done.")

if __name__ == "__main__":
    main()
