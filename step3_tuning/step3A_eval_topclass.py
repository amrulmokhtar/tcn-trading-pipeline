
import argparse
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REQUIRED_COLS = [
    "top_class", "realized_dir", "correct",
    "p_long_smooth", "p_short_smooth", "p_none_smooth"
]

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

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS:
        if c not in df.columns:
            if c in ("top_class", "realized_dir"):
                df[c] = "None"
            else:
                df[c] = 0.0

    for c in ["correct", "p_long_smooth", "p_short_smooth", "p_none_smooth"]:
        df[c] = coerce_num(df[c], 0.0)

    probs = df[["p_long_smooth","p_short_smooth","p_none_smooth"]].to_numpy(dtype=float)
    top_prob = probs.max(axis=1)
    second_best = np.partition(probs, -2, axis=1)[:, -2]
    delta_gap = top_prob - second_best
    df["__top_prob"] = top_prob
    df["__delta_gap"] = delta_gap

    df["top_class"] = df["top_class"].astype(str)
    df["realized_dir"] = df["realized_dir"].astype(str)

    corr = (df["top_class"] == df["realized_dir"]).astype(int)
    if "correct" in df.columns:
        df["__correct_eval"] = np.where(df["correct"].isna(), corr, df["correct"]).astype(int)
    else:
        df["__correct_eval"] = corr

    return df

def eval_metrics(df: pd.DataFrame, p: float, d: float) -> dict:
    mask_acc = (df["__top_prob"] >= p) & (df["__delta_gap"] >= d)
    acc_df = df[mask_acc]

    total = len(df)
    acc_rows = len(acc_df)
    acc_rate = acc_rows / total if total else 0.0

    accuracy_on_accepted = acc_df["__correct_eval"].mean() if acc_rows else 0.0

    is_ls = acc_df["top_class"].isin(["Long","Short"])
    pnl = np.where(is_ls & (acc_df["top_class"] == acc_df["realized_dir"]), 1,
                   np.where(is_ls & (acc_df["top_class"] != acc_df["realized_dir"]), -1, 0))
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses > 0:
        pf = gains / losses
    else:
        pf = float("inf") if gains > 0 else 0.0

    return {
        "p": round(p, 4),
        "d": round(d, 4),
        "rows_total": int(total),
        "accepted_rows": int(acc_rows),
        "acceptance_rate": round(acc_rate, 6),
        "accuracy_on_accepted": round(float(accuracy_on_accepted), 6),
        "pf_rr1": round(float(pf), 6),
    }

def grid_search(df: pd.DataFrame, p_min=0.50, p_max=0.95, p_step=0.02,
                d_min=0.05, d_max=0.25, d_step=0.02):
    ps = np.arange(p_min, p_max + 1e-9, p_step)
    ds = np.arange(d_min, d_max + 1e-9, d_step)
    rows = []
    for p in ps:
        for d in ds:
            rows.append(eval_metrics(df, float(p), float(d)))
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
    ap = argparse.ArgumentParser(description="Option A, use CSV top_class for prediction, smoothed probs for acceptance")
    ap.add_argument("--pattern", required=True, help="Glob, for example: predictions_csv/predictions_*.csv")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--pmin", type=float, default=0.50)
    ap.add_argument("--pmax", type=float, default=0.95)
    ap.add_argument("--pstep", type=float, default=0.02)
    ap.add_argument("--dmin", type=float, default=0.05)
    ap.add_argument("--dmax", type=float, default=0.25)
    ap.add_argument("--dstep", type=float, default=0.02)
    args = ap.parse_args()

    frames = load_frames(args.pattern)
    combined = pd.concat(frames, ignore_index=True)
    combined = prepare(combined)

    grid, ps, ds = grid_search(combined, args.pmin, args.pmax, args.pstep, args.dmin, args.dmax, args.dstep)

    os.makedirs(args.outdir, exist_ok=True)
    grid_path = os.path.join(args.outdir, "step3A_threshold_grid.csv")
    grid.to_csv(grid_path, index=False)

    top5 = grid.sort_values(["pf_rr1","accepted_rows"], ascending=[False, False]).head(5)
    top5_path = os.path.join(args.outdir, "step3A_top5.csv")
    top5.to_csv(top5_path, index=False)

    heatmap_path = os.path.join(args.outdir, "step3A_pf_heatmap.png")
    save_heatmap(grid, ps, ds, heatmap_path, value_col="pf_rr1")

    print(f"[OK] Wrote {grid_path}")
    print(f"[OK] Wrote {top5_path}")
    print(f"[OK] Wrote {heatmap_path}")
    print("Done.")

if __name__ == "__main__":
    main()
