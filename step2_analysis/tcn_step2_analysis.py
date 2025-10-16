
import os
import glob
import argparse
import pandas as pd
from collections import Counter

VALID_DIRS = {"Long","Short","None"}

def load_csvs(path_pattern: str):
    files = sorted(glob.glob(path_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {path_pattern}")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__source"] = os.path.basename(f)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")
    if not frames:
        raise RuntimeError("No readable CSV files were found")
    return frames

def coerce_numeric(series, default=0):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([float("inf"), float("-inf")], pd.NA)
    return s.fillna(default)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Create columns if missing, to support older logs
    for col in ("accepted","top_class","realized_dir","correct","top_prob","delta_gap","cfg_p","cfg_delta"):
        if col not in df.columns:
            if col in ("accepted","correct"):
                df[col] = 0
            else:
                df[col] = pd.NA

    # Coerce numeric columns safely
    for col in ("accepted","correct","top_prob","delta_gap","cfg_p","cfg_delta"):
        df[col] = coerce_numeric(df[col], default=0)

    # Normalize classes as strings
    df["top_class"] = df["top_class"].astype(str)
    df["realized_dir"] = df["realized_dir"].astype(str)

    # If 'correct' is missing or NaN, recompute from classes when possible
    # correct = 1 when realized_dir equals top_class and top_class is Long or Short
    mask_can_eval = df["realized_dir"].isin(VALID_DIRS) & df["top_class"].isin(VALID_DIRS)
    df.loc[mask_can_eval & df["correct"].isna(), "correct"] = (df["realized_dir"] == df["top_class"]).astype(int)

    # Anything else, fill remaining NaNs in correct with 0
    df["correct"] = coerce_numeric(df["correct"], default=0).astype(int)

    # Ensure accepted is integer 0 or 1
    df["accepted"] = df["accepted"].clip(lower=0, upper=1).astype(int)

    return df

def compute_metrics(df: pd.DataFrame):
    df = normalize_columns(df)

    total = len(df)
    acc_all = df["correct"].mean() if total else 0.0

    accepted_df = df[df["accepted"] == 1]
    n_acc = len(accepted_df)
    acc_rate = n_acc / total if total else 0.0
    acc_on_accepted = accepted_df["correct"].mean() if n_acc else 0.0

    preds = Counter(accepted_df["top_class"]) if n_acc else Counter()
    reals = Counter(accepted_df["realized_dir"]) if n_acc else Counter()

    conf = accepted_df.groupby(["top_class","realized_dir"]).size().unstack(fill_value=0) if n_acc else pd.DataFrame()

    if n_acc:
        pnl = accepted_df.apply(lambda r: 1 if r["correct"]==1 and r["top_class"] in ("Long","Short") 
                                else (-1 if r["correct"]==0 and r["top_class"] in ("Long","Short") else 0), axis=1)
        gains = pnl[pnl>0].sum()
        losses = -pnl[pnl<0].sum()
        pf = (gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
    else:
        pf = 0.0

    summary = {
        "rows_total": int(total),
        "accepted_rows": int(n_acc),
        "acceptance_rate": round(acc_rate, 6),
        "accuracy_all_rows": round(acc_all, 6),
        "accuracy_on_accepted": round(acc_on_accepted, 6),
        "pf_estimate_rr1": round(pf, 6),
        "pred_Long": int(preds.get("Long", 0)),
        "pred_Short": int(preds.get("Short", 0)),
        "pred_None": int(preds.get("None", 0)),
        "real_Long": int(reals.get("Long", 0)),
        "real_Short": int(reals.get("Short", 0)),
        "real_None": int(reals.get("None", 0)),
    }

    return summary, conf

def main():
    ap = argparse.ArgumentParser(description="EA_3_0_TCN Step 2 analysis, robust to mixed CSV versions")
    ap.add_argument("--pattern", required=True, help="Glob pattern for prediction CSVs, for example: 'predictions_csv/predictions_*.csv'")
    ap.add_argument("--outdir", default=".", help="Directory to write summary files")
    args = ap.parse_args()

    frames = load_csvs(args.pattern)

    per_file = []
    for df in frames:
        src = df["__source"].iloc[0]
        try:
            s, conf = compute_metrics(df)
            s["source"] = src
            per_file.append(s)
            if not conf.empty:
                conf_path = os.path.join(args.outdir, f"confusion_{src}.csv")
                conf.to_csv(conf_path, index=True)
        except Exception as e:
            print(f"[WARN] Skipping {src} due to error: {e}")

    if per_file:
        per_file_df = pd.DataFrame(per_file).sort_values("source")
        per_file_path = os.path.join(args.outdir, "step2_metrics_per_file.csv")
        per_file_df.to_csv(per_file_path, index=False)
        print(f"[OK] Wrote {per_file_path}")

        combined = pd.concat(frames, ignore_index=True)
        combined_summary, combined_conf = compute_metrics(combined)
        combined_df = pd.DataFrame([combined_summary])
        combined_path = os.path.join(args.outdir, "step2_metrics_overall.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"[OK] Wrote {combined_path}")
        if not combined_conf.empty:
            combined_conf.to_csv(os.path.join(args.outdir, "confusion_overall.csv"), index=True)
            print("[OK] Wrote confusion_overall.csv")
    else:
        print("[WARN] No files produced metrics")

if __name__ == "__main__":
    main()
