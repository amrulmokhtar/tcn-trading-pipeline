# generate_phase_summary.py
# Generates a Word document summarizing Phase 0–5, lessons learned, achievements,
# best parameters/strategy notes, and adds a Phase 0→10 roadmap table.

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path
from datetime import datetime

# ---------- Output path ----------
OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "Phase0_to_Phase5_Summary_with_Lessons.docx"

# ---------- Helpers ----------
def add_title(doc: Document, text: str):
    h = doc.add_heading(text, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER

def add_kv_para(doc: Document, key: str, val: str):
    p = doc.add_paragraph()
    r1 = p.add_run(f"{key}: ")
    r1.bold = True
    p.add_run(val)

def add_table(doc: Document, headers, rows, col_widths_in=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
    for row in rows:
        cells = table.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    if col_widths_in:
        for col_i, w in enumerate(col_widths_in):
            for row in table.rows:
                row.cells[col_i].width = Inches(w)
    return table

# ---------- Build document ----------
doc = Document()

# Title
add_title(doc, "TCN_1.0 — Phase 0 to Phase 5 Summary")

# Metadata
doc.add_paragraph().add_run(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}").italic = True
doc.add_paragraph().add_run("Branch: pre_ml").italic = True

# 1. Overview
doc.add_heading("1. Overview", level=1)
doc.add_paragraph(
    "This document summarizes the progress from Phase 0 through Phase 5 of the TCN_1.0 project, "
    "including environment setup, data preparation, feature engineering, initial model training, "
    "and rule-based backtesting with diagnostics and Expected Value (EV). It also includes lessons learned, "
    "current best-known parameters for the pre-ML rules, and the roadmap through Phase 10."
)

# 1.1 Development Phases Overview (0→10)
doc.add_heading("1.1 Development Phases Overview (Phase 0 → 10)", level=2)
phases_overview = [
    ("Phase 0",  "Environment Setup",             "Git repo, venv, structure, deps"),
    ("Phase 1",  "Data Cleaning",                 "15m/1h historical prep & alignment"),
    ("Phase 2",  "Feature Engineering",           "ATR/EMA/ADX/OBV, sessions, volume flow"),
    ("Phase 3",  "Dataset Assembly",              "Merged, ML-ready parquet"),
    ("Phase 4",  "Model Training (Prototype)",    "Baseline TCN + scaler persistence"),
    ("Phase 5",  "Backtest + EV Tracking",        "Rules engine, diagnostics, grid sweeps"),
    ("Phase 6",  "ML Integration",                "Hybrid classifier (TCN/LightGBM), EV-weighted"),
    ("Phase 7",  "Real-Time Signal Service",      "Rolling feature inference in Python"),
    ("Phase 8",  "MT5 Bridge Integration",        "ZeroMQ or CSV bridge, order mgmt"),
    ("Phase 9",  "Monitoring & Risk Analytics",   "Live drift, PnL, DD, exposures"),
    ("Phase 10", "Packaging & CI/CD",             "Versioned release, cloud backtests"),
]
add_table(
    doc,
    headers=["Phase", "Focus", "Deliverable"],
    rows=phases_overview,
    col_widths_in=[1.2, 2.2, 3.4],
)
doc.add_paragraph("Current Progress: ✅ Phases 0–5 completed. 🚧 Phases 6–10 upcoming (ML, live service, MT5, CI/CD).")

# 2. Phase-by-Phase (0–5)
doc.add_heading("2. Phase-by-Phase Summary (0 → 5)", level=1)

# Phase 0
doc.add_heading("Phase 0 — Environment Setup", level=2)
doc.add_paragraph(
    "Created repository structure, virtual environment, .gitignore, and installed dependencies "
    "(pandas, numpy, scikit-learn, torch, python-docx). Confirmed local run and results folder."
)

# Phase 1
doc.add_heading("Phase 1 — Data Cleaning", level=2)
doc.add_paragraph(
    "Loaded raw 15-minute (M15) and 1-hour (H1) series, aligned timestamps, forward-filled missing bars, "
    "and validated monotonicity. Saved cleaned parquet (`data/m15_features.parquet`)."
)

# Phase 2
doc.add_heading("Phase 2 — Feature Engineering", level=2)
doc.add_paragraph(
    "Generated features: M15 ATR(14), EMA(20), spread_to_atr; H1 SMA(20/200), ADX(14), ATR(14), volatility_pct, "
    "OBV, OBV_SMA50, OBV slope; trading sessions (Asia/London/NY); high-level regime marks (range/breakout)."
)

# Phase 3
doc.add_heading("Phase 3 — Dataset Assembly", level=2)
doc.add_paragraph(
    "Merged features into a consistent, model-ready DataFrame with proper column ordering, "
    "and stored metadata (lookback, feature names) in `data/dataset.npz`."
)

# Phase 4
doc.add_heading("Phase 4 — Model Training (Prototype)", level=2)
doc.add_paragraph(
    "Implemented a baseline TCN prototype in PyTorch for later ML integration. Saved the feature scaler "
    "to `models/scaler.pkl` for consistent normalization during backtests."
)

# Phase 5
doc.add_heading("Phase 5 — Backtest + Diagnostics + EV", level=2)
doc.add_paragraph(
    "Built a rules-only backtester with session filters, regime/range checks, ADX gate, and OBV options. "
    "Added safe handling of missing `timestamp` by synthetically generating 15-min bars and deriving "
    "missing `spread_points` as needed. Implemented Expected Value (EV) tracking and created sweeps "
    "(`sweep_spread_cap.py`, `sweep_filters.py`, `grid_filters_spread.py`) to grid-search parameters."
)

doc.add_paragraph("Key artifacts saved to `results/`:")
add_table(
    doc,
    headers=["File", "What it contains"],
    rows=[
        ("backtest_summary.json", "Totals: win rate, PF, max DD, avg R, EV, diagnostics"),
        ("trades.csv", "All simulated trades (timestamp, side, R, etc.)"),
        ("spread_cap_sweep.csv", "Single-axis sweep results for spread/ATR cap"),
        ("filter_sweep.csv", "Combinations of filter toggles with metrics"),
        ("grid_sweep.csv", "Multi-parameter grid (caps + filters) ranked by score"),
    ],
    col_widths_in=[2.2, 4.0],
)

# 3. Lessons Learned
doc.add_heading("3. Lessons Learned", level=1)
doc.add_paragraph("• Always normalize features consistently using the saved scaler from training.")
doc.add_paragraph("• Many data sources are timestamp-naive; handle with synthetic bars or tz-localize early.")
doc.add_paragraph("• Spread constraints are the biggest driver of PF vs trade count — sweep narrow ranges.")
doc.add_paragraph("• EV (Expected Value) is essential to avoid over-indexing on PF alone.")
doc.add_paragraph("• Keep rules modular so we can flip filters on/off for apples-to-apples testing.")

# 4. Current Best (Pre-ML Rules)
doc.add_heading("4. Current Best (Pre-ML Rules)", level=1)
doc.add_paragraph(
    "From recent grid runs (≈5.9 years of M15), strong candidates cluster around "
    "`SPREAD_TO_ATR_CAP ∈ [8, 12]` and `MAX_SPREAD_POINTS ≈ 20–40`, with session/regime filters ON, ADX filter ON, "
    "and OBV confirmation optional. Use sweeps for your final instrument/broker feed."
)
add_table(
    doc,
    headers=["Param", "Value / Guidance", "Notes"],
    rows=[
        ("SPREAD_TO_ATR_CAP", "8–12", "Lower → fewer trades, higher PF; higher → more trades, lower PF"),
        ("MAX_SPREAD_POINTS", "20–40", "Protect against abnormal spikes on exotic pairs"),
        ("USE_SESSION", "True", "Asia/London/NY windows in MYT (09:00–03:00 wrap)"),
        ("USE_REGIME_FILTER", "True", "Skip range or include breakout per your rules"),
        ("USE_RANGE_SKIP", "True", "Avoid tight ranges before breakouts"),
        ("USE_ADX_FILTER", "True", "Gate by trend strength (e.g., ADX long ≥22, short ≥30)"),
        ("USE_OBV_PCT", "False", "Enable only if you add `% OBV` feature"),
        ("USE_OBV_CONFIRM", "True/False", "Slightly improves stability on some feeds"),
    ],
    col_widths_in=[2.0, 2.0, 4.0],
)

# 5. Strategy Notes (Pre-ML)
doc.add_heading("5. Strategy Notes (Pre-ML)", level=1)
doc.add_paragraph(
    "Entries: rule-based (range skip + ADX gate + sessions), with OBV optional confirmation. "
    "Exits: fixed TP/SL in R-multiples; diagnostics compute PF, max DD (in R), and avg R."
)
doc.add_paragraph(
    "Expected Value (EV): EV_R = win_rate * avg_win_R + (1 − win_rate) * avg_loss_R. "
    "We track avg_win_R and avg_loss_R separately to avoid PF illusions."
)

# 6. Roadmap (6–10) — action-oriented
doc.add_heading("6. Roadmap (Phases 6 → 10)", level=1)
doc.add_paragraph("• Phase 6 — ML Integration: train a classifier (e.g., LightGBM) on features + labels to predict EV-positive setups.")
doc.add_paragraph("• Phase 7 — Real-Time Service: convert backtest features to rolling calculation for live inference.")
doc.add_paragraph("• Phase 8 — MT5 Bridge: send signals via ZeroMQ or CSV drop; implement order/position sync & slippage control.")
doc.add_paragraph("• Phase 9 — Monitoring: add live dashboards (PnL, DD, drift, exposures) and daily EV sanity checks.")
doc.add_paragraph("• Phase 10 — Packaging: versioned configs, automated backtests, CI/CD for research and deployment.")

# Appendix
doc.add_heading("Appendix — Key Repository Paths", level=1)
add_table(
    doc,
    headers=["Path", "Description"],
    rows=[
        ("data/m15_features.parquet", "Primary features parquet used by backtester"),
        ("models/scaler.pkl", "Feature scaler learned during training"),
        ("scripts/phase5_backtest.py", "Main rules backtester with EV"),
        ("scripts/sweep_spread_cap.py", "One-dim sweep for SPREAD_TO_ATR_CAP"),
        ("scripts/sweep_filters.py", "Filter-toggle sweep"),
        ("scripts/grid_filters_spread.py", "Grid search for caps + filters with score"),
        ("results/", "All outputs (trades.csv, summaries, sweeps)"),
    ],
    col_widths_in=[2.3, 4.2],
)

# Save
doc.save(str(OUT_FILE))
print(f"✅ Word file written: {OUT_FILE.resolve()}")
