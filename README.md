README.md
---

# 🧠 Temporal Convolutional Network (TCN) Trading Pipeline

A complete end-to-end machine learning pipeline that builds, trains, and evaluates a **Temporal Convolutional Network (TCN)** model for intraday trading.  
The objective is to **maximize Profit Factor (PF > 2.0)** while keeping **Drawdown (DD < 10%)**, using multi-timeframe features from 15-minute and 1-hour charts.

---

## 🎯 Objectives

| Goal | Metric | Target |
|------|---------|--------|
| **Profitability** | Profit Factor (PF) | **> 2.0** |
| **Risk Management** | Max Drawdown (DD) | **< 10%** |
| **Consistency** | Win Rate | > 60% |
| **Stability** | Sharpe Ratio | > 1.5 |

---

## ⚙️ Architecture Overview

### **Data Flow**
Raw CSV (15M + 1H)
↓
Phase 1 – Data Cleaning
↓
Phase 2 – Feature Engineering (EMA, ATR, ADX, OBV, Session Flags)
↓
Phase 3 – Dataset Assembly (Lookback + Target)
↓
Phase 4 – Model Training (TCN)
↓
Phase 5 – Evaluation & Backtesting

---

## 🧩 Project Structure
tcn_1.0/
│
├── data/ # Raw, cleaned, and feature files
│ ├── 15M_data.csv
│ ├── 1H_data.csv
│ ├── cleaned_15M_data.csv
│ ├── cleaned_1H_data.csv
│ ├── m15_features.parquet
│ ├── dataset.npz
│ └── splits.json
│
├── models/ # Trained model artifacts
│ ├── tcn_signal.pt
│ └── scaler.pkl
│
├── results/ # Backtest and evaluation metrics
│
├── scripts/ # Core pipeline scripts
│ ├── phase1_clean_data.py
│ ├── phase2_features.py
│ ├── phase3_dataset.py
│ ├── phase4_train_tcn.py # (to be added)
│ ├── optimize_thresholds.py
│ └── run_sliding_inference.py
│
└── README.md


---

## 🧠 Key Features

- **Multi-Timeframe Data Fusion**  
  Combines 15-minute execution data with 1-hour market regime context.

- **Feature Engineering**  
  - EMA (20), ATR(14), ADX(14), SMA(200)  
  - OBV, ATR percentile, Volume percentile  
  - Market Session Flags (Asia, London, New York)

- **Model: Temporal Convolutional Network (TCN)**  
  Built using `pytorch-tcn`, trained with early stopping and validation metrics (PF, Sharpe, DD).

- **Pipeline Automation**  
  Modular scripts handle each stage: cleaning, feature generation, dataset assembly, training, and evaluation.

---

## 🧰 Environment Setup

### 1️⃣ Create a virtual environment
```bash
python -m venv .venv

2️⃣ Activate it
.venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ (Optional) Save environment
pip freeze > requirements.txt

🧪 Run the Pipeline
Step 1: Clean Data
python scripts/phase1_clean_data.py

Step 2: Build Features
python scripts/phase2_features.py

Step 3: Prepare Dataset
python scripts/phase3_dataset.py

Step 4: Train TCN Model (up next)
python scripts/phase4_train_tcn.py

Evaluation Metrics
- Profit Factor (PF) = Gross Profit / Gross Loss
- Max Drawdown (DD) = Peak-to-trough equity decline
- Sharpe Ratio = Risk-adjusted performance
- Confusion Matrix for directional accuracy

Technologies Used
- Python 3.10
- PyTorch + pytorch-tcn
- NumPy, Pandas, Scikit-learn
- Matplotlib, TQDM
- Joblib, Parquet, JSON for data storage

Phase 1–3 completed (Data Cleaning, Feature Engineering, Dataset Preparation)
- Next: Phase 4 – Model Training & Tuning
- Add attention layer or Transformer hybrid for better context capture
- Integrate live inference with cTrader or MT5 via API
- Add dashboard for live PF/DD monitoring