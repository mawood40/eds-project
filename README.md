# EDS Project – Electricity Dataset Preprocessing

Loads and preprocesses the [Electricity dataset](https://www.openml.org/d/151) for machine learning benchmarking. The script reads the CSV, splits it into train/test sets, and applies standard scaling so the data is ready for model training (e.g., neural networks).

## Dataset

**File:** `electricity_binarized_UP.csv` (38,474 rows, 8 columns)

| Column      | Type    | Description                              |
|-------------|---------|------------------------------------------|
| date        | float64 | Normalized date                          |
| period      | float64 | Time period within the day               |
| nswprice    | float64 | New South Wales electricity price        |
| nswdemand   | float64 | New South Wales electricity demand       |
| vicprice    | float64 | Victoria electricity price               |
| vicdemand   | float64 | Victoria electricity demand              |
| transfer    | float64 | Scheduled transfer between states        |
| target      | int64   | Binary class label (0 or 1) — price up/down |

Columns 1–7 are used as features (`X`). The last column (`target`) is the label (`y`).

## Requirements

- Python 3.11+
- Packages listed in `requirements.txt`:
  - `pandas >= 2.0.0`
  - `scikit-learn >= 1.3.0`

## Setup

```bash
# Create a virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python process.py
```

### Expected Output

```
[Load CSV]        0.008s
[Separate X/y]    0.000s
[Train/test split] 0.002s
[Scale features]  0.002s
────────────────────────────────────────
Total:            0.012s
Dataset loaded: 38474 rows, 8 columns
Training set: 30779 samples
Test set: 7695 samples
```

## What the Script Does

1. **Loads the CSV** into a pandas DataFrame.
2. **Separates features and target** — all columns except the last become `X`, the last column becomes `y`.
3. **Splits into train/test** — 80% training, 20% test (`random_state=42` for reproducibility).
4. **Scales features** — applies `StandardScaler` (zero mean, unit variance), which is important for gradient-based models like neural networks. The scaler is fit on training data only to avoid data leakage.

## Project Structure

```
eds-project/
├── electricity_binarized_UP.csv   # Dataset
├── process.py                     # Preprocessing script
├── requirements.txt               # pip dependencies
└── README.md                      # This file
```
