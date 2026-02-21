# EDS Project – Tabular Binary Classification with PyTorch

Loads a CSV dataset, preprocesses it, and trains a PyTorch neural network for binary classification. Optimized for Apple Silicon GPU (MPS), with CUDA and CPU fallback. Accepts any CSV where the last column is the binary target.

## Included Datasets

### electricity_binarized_UP.csv (default)

The [Electricity dataset](https://www.openml.org/d/151) — 38,474 rows, 8 columns. Predicts electricity price direction (up/down).

| Column      | Type    | Description                              |
|-------------|---------|------------------------------------------|
| date        | float64 | Normalized date                          |
| period      | float64 | Time period within the day               |
| nswprice    | float64 | New South Wales electricity price        |
| nswdemand   | float64 | New South Wales electricity demand       |
| vicprice    | float64 | Victoria electricity price               |
| vicdemand   | float64 | Victoria electricity demand              |
| transfer    | float64 | Scheduled transfer between states        |
| target      | int64   | Binary class label (0 or 1)              |

### covertype.csv (converted from ARFF)

The [Covertype dataset](https://www.openml.org/d/293) — 566,602 rows, 11 columns. Predicts forest cover type (binarized).

Convert from the included ARFF source file:

```bash
python convert_covertype.py
```

## Requirements

- Python 3.11+ (enforced in `pyproject.toml`)
- Packages listed in `requirements.txt`:
  - `pandas >= 2.0.0`
  - `scikit-learn >= 1.3.0`
  - `torch >= 2.0.0`

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
python process.py [dataset] [epochs] [--compile] [--early-stop]
```

| Argument | Default | Description |
|---|---|---|
| `dataset` | `electricity_binarized_UP.csv` | Path to input CSV file |
| `epochs` | `1000` | Maximum number of training epochs |
| `--compile` | *(off)* | Enable `torch.compile` for fused GPU kernels |
| `--early-stop` | *(off)* | Enable early stopping (patience=50, restores best weights) |

### Examples

```bash
python process.py                                          # defaults
python process.py covertype.csv                            # different dataset
python process.py covertype.csv 500                        # 500 epochs
python process.py covertype.csv 1000 --early-stop          # with early stopping
python process.py covertype.csv 1000 --compile --early-stop  # all options
```

### Expected Output

```
==================================================
Data file:   covertype.csv
Python:      3.11.12
Device:      mps
Dataset:     566602 rows, 11 columns
Training:    453281 samples | Test: 113321 samples
Model:       3,009 parameters
Epochs:      1000
Batch:       full-batch (453281 samples)
Precision:   mixed (float16)
Compile:     not requested (--compile to enable)
Compiled:    no
==================================================

Epoch   50/1000  Loss: 0.5432  LR: 0.005432  (5%)
Epoch  100/1000  Loss: 0.4891  LR: 0.009123  (10%)
...

Training complete in 12.345s (1000 epochs)
--------------------------------------------------
Evaluating on test set... done

Test Accuracy: 0.7523 (75.2%)
Total time:    14.567s
```

## What the Script Does

1. **Loads the CSV** into a pandas DataFrame (any CSV with a binary last column).
2. **Separates features and target** — all columns except the last become `X`, the last column becomes `y`.
3. **Splits into train/test** — 80% training, 20% test (`random_state=42` for reproducibility).
4. **Scales features** — applies `StandardScaler` (zero mean, unit variance). Fit on training data only to avoid data leakage.
5. **Moves data to GPU** — tensors are placed on MPS (Apple Silicon), CUDA, or CPU depending on availability.
6. **Optionally compiles the model** — `torch.compile()` for fused GPU kernels (enable with `--compile`).
7. **Trains a neural network** — a 3-layer fully connected network (input->64->32->1) with:
   - Full-batch gradient descent (all samples per step)
   - Adam optimizer with OneCycleLR scheduler (max LR 0.01)
   - BCEWithLogitsLoss
   - Mixed precision (float16) on GPU for faster throughput
   - Optional early stopping with `--early-stop` (patience=50, best-weight restoration)
8. **Evaluates on test set** — reports binary classification accuracy.

## Model Architecture

```
Input (N features)
  -> Linear(N, 64) -> ReLU
  -> Linear(64, 32) -> ReLU
  -> Linear(32, 1)
  -> BCEWithLogitsLoss
```

Parameter count depends on input features (e.g., 2,625 for 7 features, 3,009 for 10 features).

## Tags

| Tag      | Description                                              |
|----------|----------------------------------------------------------|
| `hybrid` | Original PyTorch version with BatchNorm/Dropout          |
| `GPU-1`  | GPU-optimized, lean model, 30 epochs                     |
| `GPU-2`  | GPU-optimized, lean model, 1000 epochs                   |
| `GPU-3`  | torch.compile, full-batch, early stopping, OneCycleLR    |
| `GPU-4`  | CLI args for dataset/epochs/compile, ARFF converter      |
| `GPU-5`  | Mixed precision, early-stop opt-in, compile off by default |

## Changelog: Tag-by-Tag

### `hybrid` -> `GPU-1`
- Removed BatchNorm and Dropout for speed
- Increased batch size from 32 to 2048
- Replaced DataLoader with GPU-side `torch.randperm` shuffling
- Reduced epochs from 100 to 30
- Used `zero_grad(set_to_none=True)` for memory efficiency
- Changed eval from `sigmoid(logits) >= 0.5` to `logits >= 0`
- Added MPS (Apple Silicon) device selection

### `GPU-1` -> `GPU-2`
- Increased epochs from 30 to 1000

### `GPU-2` -> `GPU-3`
- Added `torch.compile()` for fused GPU kernels
- Switched to full-batch training (eliminated inner mini-batch loop)
- Added early stopping (patience=50) with best-weight restoration
- Added OneCycleLR learning rate scheduler (max LR 0.01)
- Reduced print frequency to every 50 epochs
- Added summary banner before training

### `GPU-3` -> `GPU-4`
- Added CLI argument for input dataset (first arg, default: electricity CSV)
- Added CLI argument for epoch count (second arg, default: 1000)
- Added `--no-compile` flag to skip `torch.compile`
- Added compile status reporting in summary banner (requested vs actual)
- Added `convert_covertype.py` for ARFF-to-CSV conversion
- Added `pyproject.toml` with `requires-python >= 3.11`
- Wrapped `torch.compile` in try/except for Windows compatibility

### `GPU-4` -> `GPU-5`
- Added mixed precision training (float16 via `torch.autocast` + `GradScaler`)
- Changed compile default to OFF (use `--compile` to enable); compile was slower in practice for this model size
- Changed early stopping default to OFF (use `--early-stop` to enable); was triggering too aggressively
- Summary banner now reports precision mode (mixed float16 vs float32)
- Prints Python version at startup

## Project Structure

```
eds-project/
|-- electricity_binarized_UP.csv   # Default dataset
|-- covertype.                     # Covertype dataset (ARFF format)
|-- covertype.csv                  # Covertype dataset (after conversion)
|-- process.py                     # Training script
|-- convert_covertype.py           # ARFF-to-CSV converter
|-- convert.txt                    # Ed's original conversion reference
|-- requirements.txt               # pip dependencies
|-- pyproject.toml                 # Python version requirement
|-- README.md                      # This file
```
