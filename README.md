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
python process.py [dataset] [epochs] [--no-compile]
```

| Argument | Default | Description |
|---|---|---|
| `dataset` | `electricity_binarized_UP.csv` | Path to input CSV file |
| `epochs` | `1000` | Maximum number of training epochs |
| `--no-compile` | *(off)* | Skip `torch.compile` (useful on Windows without MSVC) |

### Examples

```bash
python process.py                                          # defaults
python process.py covertype.csv                            # different dataset
python process.py covertype.csv 500                        # 500 epochs
python process.py electricity_binarized_UP.csv 200 --no-compile  # no compile
```

### Expected Output

```
==================================================
Data file:   electricity_binarized_UP.csv
Python:      3.11.12
Device:      mps
Dataset:     38474 rows, 8 columns
Training:    30779 samples | Test: 7695 samples
Model:       2,625 parameters
Epochs:      up to 1000 (early stop patience=50)
Batch:       full-batch (30779 samples)
Compile:     requested
Compiled:    yes
==================================================

Epoch   50/1000  Loss: 0.5432  LR: 0.005432  (5%)
Epoch  100/1000  Loss: 0.4891  LR: 0.009123  (10%)
...

Training complete in 1.234s (350 epochs, best loss: 0.4567)
--------------------------------------------------
Evaluating on test set... done

Test Accuracy: 0.8061 (80.6%)
Total time:    2.345s
```

## What the Script Does

1. **Loads the CSV** into a pandas DataFrame (any CSV with a binary last column).
2. **Separates features and target** — all columns except the last become `X`, the last column becomes `y`.
3. **Splits into train/test** — 80% training, 20% test (`random_state=42` for reproducibility).
4. **Scales features** — applies `StandardScaler` (zero mean, unit variance). Fit on training data only to avoid data leakage.
5. **Moves data to GPU** — tensors are placed on MPS (Apple Silicon), CUDA, or CPU depending on availability.
6. **Compiles the model** — uses `torch.compile()` for fused GPU kernels (skipped with `--no-compile` or if no C++ compiler is available).
7. **Trains a neural network** — a 3-layer fully connected network (input→64→32→1) with:
   - Full-batch gradient descent (all samples per step)
   - Adam optimizer with OneCycleLR scheduler (max LR 0.01)
   - BCEWithLogitsLoss
   - Early stopping (patience=50) with best-weight restoration
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

## Evolution: `hybrid` vs `GPU-4`

The `hybrid` tag represents the initial working version. `GPU-4` is the current version with all optimizations and CLI support.

| Aspect | `hybrid` | `GPU-4` |
|---|---|---|
| **Input** | Hardcoded CSV path | CLI argument (any CSV) |
| **Model layers** | Linear->ReLU->BatchNorm->Dropout(0.3)->Linear->ReLU->Linear | Linear->ReLU->Linear->ReLU->Linear |
| **Regularization** | BatchNorm + Dropout (0.3) | None (removed for speed) |
| **Batch size** | 32 (mini-batch via DataLoader) | Full-batch (all samples) |
| **Shuffling** | DataLoader on CPU | Not needed (full-batch) |
| **Epochs** | 100 (fixed) | CLI configurable, up to 1000 default, early stopping |
| **Learning rate** | Fixed 0.001 | OneCycleLR (ramps to 0.01, then decays) |
| **torch.compile** | No | Yes by default (--no-compile to disable) |
| **Early stopping** | No | Yes (patience=50, restores best weights) |
| **Evaluation** | `sigmoid(logits) >= 0.5` | `logits >= 0` (equivalent, faster) |
| **Device selection** | CUDA > CPU | MPS > CUDA > CPU |
| **Progress output** | Every epoch | Every 50 epochs + summary banner |
| **Cross-platform** | Windows path hardcoded | Runs on macOS, Windows, Linux |

### Why these changes matter

- **Full-batch** eliminates Python-level GPU round trips per epoch, drastically reducing overhead.
- **torch.compile** fuses the model's layers into optimized kernels, cutting per-step dispatch time.
- **OneCycleLR** converges faster than a fixed learning rate, reaching better loss in fewer epochs.
- **Early stopping** prevents wasting time on epochs after convergence and avoids overfitting.
- **CLI arguments** allow testing on different datasets and configurations without editing code.

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
