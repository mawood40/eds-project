# EDS Project – Electricity Dataset Neural Network

Loads the [Electricity dataset](https://www.openml.org/d/151), preprocesses it, and trains a PyTorch neural network for binary classification (predicting electricity price direction). Optimized for Apple Silicon GPU (MPS), with CUDA and CPU fallback.

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
python process.py
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

1. **Loads the CSV** into a pandas DataFrame.
2. **Separates features and target** — all columns except the last become `X`, the last column becomes `y`.
3. **Splits into train/test** — 80% training, 20% test (`random_state=42` for reproducibility).
4. **Scales features** — applies `StandardScaler` (zero mean, unit variance). Fit on training data only to avoid data leakage.
5. **Moves data to GPU** — tensors are placed on MPS (Apple Silicon), CUDA, or CPU depending on availability.
6. **Compiles the model** — uses `torch.compile()` for fused GPU kernels and reduced Python overhead.
7. **Trains a neural network** — a 3-layer fully connected network (7→64→32→1) with:
   - Full-batch gradient descent (all samples per step)
   - Adam optimizer with OneCycleLR scheduler (max LR 0.01)
   - BCEWithLogitsLoss
   - Early stopping (patience=50) with best-weight restoration
8. **Evaluates on test set** — reports binary classification accuracy.

## Model Architecture

```
Input (7 features)
  → Linear(7, 64) → ReLU
  → Linear(64, 32) → ReLU
  → Linear(32, 1)
  → BCEWithLogitsLoss
```

2,625 total parameters.

## Tags

| Tag      | Description                                          |
|----------|------------------------------------------------------|
| `hybrid` | Original PyTorch version with BatchNorm/Dropout      |
| `GPU-1`  | GPU-optimized, lean model, 30 epochs                 |
| `GPU-2`  | GPU-optimized, lean model, 1000 epochs               |
| `GPU-3`  | torch.compile, full-batch, early stopping, OneCycleLR |

## Evolution: `hybrid` vs `GPU-3`

The `hybrid` tag represents the initial working version. `GPU-3` is the current optimized version. Here is a detailed comparison:

| Aspect | `hybrid` | `GPU-3` |
|---|---|---|
| **Model layers** | Linear→ReLU→BatchNorm→Dropout(0.3)→Linear→ReLU→Linear | Linear→ReLU→Linear→ReLU→Linear |
| **Regularization** | BatchNorm + Dropout (0.3) | None (removed for speed) |
| **Parameters** | 2,753 | 2,625 |
| **Batch size** | 32 (mini-batch via DataLoader) | Full-batch (all 30,779 samples) |
| **Shuffling** | DataLoader on CPU | Not needed (full-batch) |
| **Epochs** | 100 (fixed) | Up to 1000 with early stopping (patience=50) |
| **Learning rate** | Fixed 0.001 | OneCycleLR (ramps to 0.01, then decays) |
| **LR scheduler** | None | OneCycleLR |
| **Optimizer** | Adam (lr=0.001, weight_decay=0.0001) | Adam (lr=0.001, weight_decay=0.0001) |
| **zero_grad** | Default | `set_to_none=True` (saves memory) |
| **torch.compile** | No | Yes (fused GPU kernels) |
| **Early stopping** | No | Yes (patience=50, restores best weights) |
| **Evaluation** | `sigmoid(logits) >= 0.5` | `logits >= 0` (equivalent, skips sigmoid) |
| **Device selection** | CUDA > CPU | MPS (Apple Silicon) > CUDA > CPU |
| **Progress output** | Every epoch | Every 50 epochs + summary banner |

### Why these changes matter

- **Full-batch** eliminates 14 of 15 Python-level GPU round trips per epoch, drastically reducing overhead for this small dataset.
- **torch.compile** fuses the model's layers into optimized kernels, cutting per-step dispatch time.
- **OneCycleLR** converges faster than a fixed learning rate, reaching better loss in fewer epochs.
- **Early stopping** prevents wasting time on epochs after convergence and avoids overfitting.
- **Removing BatchNorm/Dropout** simplifies the model for speed; with full-batch training on 30K samples, the regularization benefit is minimal.

## Project Structure

```
eds-project/
|-- electricity_binarized_UP.csv   # Dataset
|-- process.py                     # Training script
|-- requirements.txt               # pip dependencies
|-- pyproject.toml                 # Python version requirement
|-- README.md                      # This file
```
