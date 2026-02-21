import sys
import time
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print = lambda *args, **kwargs: __builtins__.__dict__["print"](*args, **kwargs, flush=True)

DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else "electricity_binarized_UP.csv"
NUM_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
USE_COMPILE = "--compile" in sys.argv

print(f"Python {sys.version}")

# ============================
# Device selection: MPS (Apple Silicon GPU) > CUDA > CPU
# ============================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ============================
# Load and preprocess dataset
# ============================
total_start = time.perf_counter()

print("Loading CSV...", end=" ")
t = time.perf_counter()
df = pd.read_csv(DATA_FILE)
print(f"done ({time.perf_counter() - t:.3f}s)")

print("Separating features/target...", end=" ")
t = time.perf_counter()
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(f"done ({time.perf_counter() - t:.3f}s)")

print("Splitting train/test...", end=" ")
t = time.perf_counter()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"done ({time.perf_counter() - t:.3f}s)")

print("Scaling features...", end=" ")
t = time.perf_counter()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"done ({time.perf_counter() - t:.3f}s)")

print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Training: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
print("-" * 50)

# ============================
# Convert data to tensors (move to device before training)
# ============================
print("Moving tensors to device...", end=" ")
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
print("done")

# ============================
# Define model (streamlined for speed)
# ============================
class TabularNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)

n_features = X_train.shape[1]
torch_model = TabularNN(n_features).to(device)
print(f"Model created: {sum(p.numel() for p in torch_model.parameters()):,} parameters")
print(f"Model device:  {next(torch_model.parameters()).device}")

# Compile model for fused GPU kernels (may fail on Windows without MSVC)
compiled = False
if USE_COMPILE:
    print("Compiling model...", end=" ")
    t = time.perf_counter()
    try:
        torch_model = torch.compile(torch_model)
        compiled = True
        print(f"done ({time.perf_counter() - t:.3f}s)")
    except Exception as e:
        print(f"skipped ({e})")
        print("  (torch.compile requires a C++ compiler on CPU; falling back to eager mode)")
else:
    print("Skipping torch.compile (pass --compile to enable)")

# ============================
# Loss + optimizer + scheduler
# ============================
N = X_train_t.shape[0]

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(torch_model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, epochs=NUM_EPOCHS, steps_per_epoch=1
)

# ============================
# Training loop (full-batch, early stopping)
# ============================
PATIENCE = 50
PRINT_EVERY = 50

best_loss = float("inf")
best_weights = None
wait = 0
stop_epoch = NUM_EPOCHS

print(f"\n{'=' * 50}")
print(f"Data file:   {DATA_FILE}")
print(f"Python:      {sys.version.split()[0]}")
print(f"Device:      {device}")
print(f"Dataset:     {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Training:    {X_train_t.shape[0]} samples | Test: {X_test_t.shape[0]} samples")
print(f"Model:       {sum(p.numel() for p in torch_model.parameters()):,} parameters")
print(f"Epochs:      up to {NUM_EPOCHS} (early stop patience={PATIENCE})")
print(f"Batch:       full-batch ({N} samples)")
print(f"Compile:     {'requested' if USE_COMPILE else 'not requested (--no-compile)'}")
print(f"Compiled:    {'yes' if compiled else 'no'}")
print(f"{'=' * 50}")
print()

t = time.perf_counter()
for epoch in range(NUM_EPOCHS):
    torch_model.train()

    optimizer.zero_grad(set_to_none=True)
    logits = torch_model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()
    scheduler.step()

    epoch_loss = loss.item()

    # Early stopping check
    if epoch_loss < best_loss - 1e-5:
        best_loss = epoch_loss
        best_weights = copy.deepcopy(torch_model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            stop_epoch = epoch + 1
            break

    if (epoch + 1) % PRINT_EVERY == 0:
        pct = (epoch + 1) / NUM_EPOCHS * 100
        print(f"Epoch {epoch+1:4d}/{NUM_EPOCHS}  Loss: {epoch_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}  ({pct:.0f}%)")

train_time = time.perf_counter() - t
print(f"\nTraining complete in {train_time:.3f}s ({stop_epoch} epochs, best loss: {best_loss:.4f})")

# Restore best weights
if best_weights is not None:
    torch_model.load_state_dict(best_weights)

# ============================
# Evaluation
# ============================
print("-" * 50)
print("Evaluating on test set...", end=" ")
torch_model.eval()
with torch.no_grad():
    logits = torch_model(X_test_t)
    preds = (logits >= 0).float()
    acc = preds.eq(y_test_t).sum().item() / len(y_test_t)

print("done")
print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.1f}%)")
print(f"Total time:    {time.perf_counter() - total_start:.3f}s")
