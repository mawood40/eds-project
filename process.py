import sys
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print = lambda *args, **kwargs: __builtins__.__dict__["print"](*args, **kwargs, flush=True)

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
df = pd.read_csv("electricity_binarized_UP.csv")
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
# Define model (streamlined for speed — no BatchNorm/Dropout)
# ============================
BATCH_SIZE = 2048

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

# ============================
# Loss + optimizer
# ============================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(torch_model.parameters(), lr=0.001, weight_decay=0.0001)

# ============================
# Training loop (GPU-side shuffling, large batches)
# ============================
NUM_EPOCHS = 1000
N = X_train_t.shape[0]
print(f"\nTraining for {NUM_EPOCHS} epochs (batch size {BATCH_SIZE})...")
print("-" * 50)

t = time.perf_counter()
for epoch in range(NUM_EPOCHS):
    torch_model.train()
    perm = torch.randperm(N, device=device)

    for i in range(0, N, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb = X_train_t[idx]
        yb = y_train_t[idx]

        optimizer.zero_grad(set_to_none=True)
        logits = torch_model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    pct = (epoch + 1) / NUM_EPOCHS * 100
    bar = "#" * int(pct // 2) + "-" * (50 - int(pct // 2))
    print(f"\r[{bar}] {pct:5.1f}%  Epoch {epoch+1:3d}/{NUM_EPOCHS}  Loss: {loss.item():.4f}", end="")

    if (epoch + 1) % 10 == 0:
        print()

train_time = time.perf_counter() - t
print(f"\nTraining complete in {train_time:.3f}s")

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
