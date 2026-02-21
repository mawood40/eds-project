import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

total_start = time.perf_counter()

# Load dataset
t = time.perf_counter()
df = pd.read_csv("electricity_binarized_UP.csv")
print(f"[Load CSV]        {time.perf_counter() - t:.3f}s")

# Separate features and target
t = time.perf_counter()
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(f"[Separate X/y]    {time.perf_counter() - t:.3f}s")

# Train/test split
t = time.perf_counter()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[Train/test split] {time.perf_counter() - t:.3f}s")

# Scale features (important for neural nets)
t = time.perf_counter()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(f"[Scale features]  {time.perf_counter() - t:.3f}s")

print(f"{'─' * 40}")
print(f"Total:            {time.perf_counter() - total_start:.3f}s")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
