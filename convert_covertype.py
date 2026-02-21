from pathlib import Path
import csv
import pandas as pd

input_file = Path("covertype.")
output_file = Path("covertype.csv")

attribute_names = []
data_rows = []
in_data_section = False

with input_file.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.lower().startswith("@attribute"):
            parts = line.split()
            name = parts[1].strip('"')
            attribute_names.append(name)
        elif line.lower().startswith("@data"):
            in_data_section = True
        elif in_data_section:
            data_rows.append(line.split(","))

with output_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(attribute_names)
    writer.writerows(data_rows)

print(f"\nCSV saved to: {output_file}")

df = pd.read_csv(output_file)

print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

print("\nColumn names and inferred types:")
print(df.dtypes)

print("---------------------------------------")
print("\nUnique value counts per column:")
unique_counts = df.nunique(dropna=False)
for col, count in unique_counts.items():
    print(f"  {col}: {count} unique values")

print("---------------------------------------")
print("\nSummary statistics (numeric columns):")
print(df.describe(include='number'))

print("\nTop value counts (categorical columns):")
for col in df.select_dtypes(include='object').columns:
    print(f"\n  {col} (top 5 values):")
    print(df[col].value_counts().head(5))
