
import json
import pandas as pd

# Load ground_truth.json
with open("ground_truth.json", "r") as f:
    ground_truth = json.load(f)

# Load results.json
with open("results.json", "r") as f:
    results = json.load(f)

# Convert to pandas DataFrames
df_gt = pd.DataFrame(ground_truth)
df_res = pd.DataFrame(results)

df_gt['year'] = df_gt['year'].astype(str)
df_res['year'] = df_res['year'].astype(str)

# Merge on 'year'
merged = pd.merge(df_gt, df_res, on="year", suffixes=("_gt", "_res"))
print(merged)

# Save as CSV
merged.to_csv("merged_results.csv", index=False)
