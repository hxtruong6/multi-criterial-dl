import pandas as pd

# Load predictions
preds = pd.read_csv("results/predictions.csv")

# Show sample predictions
print(preds.head())

# Get concept distribution
concepts = [c for concepts in preds["concepts"] for c in concepts]
print("\nMost common concepts:")
print(pd.Series(concepts).value_counts().head())
