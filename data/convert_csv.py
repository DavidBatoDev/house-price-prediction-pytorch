import pandas as pd
import os
from sklearn.datasets import fetch_california_housing

# 1️⃣ Load dataset from sklearn
california = fetch_california_housing(as_frame=True)  # Returns a Pandas DataFrame

# Convert to DataFrame
df = california.frame  # Already has feature names

# 2️⃣ Ensure 'data/' folder exists
os.makedirs("data", exist_ok=True)

# 3️⃣ Save CSV file
csv_path = "data/california_housing.csv"
df.to_csv(csv_path, index=False)

print(f"✅ Dataset successfully saved to {csv_path}")
