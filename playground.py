from sklearn.datasets import fetch_california_housing
import pandas as pd

# 1️⃣ Load dataset from sklearn
california = fetch_california_housing(as_frame=True)  # Returns a Pandas DataFrame

# Convert to DataFrame
df = california.frame  # Already has feature names

#  get all columns except MedHouseVal since this is the target column
X = df.drop(columns=["MedHouseVal"]).values

# get the target values
y = df['MedHouseVal'].values

print(X.shape)
print(y.shape)
print(y.reshape(1,-1))