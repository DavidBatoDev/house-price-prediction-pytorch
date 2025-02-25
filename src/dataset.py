import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class CaliforniaHousingDataset(Dataset):
    def __init__(self, csv_file=None, df=None, mean=None, std=None):
        if df is None:
            self.data = pd.read_csv(csv_file)
        else:
            self.data = df.copy()
        
        self.X = self.data.drop(columns=["MedHouseVal"]).values
        self.y = self.data["MedHouseVal"].values.reshape(-1, 1)
        
        # If mean and std are provided, apply them
        if mean is not None and std is not None:
            self.X = (self.X - mean) / std
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    

# function to get the DataLoader
def get_dataloader(csv_file, batch_size=32, shuffle=True):
    dataset = CaliforniaHousingDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# For checking purposes
# ✅ Quick test to check dataset loading
if __name__ == "__main__":
    dataset = CaliforniaHousingDataset("data/california_housing.csv")
    print(f"Dataset size: {len(dataset)}")
    
    # Fetch a single sample
    sample_X, sample_y = dataset[0]
    print(f"Sample features: {sample_X.shape}, Target: {sample_y.shape}")
    
    # Check DataLoader
    dataloader = get_dataloader("data/california_housing.csv")
    for X_batch, y_batch in dataloader:
        print(f"Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}")