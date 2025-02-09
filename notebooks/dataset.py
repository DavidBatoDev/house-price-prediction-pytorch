import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class CaliforniaHousingDataset(Dataset):
    def __init__(self, csv_file=None, df=None):
        """
        Initialize the dataset by laoding the CSV file. 
        Normalize the input features for better training.
        """
        if df is None:
            self.data = pd.read_csv(csv_file)
        else:
            self.data = df.copy()  # Correctly copy the DataFrame
            
        # Features (X) and (y)
        self.X = self.data.drop(columns=["MedHouseVal"]).values # get all features
        self.y = self.data["MedHouseVal"].values.reshape(-1, 1)    # shape: (num_samples, 1)

        # Normalize feature (mean = 0, std = 1) or also known as standardilization
        self.X = (self.X - np.mean(self.X, axis=0))  / np.std(self.X, axis=0) # formula: (X - mean / std)

        # Convert it to tensors since we are using pytorch
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32) 

    def __len__(self):
        """Return the lenght of the total number of the sample"""
        return len(self.data)
    

    def __getitem__(self, idx):
        """Retrieve a sincel sample from dataset"""
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