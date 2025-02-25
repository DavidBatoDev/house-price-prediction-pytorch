import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset import CaliforniaHousingDataset
from model import LinearRegressionModel
from plot import plot_regression_results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Read full data
    csv_file = 'data/california_housing.csv'
    data = pd.read_csv(csv_file)
    
    # Split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # --- Compute scaling from the training data only ---
    train_features = train_data.drop(columns=["MedHouseVal"]).values
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    
    # Create datasets
    train_dataset = CaliforniaHousingDataset(df=train_data, mean=mean, std=std)
    test_dataset  = CaliforniaHousingDataset(df=test_data, mean=mean, std=std)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model, loss, optimizer
    model = LinearRegressionModel(input_dim=8).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Train one epoch
        model.train()
        running_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            y_preds = model(inputs)
            loss = criterion(y_preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                y_preds = model(inputs)
                loss = criterion(y_preds, targets)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    plot_regression_results(model, test_loader, device)

if __name__ == "__main__":
    main()
