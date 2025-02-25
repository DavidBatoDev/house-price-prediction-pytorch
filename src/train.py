import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from plot import plot_regression_results

from dataset import CaliforniaHousingDataset, get_dataloader
from model import LinearRegressionModel

# Create a function to train one epoch
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    :params 
        model - the model that we are going to use to train the dataset and make predictions
        dataloader - uses the dataloader of the  
    """

    model.train() # training mode
    running_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # put the inputs and the target to the targets
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Do the foward pass
        y_preds = model(inputs) # foward pass to predict
        loss = criterion(y_preds, targets) # Calculates the loss between the predition and the pred

        # Performs backpropagation
        loss.backward()

        # Step the optimizer
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def main():
    # Set the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_file = 'data/california_housing.csv'

    # Perform train test split
    data = pd.read_csv(csv_file)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Create dataset objects directly from the DataFrames
    train_dataset = CaliforniaHousingDataset(df=train_data)
    test_dataset = CaliforniaHousingDataset(df=test_data)


    input_dim = 8 # Our model has 8 features
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instatiate the model
    model = LinearRegressionModel(input_dim).to(device)

    # loss function
    criterion = nn.MSELoss()

    # optimzer: we will use SDG for this
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Perform Training loop on the test dataset
    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Evaluate on the test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                y_preds = model(inputs)
                loss = criterion(y_preds, targets)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
    plot_regression_results(model, test_loader, device)

if __name__ == "__main__":
    main()
