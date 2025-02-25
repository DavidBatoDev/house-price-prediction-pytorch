import torch
import torch.nn as nn
import torch.optim as optim


# Linear Regeession
def train_model(model, train_loader, test_loader, device, num_epochs=100, lr=0.01):
    """
    Trains the Linear Regression model using MSE Loss and Adam optimizer.

    Args:
        model: PyTorch model instance.
        train_loader: DataLoader for training set.
        test_loader: DataLoader for testing set.
        device: "cuda" or "cpu".
        num_epochs: Number of training epochs.
        lr: Learning rate for optimizer.

    Returns:
        Trained model.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
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

        # Evaluate model
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

    return model


#  Ridge Regession
def train_model_ridge(model, train_loader, test_loader, device, num_epochs=100, lr=0.01, weight_decay=0.0):
    """
    Train the model with L2 (Ridge) regularization via weight_decay in the optimizer.
    """
    criterion = nn.MSELoss()
    # Note: weight_decay is the L2 penalty coefficient
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
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
            print(f"[Ridge] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    return model


# Lasso Regression
def train_model_lasso(model, train_loader, test_loader, device, num_epochs=100, lr=0.01, lambda_l1=0.0):
    """
    Train the model with L1 (Lasso) regularization by manually adding an L1 penalty to the loss.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            y_preds = model(inputs)
            mse_loss = criterion(y_preds, targets)

            # Compute L1 penalty
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.sum(torch.abs(param))

            # Total loss = MSE + L1 penalty
            loss = mse_loss + lambda_l1 * l1_loss
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
                # Recompute MSE + L1 for consistent test loss
                mse_loss = criterion(y_preds, targets)
                l1_loss = 0
                for param in model.parameters():
                    l1_loss += torch.sum(torch.abs(param))
                total_loss = mse_loss + lambda_l1 * l1_loss
                test_loss += total_loss.item()

        test_loss /= len(test_loader)

        if (epoch+1) % 10 == 0:
            print(f"[Lasso] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    return model

