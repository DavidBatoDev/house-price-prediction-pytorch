import matplotlib.pyplot as plt
import torch

def plot_regression_results(model, dataloader, device):
    model.eval()
    actuals = []
    predictions = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            y_preds = model(inputs).cpu().numpy()
            actuals.extend(targets.cpu().numpy())
            predictions.extend(y_preds)

    # Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, predictions, alpha=0.5, label="Predicted vs Actual")
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r', label="Perfect Fit")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression Predictions")
    plt.legend()
    plt.grid()
    plt.show()
