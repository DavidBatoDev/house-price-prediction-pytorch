import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        """
        Initialized the linear regression model.
        :param input_dim: Number of input features
        """
        super(LinearRegressionModel, self).__init__()
        # linear layer performs: y = intercept * W^T + bias
        self.linear_layer = nn.Linear(input_dim, 1) # receive 8 features and 1 output

    def forward(self, x):
        """
        Foward pass of the model
        :param x: Input tensor with shape (batch size, input_dim)
        :return: Output of the tensor wiht shape (batch_size, 1)
        """
        return self.linear_layer(x)
    

#  Test the model functionality when running this script directly
if __name__ == "__main__":
    # For the California Housing dataset, we have 8 features
    input_dim = 8

    # Instantiate the model
    model = LinearRegressionModel(input_dim)

    # Create a sample input (batch of 10 samples)
    sample_input = torch.randn(10, input_dim)
    
    # Get model predictions
    predictions = model(sample_input)

    print(f"Sample input shape: {sample_input.shape}") # 8 Variables
    print(f"Predictions shape: {predictions.shape}") # 1 Output
