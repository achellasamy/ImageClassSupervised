import torch
import torch.nn as nn
import torch.optim as optim

# Defining the structure of the NN
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        # Input layer size 16 to hidden layer size 16
        self.fc1 = nn.Linear(16, 8)
        # Hidden layer size 16 to output layer size 4
        self.fc2 = nn.Linear(8, 4)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        #Forward through first layer with sigmoid
        x = self.fc1(x)
        x = self.sigmoid(x)

        #Forward through second layer with sigmoid
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
    
model = SimpleNN()
cost = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.2)

# Define multiple input tensors with correct output
input_data = torch.tensor([
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sample 1
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Sample 2
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # Sample 3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # Sample 4
], dtype=torch.float32)

target_output = torch.tensor([
    [1, 0, 0, 0], # Sample 1 Target Output
    [0, 1, 0, 0], # Sample 2 Target Output
    [0, 0, 1, 0], # Sample 3 Target Output
    [0, 0, 0, 1], # Sample 4 Target Output
], dtype=torch.float32)

# Forward Pass
output = model(input_data)

# Compute and print loss
loss = cost(output, target_output)
print(f"Output: {output}")
print(f"Loss: {loss.item()}")

# Parameters
print("Weights and biases of fc1 layer:")
print("Weights:", model.fc1.weight)
print("Biases:", model.fc1.bias)

print("\nWeights and biases of fc2 layer:")
print("Weights:", model.fc2.weight)
print("Biases:", model.fc2.bias)

for i in range(1000):
    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Backward pass
    optimizer.step()  # Update the weights

    # After optimization, recompute the output and loss
    output = model(input_data)  # Recompute output with updated weights
    loss = cost(output, target_output)  # Recompute the loss
    print(f"Output {i}: {output}")
    print(f"Loss {i}: {loss.item()}")

# Test Input 
test_input = torch.tensor([0.98, 0.7, 0.89, 0.99, 0.1, 0.1, 0.3, 0.01, 0.1, 0.1, 0.3, 0.01, 0.1, 0.1, 0.3, 0.01])

test_output = model(test_input)
print(f"Test Output: {test_output}")




