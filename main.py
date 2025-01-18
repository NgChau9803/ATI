import os 
import torch.onnx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set random seed for reproducibility
torch.manual_seed(42)

# Define hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 10

# Define the neural network architecture
class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        # Input is 28x28 grayscale image (flattened to 784)
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),  # First hidden layer
            nn.ReLU(),
            nn.Linear(128, 64),        # Second hidden layer
            nn.ReLU(),
            nn.Linear(64, 10)           # Output layer (10 digits)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

def export_model(model, output_path=None):
    # Determine the base path of the project
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # If no output path is specified, use the model folder
    if output_path is None:
        output_path = os.path.join(base_path, 'model', 'mnist_model.onnx')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Set the model to evaluation mode
        model.eval()
        
        # Create a dummy input on the same device as the model
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 1, 28, 28, device=device)
        
        # Move dummy input to CPU for ONNX export
        dummy_input_cpu = dummy_input.cpu()
        model_cpu = model.cpu()
        
        # Export the model
        torch.onnx.export(
            model_cpu,
            dummy_input_cpu,
            output_path,
            export_params=True,
            opset_version=12,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model exported successfully to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error exporting model: {e}")
        raise

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )

    # Initialize model, loss function, and optimizer
    model = SimpleNeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print average loss for the epoch
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}')

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'\nTest Accuracy: {100 * correct / total:.2f}%')

    export_model(model)

if __name__ == "__main__":
    main()