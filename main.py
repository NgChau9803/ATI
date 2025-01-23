import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
import numpy as np
import os

class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        
        # Convolutional Layers with Varied Kernel Sizes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # Larger kernel for more context
        
        # Spatial Attention Mechanism
        self.attention = SpatialAttention()
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Advanced Dropout
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Custom weight initialization to improve convergence
        """
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # First Convolutional Block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Second Convolutional Block with Spatial Attention
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.attention(x)  # Apply spatial attention
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SpatialAttention(nn.Module):
    """
    Spatial Attention Mechanism to focus on important features
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise mean and max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate mean and max
        x_attention = torch.cat([avg_out, max_out], dim=1)
        x_attention = self.conv(x_attention)
        
        # Generate attention map
        return x * self.sigmoid(x_attention)

def train_advanced_model(model_path='model/advanced_mnist_model.pth'):
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 20  # Slightly more epochs

    # Advanced Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),  # Increased rotation
        transforms.RandomAffine(0, 
            translate=(0.1, 0.1),       # Translation
            scale=(0.9, 1.1),           # Slight scaling
            shear=5                     # Shear transformation
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color jittering
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = AdvancedCNN()
    
    # Loss and optimizer with adaptive techniques
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    optimizer = optim.AdamW(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=1e-4)  # Adaptive weight decay
    
    # Learning rate scheduler with cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop with detailed logging
    best_accuracy = 0
    class_accuracies = [0] * 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        correct = [0] * 10
        total = [0] * 10
        overall_correct = 0
        overall_total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                # Per-class accuracy
                for t, p, label in zip(labels, predicted, labels):
                    if t == p:
                        correct[label] += 1
                    total[label] += 1
                
                overall_correct += (predicted == labels).sum().item()
                overall_total += labels.size(0)
        
        # Calculate accuracies
        val_accuracy = 100 * overall_correct / overall_total
        
        # Print per-class accuracies
        print("\nPer-class Accuracies:")
        for i in range(10):
            class_acc = 100 * correct[i] / total[i] if total[i] > 0 else 0
            print(f'Digit {i}: {class_acc:.2f}%')
            class_accuracies[i] = class_acc
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)
            print(f"\nSaved best model with accuracy {best_accuracy:.2f}%")
    
    return model, class_accuracies

def export_model_to_onnx(model_path='model/advanced_mnist_model.pth', 
                          onnx_path='model/advanced_mnist_model.onnx'):
    """
    Export PyTorch model to ONNX format
    """
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    
    # Load the PyTorch model
    model = AdvancedCNN()  # Use the new model class
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create a dummy input
    dummy_input = torch.randn(1, 1, 28, 28)

    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=12
        )
        print(f"Model exported to {onnx_path}")
    except Exception as e:
        print(f"Model export error: {e}")

def main():
    os.makedirs('model', exist_ok=True)
    model, class_accuracies = train_advanced_model()
    export_model_to_onnx()

if __name__ == "__main__":
    main()