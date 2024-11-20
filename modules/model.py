import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np

class CrackDetectorCNN(nn.Module):
    def __init__(self):
        super(CrackDetectorCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Single channel input
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, 2)  # Binary classification

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Apply convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten feature maps
        x = x.view(-1, 32 * 56 * 56)  # Adjust to match input size from conv layers

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_model():
    # Initialize the model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrackDetectorCNN() #.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Target all model parameters

    # Define the learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, criterion, optimizer, exp_lr_scheduler

def load_model(model_path, device):
    # Load the model
    model = CrackDetectorCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model