import torch
from torch import nn

class CNNClassifier(nn.Module):
    """A simple CNN classifier for MNIST. Used for evaluating the quality of the generated samples by checking how well they can be classified by a model trained on real data."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)   
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # [B, 32, 28, 28]
        x = self.pool(x)               # [B, 32, 14, 14]
        x = torch.relu(self.conv2(x))  # [B, 64, 14, 14]
        x = self.pool(x)               # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)      # Flatten to [batch_size, 64*7*7]
        x = torch.relu(self.fc1(x))    # [B, 128]
        x = self.fc2(x)                # [B, num_classes]
        return x 