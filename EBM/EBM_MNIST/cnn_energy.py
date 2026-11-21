from torch import nn
import torch

# Define the Swish activation function
class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class CNNEnergy(nn.Module):
    """A simple CNN-based energy function for MNIST images.
      The output is a scalar energy value for each input image."""

    def __init__(self, num_channels_first_layer=16):
        super().__init__()
        # We increase the number of channels over layers. Here pre-calculated for simplicity.
        c_hid1 = num_channels_first_layer
        c_hid2 = num_channels_first_layer * 2
        c_hid3 = num_channels_first_layer * 4

        # Series of convolutions and Swish activation functions
        self.conv_layers = nn.Sequential(
                nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4), # [16x16] - Padding of 4 to maintain spatial dimensions after stride 2
                Swish(),
                nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1), #  [8x8]
                Swish(),
                nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1), # [4x4]
                Swish(),
                nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1), # [2x2]
                Swish()
        )

        # Linear layers to output a single scalar energy value
        self.linear_layers = nn.Sequential(
                nn.Linear(c_hid3 * 2 * 2, 128),
                Swish(),
                nn.Linear(128, 1) # Output a single scalar energy value
        )

    def forward(self, x):
        """Forward pass through the CNN to compute energy.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 1, 28, 28).
            
        Returns:
            torch.Tensor: Energy values of shape (batch_size, 1).
        """ 
        
        batch_size = x.size(0)
        
        x = self.conv_layers(x)  # Output shape will be (batch_size, c_hid3, 2, 2)
        x = x.view(batch_size, -1)  # Flatten the tensor
        x = self.linear_layers(x)  # Output shape will be (batch_size, 1)
        
        return x