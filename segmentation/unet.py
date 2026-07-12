from torch import nn
import torch.nn.functional as F
import torch

# First implement the DoubleConv block, which consists of two convolutional layers with ReLU activations in between. 
# This is a common building block used in UNet architectures.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Pass the input through the double convolutional block.
        
        Args: 
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width)
        """
        return self.conv(x)
    

# Next implement the Downsample block, which consists of a DoubleConv block followed by a max pooling operation.
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """Pass the input through the downsampling block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            tuple: A tuple containing the output tensors from the double convolution and max pooling operations.
                - conv_out (torch.Tensor): Output tensor from the double convolution of shape (batch_size, out_channels, height, width)
                - pool_out (torch.Tensor): Output tensor from the max pooling of shape (batch_size, out_channels, height // 2, width // 2)
        """
        conv_out = self.double_conv(x)  # [batch_size, out_channels, height, width]
        pool_out = self.pool(conv_out)  # [batch_size, out_channels, height // 2, width // 2]
        return conv_out, pool_out
    

# Now implement the Upsample block, which consists of a transposed convolutional layer for upsampling, followed by a DoubleConv block.
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """Pass the input through the upsampling block.

        Args:
            x1 (torch.Tensor): Input tensor from the previous layer of shape (batch_size, in_channels, height, width)
            x2 (torch.Tensor): Input tensor from the skip connection of shape (batch_size, out_channels, height * 2, width * 2)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height * 2, width * 2)
        """
        x1 = self.upconv(x1)  # [batch_size, out_channels, height * 2, width * 2]
        x = torch.cat([x2, x1], dim=1)  # Concatenate along the channel dimension
        return self.double_conv(x)
    

# Finally, implement the UNet class, which combines the Downsample and Upsample blocks to create the full UNet architecture.
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.down1 = Downsample(in_channels, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024, kernel_size=3)
        
        self.up1 = Upsample(1024, 512)
        self.up2 = Upsample(512, 256)
        self.up3 = Upsample(256, 128)
        self.up4 = Upsample(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """Pass the input through the UNet architecture.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width)
        """
        conv1, pool1 = self.down1(x)     # [batch_size, 64, height, width], [batch_size, 64, height // 2, width // 2]
        conv2, pool2 = self.down2(pool1) # [batch_size, 128, height // 2, width // 2], [batch_size, 128, height // 4, width // 4]
        conv3, pool3 = self.down3(pool2) # [batch_size, 256, height // 4, width // 4], [batch_size, 256, height // 8, width // 8]
        conv4, pool4 = self.down4(pool3)    # [batch_size, 512, height // 8, width // 8]

        bottle_neck = self.bottle_neck(pool4)  # [batch_size, 1024, height // 16, width // 16]

        up1 = self.up1(bottle_neck, conv4)  # [batch_size, 512, height // 8, width // 8]
        up2 = self.up2(up1, conv3)    # [batch_size, 256, height // 4, width // 4]
        up3 = self.up3(up2, conv2)    # [batch_size, 128, height // 2, width // 2]
        up4 = self.up4(up3, conv1)    # [batch_size, 64, height, width]

        return self.final_conv(up4)    # [batch_size, out_channels, height, width]