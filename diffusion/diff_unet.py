## Create a Diffusion UNet model for image generation
## The architecture is that of a regular U-Net with skip connections, but with the addition of a timestep embedding that is added to the feature maps at each level of the U-Net.
import torch
from torch import nn

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
    

# Create a sinusoidal positional embedding for the timesteps, which will be added to the feature maps at each level of the U-Net. This is similar to the positional embeddings used in transformers, but adapted for 2D images.
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        """Generate sinusoidal positional embeddings for the given timesteps.
        
        Args:
            timesteps (torch.Tensor): A tensor of shape (batch_size,) containing the timesteps for each sample in the batch.
        
        Returns:
            torch.Tensor: A tensor of shape (batch_size, embedding_dim) containing the positional embeddings for each sample in the batch.
        """
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / half_dim))
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
# Create a Downsample block, which consists of a DoubleConv block followed by a max pooling operation. It 
# also takes the timestep embedding as input and adds it to the feature maps after the DoubleConv block.
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super(Downsample, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.embedding_proj = nn.Linear(time_embedding_dim, out_channels)

    def forward(self, x, t_emb):
        """Pass the input through the downsampling block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            t_emb (torch.Tensor): Timestep embedding tensor of shape (batch_size, time_embedding_dim)
        
        Returns:
            tuple: A tuple containing the output tensors from the double convolution and max pooling operations.
                - conv_out (torch.Tensor): Output tensor from the double convolution of shape (batch_size, out_channels, height, width)
                - pool_out (torch.Tensor): Output tensor from the max pooling of shape (batch_size, out_channels, height // 2, width // 2)
        """
        conv_out = self.double_conv(x)  # [batch_size, out_channels, height, width]
        # Project the timestep embedding to the same number of channels as the feature maps and add it to the feature maps.
        # This allows the model to condition on the timestep information at each level of the U-Net.
        t_emb_proj = self.embedding_proj(t_emb)
        # Reshape t_emb_proj to have the same spatial dimensions as conv_out so that we can add them together. We can use broadcasting for this.
        t_emb_proj = t_emb_proj[:, :, None, None]  # [batch_size, out_channels] -> [batch_size, out_channels, 1, 1]

        conv_out = conv_out + t_emb_proj  # Broadcast addition
        pool_out = self.pool(conv_out)  # [batch_size, out_channels, height // 2, width // 2]
        return conv_out, pool_out
    

# Now implement the Upsample block, which consists of a transposed convolutional layer for upsampling, followed by a DoubleConv block.
# We also add the timestep embedding to the feature maps before upsampling, similar to the Downsample block.
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super(Upsample, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.time_embedding_proj = nn.Linear(time_embedding_dim, in_channels)

    def forward(self, x1, x2, t_emb):
        """Pass the input through the upsampling block.

        Args:
            x1 (torch.Tensor): Input tensor from the previous layer of shape (batch_size, in_channels, height, width)
            x2 (torch.Tensor): Input tensor from the skip connection of shape (batch_size, out_channels, height * 2, width * 2)
            t_emb (torch.Tensor): Timestep embedding tensor of shape (batch_size, time_embedding_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height * 2, width * 2)
        """
        # Project the timestep embedding to the same number of channels as x1 and add it to x1 before upsampling.
        t_emb_proj = self.time_embedding_proj(t_emb)
        # Reshape t_emb_proj to have the same spatial dimensions as x1 so that we can add them together. We can use broadcasting for this.
        t_emb_proj = t_emb_proj[:, :, None, None]  # [batch_size, in_channels] -> [batch_size, in_channels, 1, 1]

        x1 = x1 + t_emb_proj  # Broadcast addition

        x1 = self.upconv(x1)  # [batch_size, out_channels, height * 2, width * 2]
        x = torch.cat([x2, x1], dim=1)  # Concatenate along the channel dimension
        return self.double_conv(x)
    
# Finally, implement the UNet class, which combines the Downsample and Upsample blocks to create the full UNet architecture.
class DiffUNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim=128):
        super(DiffUNet, self).__init__()

        # Create a sinusoidal positional embedding for the timesteps,
        # which will be added to the feature maps at each level of the U-Net.
        self.time_embedding = SinusoidalPositionalEmbedding(time_embedding_dim)

        # Create the downsampling and upsampling blocks for the U-Net architecture. We will use 4 levels of downsampling and upsampling, with a bottleneck in between.
        self.down1 = Downsample(in_channels, 64, time_embedding_dim)
        self.down2 = Downsample(64, 128, time_embedding_dim)
        self.down3 = Downsample(128, 256, time_embedding_dim)
        self.down4 = Downsample(256, 512, time_embedding_dim)

        self.bottle_neck = DoubleConv(512, 1024, kernel_size=3)
        
        self.up1 = Upsample(1024, 512, time_embedding_dim)
        self.up2 = Upsample(512, 256, time_embedding_dim)
        self.up3 = Upsample(256, 128, time_embedding_dim)
        self.up4 = Upsample(128, 64, time_embedding_dim)
        
        # Final convolutional layer to map the output of the last upsampling block to the desired number of output channels (e.g. 3 for RGB images).
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, t):
        """Pass the input through the UNet architecture.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            t (torch.Tensor): Timestep tensor of shape (batch_size,)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Generate the timestep embeddings for the input timesteps.
        t_emb = self.time_embedding(t)  # [batch_size, time_embedding_dim]

        conv1, pool1 = self.down1(x, t_emb)     # [batch_size, 64, height, width], [batch_size, 64, height // 2, width // 2]
        conv2, pool2 = self.down2(pool1, t_emb) # [batch_size, 128, height // 2, width // 2], [batch_size, 128, height // 4, width // 4]
        conv3, pool3 = self.down3(pool2, t_emb) # [batch_size, 256, height // 4, width // 4], [batch_size, 256, height // 8, width // 8]
        conv4, pool4 = self.down4(pool3, t_emb)    # [batch_size, 512, height // 8, width // 8]

        bottle_neck = self.bottle_neck(pool4)  # [batch_size, 1024, height // 16, width // 16]

        up1 = self.up1(bottle_neck, conv4, t_emb)  # [batch_size, 512, height // 8, width // 8]
        up2 = self.up2(up1, conv3, t_emb)    # [batch_size, 256, height // 4, width // 4]
        up3 = self.up3(up2, conv2, t_emb)    # [batch_size, 128, height // 2, width // 2]
        up4 = self.up4(up3, conv1, t_emb)    # [batch_size, 64, height, width]

        return self.final_conv(up4)    # [batch_size, out_channels, height, width]