"""
PyTorch implementation of UmbraLift, a tone-curve network for low-light image
enhancement.

The network consists of:
- Encoder: Uses dilated convolutions to capture multi-scale features and spatial attention
- Decoder: Uses unpooling and skip connections to reconstruct enhanced image
- Enhancement module: Applies 8 iterations of enhancement operations
- Adaptive gamma correction: Learns optimal gamma value for final enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class spatial_attention(nn.Module):
    def __init__(self):
        super(spatial_attention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class umbra_lift_net(nn.Module):
    """
    UmbraLift tone-curve network for low-light image enhancement.

    A zero-reference model: it trains without paired ground truth, driven
    entirely by losses scoring its own output.
	This network applies a series of convolutional layers and combines features
	through skip connections to enhance image quality. It includes a spatial
    attention module to focus on important image regions.
    The network takes a 3-channel image as input and outputs an enhanced version
    along with the enhancement curves used. The final enhancement includes an
    adaptive gamma correction with a learnable gamma parameter that automatically
    adjusts to optimize the image contrast.
    """
    
    def __init__(self):
        """
        Initialize the network layers including convolutions and activation functions.
        The network uses 32 filters in most layers and concatenates features from
        different levels to preserve spatial information. A spatial attention module
        is applied after initial feature extraction to highlight important regions.
        A learnable gamma parameter is initialized for adaptive gamma correction.
        """
        super(umbra_lift_net, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.spatial_att = spatial_attention()
        self.gamma = nn.Parameter(torch.tensor(0.8))
        number_f = 32  # Number of feature channels

        # Encoder with dilated convolutions for multi-scale feature extraction
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)  # Initial feature extraction
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, dilation=2, padding=2, bias=True)  # Dilated conv for larger receptive field
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, dilation=3, padding=3, bias=True)  # Further increased receptive field
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)  # Final encoder layer

        # Decoder with skip connections for feature fusion
        self.d_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)  # Combines encoder features
        self.d_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)  # Combines encoder features
        self.d_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)  # Outputs 8 sets of 3-channel enhancement curves

        # Pooling layers for downsampling and upsampling
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
                x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width)
                
        Returns:
                tuple: Contains:
                        - enhanced_image (torch.Tensor): The enhanced output image with adaptive gamma correction
                        - r (torch.Tensor): Concatenated enhancement parameters
        """
        # Encoder path with max-pooling
        x1 = self.relu(self.e_conv1(x))

        # Spatial attention module
        x1 = x1 * self.spatial_att(x1)

        x1_pool, idx1 = self.maxpool(x1)

        x2 = self.relu(self.e_conv2(x1_pool))
        x2_pool, idx2 = self.maxpool(x2)

        x3 = self.relu(self.e_conv3(x2_pool))
        x3_pool, idx3 = self.maxpool(x3)

        x4 = self.relu(self.e_conv4(x3_pool))

        # Decoder path with unpooling and skip connections
        x4_unpool = self.unpool(x4, idx3)
        x5 = self.relu(self.d_conv5(torch.cat([x3, x4_unpool], 1)))

        x5_unpool = self.unpool(x5, idx2)
        x6 = self.relu(self.d_conv6(torch.cat([x2, x5_unpool], 1)))

        x6_unpool = self.unpool(x6, idx1)
        x_r = torch.tanh(self.d_conv7(torch.cat([x1, x6_unpool], 1)))

        # Split enhancement curves into 8 iterations
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        # Apply enhancement operations iteratively
        # Each iteration applies a quadratic enhancement curve
        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        x = x + r4*(torch.pow(x,2)-x)	
        x = x + r5*(torch.pow(x,2)-x)	
        x = x + r6*(torch.pow(x,2)-x)
        x = x + r7*(torch.pow(x,2)-x)
        x = x + r8*(torch.pow(x,2)-x)

        # Ensure the pixel values are within [0,1]
        x = torch.clamp(x, 1e-7, 1.0)

        # Constrain gamma to a safe range using sigmoid for adaptive gamma correction
        safe_gamma = 0.5 + torch.sigmoid(self.gamma)  # Range: [0.5, 1.5]

        # Apply adaptive gamma correction with learnable gamma parameter
        enhanced_image = torch.pow(x, safe_gamma)

        # Concatenate all enhancement curves
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], dim=1)

        return enhanced_image, r    