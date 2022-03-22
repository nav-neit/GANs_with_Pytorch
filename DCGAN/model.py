# DCGAN Architecture
"""
Discriminator and Generator model implementation
"""
import torch
import torch.nn as nn

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        # channels_img - input images
        # features_d - channels that change as we go through the discriminator
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input shape : N * channels_img * 64 * 64
            nn.Conv2d(
                channels_img,
                features_d,
                kernel_size=4,
                stride=2,
                padding=1
            ),  # we don't use batch normalization just after the first layer
            nn.LeakyReLU(0.2),
            # 32 * 32
            self._block(features_d, features_d * 2, 4, 2, 1),
            # 16 * 16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            # 8 * 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # 4 * 4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            # 1 * 1
            nn.Sigmoid()  # A single value between 0 and 1
            # classifies the image as real or fake.
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,  # set bias=False since we are using batch normalization
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

# Generator model
class Generator(nn.Module):

    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # we use batch norm for first layer
            # Input dimension : N * z_dim * 1* 1
            self._block(z_dim, features_g*16, 4, 1, 0),
            # 4* 4
            self._block(features_g*16,features_g*8, 4, 2, 1),
            # 8 * 8
            self._block(features_g * 8, features_g*4, 4, 2, 1),
            # 16 * 16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            # 32 * 32
            # for final layer we don't want batch normalization
            nn.ConvTranspose2d(
                features_g*2, channels_img, kernel_size=4, stride=2, padding=1,
            ),
            nn.Tanh(), # values of all pixels in range [-1,1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            # Adding transpose convolutional layer
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)

# Initialize the weights
def initialize_weights(model):
    # mean 0 and std dev 0.02
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64 # RGB channels
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W)
