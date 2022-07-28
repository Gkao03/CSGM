from pyrsistent import inc
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1, scale_factor=2, norm='batch'):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, norm='batch', init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim):
        super(DCGenerator, self).__init__()
        # self.up_conv1 = nn.Conv2d(in_channels=noise_size, out_channels=conv_dim * 8, kernel_size=4, stride=1, padding=3)
        # self.inorm1= nn.InstanceNorm2d(conv_dim)
        # self.relu1 = nn.ReLU()
        # self.up_conv2 = up_conv(in_channels=conv_dim * 8, out_channels=conv_dim * 4, kernel_size=4, padding='same', norm='instance')
        # self.relu2 = nn.ReLU()
        # self.up_conv3 = up_conv(in_channels=conv_dim * 4, out_channels=conv_dim * 2, kernel_size=4, padding='same', norm='instance')
        # self.relu3 = nn.ReLU()
        # self.up_conv4 = up_conv(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=4, padding='same', norm='instance')
        # self.relu4 = nn.ReLU()
        # self.up_conv5 = up_conv(in_channels=conv_dim, out_channels=3, kernel_size=4, padding='same', norm='instance')
        # self.tanh5 = nn.Tanh()

        # attempt 2 architecture
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_size, conv_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(conv_dim * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(conv_dim * 8, conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(conv_dim * 4, conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(conv_dim * 2, conv_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(conv_dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        """Generates an image given a sample of random noise.
            Input
            -----
                z: BS x noise_size x 1 x 1   -->  16x100x1x1
            Output
            ------
                out: BS x channels x image_width x image_height  -->  16x3x64x64
        """
        # out = self.up_conv1(z)
        # out = self.inorm1(out)
        # out = self.relu1(out)
        # out = self.up_conv2(out)
        # out = self.relu2(out)
        # out = self.up_conv3(out)
        # out = self.relu3(out)
        # out = self.up_conv4(out)
        # out = self.relu4(out)
        # out = self.up_conv5(out)
        # out = self.tanh5(out)

        # attempt 2 forward
        out = self.main(z)

        return out


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim, norm):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1, norm=norm)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture.
    """
    def __init__(self, conv_dim=64, init_zero_weights=False, norm='batch'):
        super(CycleGenerator, self).__init__()

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=4, stride=2, padding=1, norm=norm, init_zero_weights=init_zero_weights)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4, stride=2, padding=1, norm=norm, init_zero_weights=init_zero_weights)
        self.relu2 = nn.ReLU()

        # # 2. Define the transformation part of the generator
        self.resnet_block1 = ResnetBlock(conv_dim=conv_dim * 2, norm='batch')
        self.resnet_block2 = ResnetBlock(conv_dim=conv_dim * 2, norm='batch')
        self.resnet_block3 = ResnetBlock(conv_dim=conv_dim * 2, norm='batch')
        self.inorm_res = nn.InstanceNorm2d(num_features=conv_dim * 2)
        self.relu_res = nn.ReLU()

        # # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.up_conv1 = up_conv(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=4, padding='same', norm=norm)
        self.relu3 = nn.ReLU()
        self.up_conv2 = up_conv(in_channels=conv_dim, out_channels=3, kernel_size=4, padding='same', norm='none')
        self.tanh = nn.Tanh()

    def forward(self, x):
        """Generates an image conditioned on an input image.
            Input
            -----
                x: BS x 3 x 32 x 32
            Output
            ------
                out: BS x 3 x 32 x 32
        """
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.resnet_block1(out)
        out = self.resnet_block2(out)
        out = self.resnet_block3(out)
        out = self.inorm_res(out)
        out = self.relu_res(out)
        out = self.up_conv1(out)
        out = self.relu3(out)
        out = self.up_conv2(out)
        out = self.tanh(out)

        return out


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture.
    """
    def __init__(self, conv_dim=64, norm='batch'):
        super(DCDiscriminator, self).__init__()

        d = 1  # dilation
        k = 4  # kernel
        s = 2  # stride
        p = 1  # padding


        # self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=k, stride=s, padding=p, norm=norm)
        # self.relu1 = nn.ReLU()
        # self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=k, stride=s, padding=p, norm=norm)
        # self.relu2 = nn.ReLU()
        # self.conv3 = conv(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=k, stride=s, padding=p, norm=norm)
        # self.relu3 = nn.ReLU()
        # self.conv4 = conv(in_channels=conv_dim * 4, out_channels=conv_dim * 8, kernel_size=k, stride=s, padding=p, norm=norm)
        # self.relu4 = nn.ReLU()
        # self.conv5 = conv(in_channels=conv_dim * 8, out_channels=1, kernel_size=k, stride=s, padding=0, norm='none')
        
        # 2nd model
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=conv_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.conv2 = conv(in_channels=conv_dim, out_channels=2*conv_dim, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.conv3 = conv(in_channels=2*conv_dim, out_channels=4*conv_dim, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv4 = conv(in_channels=4*conv_dim, out_channels=8*conv_dim, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv5 = nn.Conv2d(in_channels=8*conv_dim, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.conv5(out)
        out = self.sigmoid(out)

        return out


class PatchDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, norm='batch'):
        super().__init__()
        d = 1  # dilation
        k = 4  # kernel
        s = 2  # stride
        p = 1  # padding


        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=k, stride=s, padding=p, norm=norm)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=k, stride=s, padding=p, norm=norm)
        self.relu2 = nn.ReLU()
        self.conv3 = conv(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=k, stride=s, padding=p, norm=norm)
        self.relu3 = nn.ReLU()
        self.conv4 = conv(in_channels=conv_dim * 4, out_channels=1, kernel_size=k, stride=s, padding=p, norm='none')

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)

        return out