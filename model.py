import math

import torch
#import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor, pixel_mean=[0.4488, 0.4371, 0.4040], pixel_std=[0.2845, 0.2701, 0.2920]):
        pixel_mean = torch.tensor(pixel_mean)
        pixel_std = torch.tensor(pixel_std)
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.normalize = Normalize(pixel_mean, pixel_std)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
        )
        resblocks = [ResidualGroup(20, 64, 16) for _ in range(10)]
        self.resblocks = nn.Sequential(*resblocks)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        upscale_blocks = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
        upscale_blocks.append(nn.Conv2d(64, 3, kernel_size=3, padding=1))
        self.upscale = nn.Sequential(*upscale_blocks)
        self.denormalize = DeNormalize(pixel_mean, pixel_std)

    def forward(self, x):
        x = self.normalize(x)
        residual = self.conv1(x)
        x = self.resblocks(residual)
        x = self.conv2(x)
        x = self.upscale(residual + x)
        x = self.denormalize(x)

        #return (torch.tanh(block34) + 1) / 2
        #return torch.clamp(block34, 0, 1)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.25, mode='fan_out', nonlinearity='leaky_relu')
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = nn.Parameter(mean.view(-1, 1, 1), requires_grad=False)
        self.std = nn.Parameter(std.view(-1, 1, 1), requires_grad=False)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class DeNormalize(nn.Module):
    def __init__(self, mean, std):
        super(DeNormalize, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = nn.Parameter(mean.view(-1, 1, 1), requires_grad=False)
        self.std = nn.Parameter(std.view(-1, 1, 1), requires_grad=False)

    def forward(self, img):
        # denormalize img
        return (img * self.std) + self.mean


class ResidualBlock(nn.Module):
    def __init__(self, channels, reduction):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU(num_parameters=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        self.att_pool = nn.AdaptiveAvgPool2d(1)
        self.att_conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.att_prelu = nn.PReLU(num_parameters=(channels // reduction))
        self.att_conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.att_sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        
        attenuation = self.att_pool(residual)
        attenuation = self.att_conv1(attenuation)
        attenuation = self.att_prelu(attenuation)
        attenuation = self.att_conv2(attenuation)
        attenuation = self.att_sigmoid(attenuation)
        
        return x + residual * attenuation


class ResidualGroup(nn.Module):
    def __init__(self, blocks, channels, reduction):
        super(ResidualGroup, self).__init__()
        resblocks = [ResidualBlock(channels, reduction) for _ in range(blocks)]
        resblocks.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.resblocks = nn.Sequential(*resblocks)

    def forward(self, x):
        group_out = self.resblocks(x)
        
        return x + group_out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x



