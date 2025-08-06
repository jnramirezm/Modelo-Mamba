import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mamba_block import MambaBlock
from model.unet import DoubleConv

class UNetMamba(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, mode='enc', strategy='integrate'):
        super().__init__()
        assert mode in ['enc', 'dec', 'full'], "mode must be 'enc', 'dec' or 'full'"
        assert strategy in ['integrate', 'replace'], "strategy must be 'integrate' or 'replace'"

        def wrap(block, use_mamba):
            if not use_mamba:
                return block
            if strategy == 'replace':
                return MambaBlock(d_model=block.double_conv[0].out_channels)
            elif strategy == 'integrate':
                return nn.Sequential(block, MambaBlock(d_model=block.double_conv[0].out_channels))

        # Encoder
        self.down1 = wrap(DoubleConv(in_channels, 64), mode in ['enc', 'full'])
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = wrap(DoubleConv(64, 128), mode in ['enc', 'full'])
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = wrap(DoubleConv(128, 256), mode in ['enc', 'full'])
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = wrap(DoubleConv(512, 256), mode in ['dec', 'full'])
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = wrap(DoubleConv(256, 128), mode in ['dec', 'full'])
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = wrap(DoubleConv(128, 64), mode in ['dec', 'full'])

        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))

        bn = self.bottleneck(self.pool3(d3))

        u3 = self.up3(bn)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.output(u1)
