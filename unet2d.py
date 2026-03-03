import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convnet(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, base=32):
        super().__init__()
        
        self.down1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base*4, base*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.conv4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.conv3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.conv1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))

        b  = self.bottleneck(self.pool4(d4))

        u4 = self.up4(b)
        u4_skip = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4_skip)

        u3 = self.up3(u4)
        u3_skip = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3_skip)

        u2 = self.up2(u3)
        u2_skip = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2_skip)

        u1 = self.up1(u2)
        u1_skip = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1_skip)

        return self.out(u1)  
