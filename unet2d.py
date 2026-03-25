import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Apply two convolution-BN-ReLU blocks in sequence."""

    def __init__(self, in_ch, out_ch):
        """Initialize the double-convolution block."""
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
        """Apply the stacked convolutional block to the input tensor."""
        return self.convnet(x)


class UNet2D(nn.Module):
    """Implement a 2D U-Net with a replaceable output head."""

    def __init__(self, in_channels=4, num_classes=4, base=32, head_channels=None):
        """Initialize the encoder, decoder, and initial prediction head.

        Args:
            in_channels (int): Number of MRI input channels.
            num_classes (int): Default number of output channels/classes.
            base (int): Base feature width for the network.
            head_channels (int | None): Optional override for head output width.
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base = base
        self.head_channels = num_classes if head_channels is None else head_channels

        self.down1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, self.head_channels, 1)

    def encode(self, x):
        """Encode the input and return the bottleneck plus skip connections."""
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        bottleneck = self.bottleneck(self.pool4(d4))
        return bottleneck, (d1, d2, d3, d4)

    def decode(self, bottleneck, skips):
        """Decode bottleneck features using the stored U-Net skip connections."""
        d1, d2, d3, d4 = skips

        u4 = self.up4(bottleneck)
        u4 = self.conv4(torch.cat([u4, d4], dim=1))

        u3 = self.up3(u4)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))
        return u1

    def forward_features(self, x):
        """Run the encoder-decoder path and return decoder features."""
        bottleneck, skips = self.encode(x)
        return self.decode(bottleneck, skips)

    def forward(self, x):
        """Generate task-specific predictions from the input tensor."""
        features = self.forward_features(x)
        return self.head(features)

    def replace_head(self, out_channels):
        """Swap the prediction head for a new output dimensionality."""
        self.head_channels = out_channels
        self.head = nn.Conv2d(self.base, out_channels, 1)
        return self
