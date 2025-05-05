import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class DownSample(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.conv = DoubleConv(inchannels, outchannels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p
        
class UpSample(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.up = nn.ConvTranspose2d(inchannels, inchannels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(inchannels, outchannels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, inchannels, numclasses):
        super().__init__()
        self.down_conv_1 = DownSample(inchannels, 64)
        self.down_conv_2 = DownSample(64, 128)
        self.down_conv_3 = DownSample(128, 256)
        self.down_conv_4 = DownSample(256, 512)
        
        self.bottle_neck = DoubleConv(512, 1024)

        self.up_conv_1 = UpSample(1024, 512)
        self.up_conv_2 = UpSample(512, 256)
        self.up_conv_3 = UpSample(256, 128)
        self.up_conv_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=numclasses, kernel_size=1)

    def forward(self, x):
        down1, p1 = self.down_conv_1(x)
        down2, p2 = self.down_conv_2(p1)
        down3, p3 = self.down_conv_3(p2)
        down4, p4 = self.down_conv_4(p3)

        b = self.bottle_neck(p4)
        
        up1 = self.up_conv_1(b, down4)
        up2 = self.up_conv_2(up1, down3)
        up3 = self.up_conv_3(up2, down2)
        up4 = self.up_conv_4(up3, down1)

        out = self.out(up4)
        return out


