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

#######################################

class DoubleConv_2(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(DoubleConv_2, self).__init__()
        if batch_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # Bias inutile car sera annulé par natch norm
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # Bias inutile car sera annulé par natch norm
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True), # Bias inutile car sera annulé par natch norm
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True), # Bias inutile car sera annulé par natch norm
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)
    
class UNet_2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], batch_norm=True, kernel_pool_size=2, conv_kernel_size=2):
        super(UNet_2, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=kernel_pool_size, stride=2)


        # Down part of Unet
        for feature in features:
            self.downs.append(DoubleConv_2(in_channels, feature, batch_norm))
            in_channels = feature

        # Up par of Unet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=conv_kernel_size, stride=2
                )
            )
            self.ups.append(
                DoubleConv_2(feature*2, feature, batch_norm)
            )

        self.bottleneck = DoubleConv_2(features[-1], features[-1]*2, batch_norm)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):

        skip_connections = [] 
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNet_2(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
