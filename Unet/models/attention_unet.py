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
    
class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        
        self.w_g = nn.Sequential(
                                nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )
        
        self.w_x = nn.Sequential(
                                nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )
        
        self.psi = nn.Sequential(
                                nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0,  bias=True),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        
        return psi*x
        
    
class Attention_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], batch_norm=True, kernel_pool_size=2, conv_kernel_size=2):
        super(Attention_UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=kernel_pool_size, stride=2)


        # Down part of Unet
        for feature in features:
            self.downs.append(DoubleConv_2(in_channels, feature, batch_norm))
            in_channels = feature

        # Up part of Unet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=conv_kernel_size, stride=2
                )
            )
            self.ups.append(
                AttentionBlock(f_g=feature, f_l=feature, f_int=feature//2
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

        for idx in range(0, len(self.ups), 3):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//3]
            skip_connection = self.ups[idx+1](x, skip_connection)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+2](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 160, 160))
    model = Attention_UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
