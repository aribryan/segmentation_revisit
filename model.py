# Implement a very simple skip connection powered Unet
import torch
from torch import nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up_scale = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x1, x2):
        x2 = self.up_scale(x2)

        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


class down_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_layer, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return x


class up_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_layer, self).__init__()
        self.up = up(in_ch, out_ch)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        a = self.up(x1, x2)
        x = self.conv(a)
        return x



class simpleUNet(nn.Module):
    def __init__(self, args):

        self.dimensions=args.dimensions
        super(simpleUNet, self).__init__()
        self.conv1 = double_conv(3, self.dimensions)
        self.down1 = down_layer(self.dimensions, self.dimensions*2)
        self.down2 = down_layer(self.dimensions*2, self.dimensions*4)
        self.down3 = down_layer(self.dimensions*4, self.dimensions*8)
        self.down4 = down_layer(self.dimensions*8, self.dimensions*16)
        self.up1 = up_layer(self.dimensions*16, self.dimensions*8)
        self.up2 = up_layer(self.dimensions*8, self.dimensions*4)
        self.up3 = up_layer(self.dimensions*4, self.dimensions*2)
        self.up4 = up_layer(self.dimensions*2, self.dimensions)
        self.last_conv = nn.Conv2d(self.dimensions, 1, 1) # use 1x1

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1_up = self.up1(x4, x5)
        x2_up = self.up2(x3, x1_up)
        x3_up = self.up3(x2, x2_up)
        x4_up = self.up4(x1, x3_up)
        output = self.last_conv(x4_up)
        return output

