import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleConv(nn.Module):
    """
    Double convolution block that keeps that spatial sizes the same
    """

    def __init__(self, in_ch, out_ch, norm_layer=None):
        super(SingleConv, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """
    Double convolution block that keeps that spatial sizes the same
    """

    def __init__(self, in_ch, out_ch, norm_layer=None):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(SingleConv(in_ch, out_ch, norm_layer),
                                  SingleConv(out_ch, out_ch, norm_layer))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """
    Doubles spatial size with bilinear upsampling
    Skip connections and double convs
    """

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        mode = "bilinear"
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        """
        Args:
            x1: [b,c, h, w]
            x2: [b,c, 2*h,2*w]

        Returns: 2x upsampled double conv reselt
        """
        x = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetBottleneck(nn.Module):
    expansion = 4
    #expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out