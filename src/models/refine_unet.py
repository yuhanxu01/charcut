import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class RefineUNet(nn.Module):
    """Input: RGB(3) + coarse mask(1) = 4 channels. Output: alpha in [0,1]."""
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.b = DoubleConv(base*2, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.c2 = DoubleConv(base*4, base*2)
        self.c1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        b = self.b(self.p2(d2))
        u2 = self.u2(b)
        c2 = self.c2(torch.cat([u2, d2], dim=1))
        u1 = self.u1(c2)
        c1 = self.c1(torch.cat([u1, d1], dim=1))
        a = torch.sigmoid(self.out(c1))
        return a
