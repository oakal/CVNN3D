import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetConvBlock2d(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(UnetConvBlock2d, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=True),
                )
                setattr(self, "conv%d" % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.ReLU(inplace=True),
                )
                setattr(self, "conv%d" % i, conv)
                in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, "conv%d" % i)
            x = conv(x)
        return x


class UnetConvBlock3d(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        is_batchnorm,
        n=2,
        ks=3,
        stride=1,
        padding=1,
        dilation=2,
    ):
        super(UnetConvBlock3d, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = 1  # padding
        d = 1
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv3d(in_size, out_size, ks, s, p, d),
                    nn.BatchNorm3d(out_size),
                    nn.ELU(alpha=10, inplace=False),
                )
                setattr(self, "conv%d" % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv3d(in_size, out_size, ks, s, p, d),
                    nn.ELU(alpha=10, inplace=False),
                )
                setattr(self, "conv%d" % i, conv)
                in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, "conv%d" % i)
            x = conv(x)
        return x


class UnetDecodingBlock2d(nn.Module):
    def __init__(self, in_size, out_size, is_bnorm, n, is_deconv):
        super(UnetDecodingBlock2d, self).__init__()
        self.conv = UnetConvBlock2d(in_size, out_size, is_bnorm, n)
        if is_deconv:
            self.up = nn.ConvTranspose2d(
                in_size, out_size, kernel_size=4, stride=2, padding=1
            )
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetDecodingBlock3d(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n, is_batchnorm=True):
        super(UnetDecodingBlock3d, self).__init__()
        if is_deconv:
            self.conv = UnetConvBlock3d(in_size, out_size, is_batchnorm, n)
            self.up = nn.ConvTranspose3d(
                in_size,
                out_size,
                kernel_size=(4, 4, 1),
                stride=(2, 2, 1),
                padding=(1, 1, 0),
            )
        else:
            self.conv = UnetConvBlock3d(in_size + out_size, out_size, is_batchnorm, n)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode="trilinear")

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))
