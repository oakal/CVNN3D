from blocks import *

from layers import *


class CVNN2D(nn.Module):
    def __init__(
        self, nIter, eta, epsilon, update, UNet=False, nPoly=1, multiPoly=False
    ):
        super(CVNN2D, self).__init__()
        self.eta = eta
        self.update = update
        self.nIter = nIter
        self.nPoly = nPoly
        self.multiPoly = multiPoly
        k1 = 3
        if self.multiPoly:
            if not UNet:
                for i in range(1, nPoly + 1):
                    layer = ConvBlock(k1)
                    setattr(self, "layer%d" % i, layer)
        else:
            if not UNet:
                self.layer1 = ConvBlock(k1)
            else:
                self.layer1 = UNetGenerator2D(
                    feature_scale=UNet[0], nConv=UNet[1], nBlocks=UNet[2]
                )

        self.eta = eta
        self.etaP = 1.0

        if self.update == "phi_etadelta_kappa_mu":
            self.phi_etadelta_kappa_mu = phi_etadelta_kappa_mu_Layer(epsilon)
        elif self.update == "phi_etakappa_mu":
            self.phi_etakappa_mu = phi_etakappa_mu_Layer(epsilon)
        elif self.update == "phi_etakappa":
            self.phi_etakappa = phi_etakappa_Layer()
        elif self.update == "etakappa":
            self.etakappa = etakappa_Layer()

    def forward(self, phi_0, pbhist, img):
        phi = {0: phi_0}
        kappa = {}

        for i in range(self.nIter):
            kappa[i + 1, 1] = self.layer1(phi[i])
            curvature = kappa[i + 1, 1]
            for j in range(2, self.nPoly + 1):
                if self.multiPoly:
                    conv = getattr(self, "layer%d" % j)
                    kappa[i + 1, j] = conv(kappa[i + 1, j - 1])
                else:
                    kappa[i + 1, j] = self.layer1(kappa[i + 1, j - 1])
                curvature = curvature + kappa[i + 1, j]  # *self.eta**(j-1)

            if self.update == "phi_etadelta_kappa_mu":
                phi[i + 1] = self.phi_etadelta_kappa_mu(
                    curvature, phi[i], pbhist, self.eta, i
                )
            elif self.update == "phi_etakappa_mu":
                phi[i + 1] = self.phi_etakappa_mu(
                    curvature, phi[i], pbhist, self.eta * (self.etaP**i), i
                )
            elif self.update == "phi_etakappa":
                phi[i + 1] = self.phi_etakappa(curvature, phi[i], self.eta)
            if self.update == "etakappa":
                phi[i + 1] = self.etakappa(curvature, self.eta)

        return phi


class ConvBlock(nn.Module):
    def __init__(self, k1):
        super(ConvBlock, self).__init__()
        k = 3
        p = (k - 1) // 2
        self.layer = nn.Sequential(
            nn.Conv2d(1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
            nn.ELU(alpha=1.0, inplace=False),
            nn.Conv2d(k1, 1, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return self.layer(x)


class CVNN3D(nn.Module):
    def __init__(
        self, nIter, eta, epsilon, UNet=False, nPoly=1, multiPoly=False, version=1
    ):
        super(CVNN3D, self).__init__()
        self.eta = eta
        self.nIter = nIter
        self.nPoly = nPoly
        self.multiPoly = multiPoly
        self.version = version
        k1 = 3
        if self.multiPoly:
            if not UNet:
                for i in range(1, nPoly + 1):
                    layer = ConvBlock3D(k1)
                    setattr(self, "layer%d" % i, layer)
        else:
            if not UNet:
                self.layer1 = ConvBlock3D(k1)
            else:
                self.layer1 = UNetGenerator3D(
                    feature_scale=UNet[0], nConv=UNet[1], nBlocks=UNet[2]
                )

        self.eta = eta
        self.etaP = 1.0

        if self.version == 1:
            self.phi_etakappa_mu = PhiEtaKappaMu3DLayer(epsilon)
        elif self.version == 2:
            self.phi_eta_mu = PhiEtaMu3DLayer(epsilon)
        else:
            raise NotImplementedError

    def forward(self, phi_0, pbhist, img):

        if self.version == 1:
            phi = {0: phi_0}
            kappa = {}
            for i in range(self.nIter):
                kappa[i + 1, 1] = self.layer1(phi[i])
                curvature = kappa[i + 1, 1]
                for j in range(2, self.nPoly + 1):
                    if self.multiPoly:
                        conv = getattr(self, "layer%d" % j)
                        kappa[i + 1, j] = conv(kappa[i + 1, j - 1])
                    else:
                        kappa[i + 1, j] = self.layer1(kappa[i + 1, j - 1])
                    curvature = curvature + kappa[i + 1, j]

                phi[i + 1] = self.phi_etakappa_mu(
                    curvature, phi[i], pbhist, self.eta * (self.etaP**i), i
                )

        elif self.version == 2:
            phi = {0: phi_0}
            kappa = {}

            for i in range(self.nIter):

                phi_pt5 = self.phi_eta_mu(
                    phi[i], pbhist, self.eta * (self.etaP**i), i
                )
                kappa[i + 1, 1] = self.layer1(phi_pt5)
                segmentation = kappa[i + 1, 1]

                for j in range(2, self.nPoly + 1):
                    if self.multiPoly:
                        conv = getattr(self, "layer%d" % j)
                        kappa[i + 1, j] = conv(kappa[i + 1, j - 1])
                    else:
                        kappa[i + 1, j] = self.layer1(kappa[i + 1, j - 1])
                    segmentation = segmentation + kappa[i + 1, j]
                phi[i + 1] = segmentation / self.nPoly

        return phi


class ConvBlock3D(nn.Module):
    def __init__(self, k1):
        super(ConvBlock3D, self).__init__()
        k = 3
        d = 1
        p = d * (k - 1) // 2
        self.layer = nn.Sequential(
            nn.Conv3d(1, k1, kernel_size=k, stride=1, padding=p, dilation=d),
            nn.Conv3d(k1, k1, kernel_size=k, stride=1, padding=p, dilation=d),
            nn.Conv3d(k1, k1, kernel_size=k, stride=1, padding=p, dilation=d),
            nn.Conv3d(k1, k1, kernel_size=1, stride=1),
            nn.ELU(alpha=1.0, inplace=False),
            nn.Conv3d(k1, 1, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return self.layer(x)


class UNetGenerator3D(nn.Module):
    def __init__(
        self,
        feature_scale=1,
        nConv=3,
        nBlocks=4,
        is_batchnorm=False,
        n_classes=1,
        in_channels=1,
    ):
        super(UNetGenerator3D, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n = nConv
        self.nBlocks = nBlocks
        self.is_deconv = True
        x = 64 // self.feature_scale
        filters = [
            x * 2**p if p > -1 else in_channels for p in range(-1, nBlocks + 1)
        ]  # in_channels is filters[0]
        # encoding
        for i in range(1, nBlocks + 1):
            encode = UnetConvBlock3d(
                filters[i - 1], filters[i], self.is_batchnorm, self.n
            )
            setattr(self, "conv%d" % i, encode)
            maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1))
            setattr(self, "maxpool%d" % i, maxpool)

        # BottleNeck
        self.center = UnetConvBlock3d(
            filters[nBlocks], filters[nBlocks + 1], self.is_batchnorm, self.n
        )

        # Decoding
        for i in range(nBlocks, 0, -1):
            decode = UnetDecodingBlock3d(
                filters[i + 1], filters[i], self.is_batchnorm, self.n, self.is_deconv
            )
            setattr(self, "up%d" % i, decode)

        # final conv
        self.final = nn.Conv3d(filters[1], n_classes, 1)

    def forward(self, inputs):
        Conv = {}
        MaxPool = {0: inputs}
        Up = {}
        for i in range(1, self.nBlocks + 1):
            conv = getattr(self, "conv%d" % i)
            Conv[i] = conv(MaxPool[i - 1])
            maxpool = getattr(self, "maxpool%d" % i)
            MaxPool[i] = maxpool(Conv[i])

        Up[self.nBlocks + 1] = self.center(MaxPool[self.nBlocks])

        for i in range(self.nBlocks, 0, -1):
            up = getattr(self, "up%d" % i)
            Up[i] = up(Conv[i], Up[i + 1])

        final = self.final(Up[1])

        return final


class UNetGenerator2D(nn.Module):
    def __init__(
        self,
        feature_scale=1,
        nConv=3,
        nBlocks=4,
        is_batchnorm=False,
        n_classes=1,
        in_channels=1,
    ):
        super(UNetGenerator2D, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n = nConv
        self.nBlocks = nBlocks
        self.is_deconv = True
        x = 64 // self.feature_scale
        filters = [
            x * 2**p if p > -1 else in_channels for p in range(-1, nBlocks + 1)
        ]  # in_channels is filters[0]
        # encoding
        for i in range(1, nBlocks + 1):
            encode = UnetConvBlock2d(
                filters[i - 1], filters[i], self.is_batchnorm, self.n
            )
            setattr(self, "conv%d" % i, encode)
            maxpool = nn.MaxPool2d(kernel_size=2)
            setattr(self, "maxpool%d" % i, maxpool)

        # BottleNeck
        self.center = UnetConvBlock2d(
            filters[nBlocks], filters[nBlocks + 1], self.is_batchnorm, self.n
        )

        # Decoding
        for i in range(nBlocks, 0, -1):
            decode = UnetDecodingBlock2d(
                filters[i + 1], filters[i], self.is_batchnorm, self.n, self.is_deconv
            )
            setattr(self, "up%d" % i, decode)

        # final conv
        self.final = nn.Conv2d(filters[1], n_classes, 1)

    def forward(self, inputs):
        Conv = {}
        MaxPool = {0: inputs}
        Up = {}
        for i in range(1, self.nBlocks + 1):
            conv = getattr(self, "conv%d" % i)
            Conv[i] = conv(MaxPool[i - 1])
            maxpool = getattr(self, "maxpool%d" % i)
            MaxPool[i] = maxpool(Conv[i])

        Up[self.nBlocks + 1] = self.center(MaxPool[self.nBlocks])

        for i in range(self.nBlocks, 0, -1):
            up = getattr(self, "up%d" % i)
            Up[i] = up(Conv[i], Up[i + 1])

        final = self.final(Up[1])

        return final


class CVNN_ICIP(nn.Module):
    def __init__(self, nIter, eta, Epsilon, update):
        super(CVNN_ICIP, self).__init__()
        self.eta = eta
        self.update = update
        self.nIter = nIter
        k = 7
        p = 3
        k1 = 7
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(k1, 1, kernel_size=1, stride=1),
        )
        if self.update == "phi_etadelta_kappa_mu":
            self.phi_etadelta_kappa_mu = phi_etadelta_kappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa_mu":
            self.phi_etakappa_mu = phi_etakappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa":
            self.phi_etakappa = phi_etakappa_Layer()
        elif self.update == "etakappa":
            self.etakappa = etakappa_Layer()

    def forward(self, phi_0, pbhist):
        phi = {0: phi_0}
        kappa = {}
        for i in range(self.nIter):
            kappa[i] = self.layer1(phi[i])
            if self.update == "phi_etadelta_kappa_mu":
                phi[i + 1] = self.phi_etadelta_kappa_mu(
                    kappa[i], phi[i], pbhist, self.eta
                )
            elif self.update == "phi_etakappa_mu":
                phi[i + 1] = self.phi_etakappa_mu(
                    kappa[i], phi[i], pbhist, self.eta * (0.5 ** (max(0, i - 2))), i
                )
            elif self.update == "phi_etakappa":
                phi[i + 1] = self.phi_etakappa(kappa[i], phi[i], self.eta)
            if self.update == "etakappa":
                phi[i + 1] = self.etakappa(kappa[i], self.eta)

        return phi[self.nIter]


# the rest are stale
class CVNN_Poly(nn.Module):
    def __init__(
        self, nIter, eta, Epsilon, update, nPoly=1, multiPoly=False, UNet=False
    ):
        super(CVNN_Poly, self).__init__()
        self.eta = eta
        self.update = update
        self.nIter = nIter
        self.nPoly = nPoly
        self.multiPoly = multiPoly
        k1 = 3
        if self.multiPoly == True:
            if UNet == False:
                for i in range(1, nPoly + 1):
                    layer = ConvBlock(k1)  # *2**(i-1))
                    setattr(self, "layer%d" % i, layer)
        else:
            if UNet == False:
                self.layer1 = ConvBlock(k1)
            else:
                self.layer1 = UNetGenerator2D(
                    feature_scale=UNet[0], nConv=UNet[1], nBlocks=UNet[2]
                )

        self.eta = eta
        self.etaP = 1.0

        if self.update == "phi_etadelta_kappa_mu":
            self.phi_etadelta_kappa_mu = phi_etadelta_kappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa_mu":
            self.phi_etakappa_mu = phi_etakappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa":
            self.phi_etakappa = phi_etakappa_Layer()
        elif self.update == "etakappa":
            self.etakappa = etakappa_Layer()

    def forward(self, phi_0, pbhist, img):
        phi = {}
        phi[0] = phi_0
        kappa = {}

        for i in range(self.nIter):
            kappa[i + 1, 1] = self.layer1(phi[i])
            curvature = kappa[i + 1, 1]
            for j in range(2, self.nPoly + 1):
                if self.multiPoly:
                    conv = getattr(self, "layer%d" % j)
                    kappa[i + 1, j] = conv(kappa[i + 1, j - 1])
                else:
                    kappa[i + 1, j] = self.layer1(kappa[i + 1, j - 1])
                curvature = curvature + kappa[i + 1, j]  # *self.eta**(j-1)

            if self.update == "phi_etadelta_kappa_mu":
                phi[i + 1] = self.phi_etadelta_kappa_mu(
                    curvature, phi[i], pbhist, self.eta, i
                )
            elif self.update == "phi_etakappa_mu":
                phi[i + 1] = self.phi_etakappa_mu(
                    curvature, phi[i], pbhist, self.eta * (self.etaP**i), i
                )
            elif self.update == "phi_etakappa":
                phi[i + 1] = self.phi_etakappa(curvature, phi[i], self.eta)
            if self.update == "etakappa":
                phi[i + 1] = self.etakappa(curvature, self.eta)

        return phi  # [self.nIter]


class CVNN_Unet(nn.Module):
    def __init__(self, nIter, eta, Epsilon, update):
        super(CVNN_Unet, self).__init__()
        # self.eta = eta
        self.update = update
        self.nIter = nIter

        self.layer1 = nn.Sequential(
            unet_2D_2Block(feature_scale=4, nconv=3, is_batchnorm=False),
        )
        self.layer2 = nn.Sequential(
            unet_2D_2Block(feature_scale=4, nconv=3, is_batchnorm=False),
        )
        self.layer3 = nn.Sequential(
            unet_2D_2Block(feature_scale=4, nconv=3, is_batchnorm=False),
        )
        self.layer4 = nn.Sequential(
            unet_2D_2Block(feature_scale=4, nconv=3, is_batchnorm=False),
        )

        self.eta = nn.Parameter(
            torch.cuda.FloatTensor([eta])
        )  # Variable(torch.cuda.FloatTensor(torch.randn(1)), requires_grad=True)
        self.etaP = nn.Parameter(
            torch.cuda.FloatTensor([1.0])
        )  # Variable(torch.cuda.FloatTensor([1.0]),requires_grad=True)
        if self.update == "phi_etadelta_kappa_mu":
            self.phi_etadelta_kappa_mu = phi_etadelta_kappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa_mu":
            self.phi_etakappa_mu = phi_etakappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa":
            self.phi_etakappa = phi_etakappa_Layer()
        elif self.update == "etakappa":
            self.etakappa = etakappa_Layer()

    def forward(self, phi_0, pbhist, img):
        # pbhist=img
        phi = {}
        phi[0] = phi_0
        kappa = {}
        kappa2 = {}
        kappa3 = {}
        kappa4 = {}
        for i in range(self.nIter):
            kappa[i + 1] = self.layer1(phi[i])
            kappa2[i + 1] = self.layer2(kappa[i + 1])
            kappa3[i + 1] = self.layer3(kappa2[i + 1])
            # kappa4[i+1] = self.layer4(kappa3[i+1])

            if self.update == "phi_etadelta_kappa_mu":
                phi[i + 1] = self.phi_etadelta_kappa_mu(
                    kappa[i], phi[i], pbhist, self.eta
                )
            elif self.update == "phi_etakappa_mu":
                # phi[i + 1] = self.phi_etakappa_mu(kappa[i], phi[i], pbhist, self.eta*(self.etaP**i),i)
                phi[i + 1] = self.phi_etakappa_mu(
                    kappa[i + 1] + kappa2[i + 1] + kappa3[i + 1],
                    phi[i],
                    pbhist,
                    self.eta * (self.etaP**i),
                    i,
                )
            elif self.update == "phi_etakappa":
                phi[i + 1] = self.phi_etakappa(kappa[i], phi[i], self.eta)
            if self.update == "etakappa":
                phi[i + 1] = self.etakappa(kappa[i], self.eta)

        return phi  # [self.nIter]


class CVNN_v2_3D(nn.Module):
    def __init__(self, nIter, eta, Epsilon, update):
        super(CVNN_v2_3D, self).__init__()
        self.eta = eta
        self.update = update
        self.nIter = nIter
        k = 3
        p = (k - 1) // 2
        k1 = 3
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv3d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv3d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv3d(k1, k1, kernel_size=1, stride=1),
            nn.ELU(alpha=100.0, inplace=False),
            nn.Conv3d(k1, 1, kernel_size=1, stride=1),
        )

        self.eta = nn.Parameter(torch.cuda.FloatTensor([eta]))
        self.etaP = nn.Parameter(torch.cuda.FloatTensor([1.0]))

        if self.update == "phi_etadelta_kappa_mu":
            self.phi_etadelta_kappa_mu = phi_etadelta_kappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa_mu":
            self.phi_etakappa_mu = phi_etakappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa":
            self.phi_etakappa = phi_etakappa_Layer()
        elif self.update == "etakappa":
            self.etakappa = etakappa_Layer()

    def forward(self, phi_0, pbhist, img):
        phi = {}
        phi[0] = phi_0
        kappa = {}
        for i in range(self.nIter):
            # print(00000)
            kappa[i + 1] = self.layer1(phi[i])
            # print(kappa[i+1].shape)
            if self.update == "phi_etadelta_kappa_mu":
                phi[i + 1] = self.phi_etadelta_kappa_mu(
                    kappa[i + 1], phi[i], pbhist, self.eta
                )
            elif self.update == "phi_etakappa_mu":
                phi[i + 1] = self.phi_etakappa_mu(
                    kappa[i + 1], phi[i], pbhist, self.eta * (self.etaP**i), i
                )
                # print(100, phi[i+1].shape )
            elif self.update == "phi_etakappa":
                phi[i + 1] = self.phi_etakappa(kappa[i + 1], phi[i], self.eta)
            if self.update == "etakappa":
                phi[i + 1] = self.etakappa(kappa[i + 1], self.eta)

        return phi  # [self.nIter]


class CVNN_v2(nn.Module):
    def __init__(self, nIter, eta, Epsilon, update):
        super(CVNN_v2, self).__init__()
        self.eta = eta
        self.update = update
        self.nIter = nIter
        k = 3
        p = (k - 1) // 2
        k1 = 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
            nn.ELU(alpha=100.0, inplace=False),
            nn.Conv2d(k1, 1, kernel_size=1, stride=1),
        )

        self.eta = nn.Parameter(torch.cuda.FloatTensor([eta]))
        self.etaP = nn.Parameter(torch.cuda.FloatTensor([1.0]))

        if self.update == "phi_etadelta_kappa_mu":
            self.phi_etadelta_kappa_mu = phi_etadelta_kappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa_mu":
            self.phi_etakappa_mu = phi_etakappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa":
            self.phi_etakappa = phi_etakappa_Layer()
        elif self.update == "etakappa":
            self.etakappa = etakappa_Layer()

    def forward(self, phi_0, pbhist, img):
        phi = {}
        phi[0] = phi_0
        kappa = {}
        for i in range(self.nIter):
            kappa[i + 1] = self.layer1(phi[i])
            if self.update == "phi_etadelta_kappa_mu":
                phi[i + 1] = self.phi_etadelta_kappa_mu(
                    kappa[i + 1], phi[i], pbhist, self.eta
                )
            elif self.update == "phi_etakappa_mu":
                phi[i + 1] = self.phi_etakappa_mu(
                    kappa[i + 1], phi[i], pbhist, self.eta * (self.etaP**i), i
                )
            elif self.update == "phi_etakappa":
                phi[i + 1] = self.phi_etakappa(kappa[i + 1], phi[i], self.eta)
            if self.update == "etakappa":
                phi[i + 1] = self.etakappa(kappa[i + 1], self.eta)

        return phi  # [self.nIter]


class CVNN_v2_2(nn.Module):
    def __init__(self, nIter, eta, Epsilon, update):
        super(CVNN_v2_2, self).__init__()
        self.eta = eta
        self.update = update
        self.nIter = nIter
        k = 3
        p = (k - 1) // 2
        k1 = 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
            nn.ELU(alpha=100.0, inplace=False),
            nn.Conv2d(k1, 1, kernel_size=1, stride=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
            nn.ELU(alpha=100.0, inplace=False),
            nn.Conv2d(k1, 1, kernel_size=1, stride=1),
        )

        self.eta = nn.Parameter(torch.cuda.FloatTensor([eta]))
        self.etaP = nn.Parameter(torch.cuda.FloatTensor([1.0]))

        self.eta2 = nn.Parameter(torch.cuda.FloatTensor([eta]))
        self.etaP2 = nn.Parameter(torch.cuda.FloatTensor([1.0]))
        if self.update == "phi_etadelta_kappa_mu":
            self.phi_etadelta_kappa_mu = phi_etadelta_kappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa_mu":
            self.phi_etakappa_mu = phi_etakappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa":
            self.phi_etakappa = phi_etakappa_Layer()
        elif self.update == "etakappa":
            self.etakappa = etakappa_Layer()

    def forward(self, phi_0, pbhist, img):
        phi = {}
        phi[0] = phi_0
        kappa = {}
        for i in range(self.nIter // 2):
            kappa[i] = self.layer1(phi[i])
            phi[i + 1] = self.phi_etakappa_mu(
                kappa[i], phi[i], pbhist, self.eta * (self.etaP**i), i
            )

        for i in range(self.nIter // 2, self.nIter):
            kappa[i] = self.layer2(phi[i])
            phi[i + 1] = self.phi_etakappa_mu(
                kappa[i], phi[i], pbhist, self.eta2 * (self.etaP2**i), i
            )
        return phi  # [self.nIter]


class CVNN_v2_Im(nn.Module):
    def __init__(self, nIter, eta, Epsilon, update):
        super(CVNN_v2_Im, self).__init__()
        self.eta = eta
        self.update = update
        self.nIter = nIter
        k = 3
        p = (k - 1) // 2
        k1 = 3
        self.layer1 = nn.Conv2d(1, k1, kernel_size=k, stride=1, padding=p)
        self.layer2 = nn.Conv2d(1, k1, kernel_size=k, stride=1, padding=p)
        self.layer3 = nn.Sequential(
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
            nn.ELU(alpha=100.0, inplace=False),
            nn.Conv2d(k1, 1, kernel_size=1, stride=1),
        )

        self.eta = nn.Parameter(torch.cuda.FloatTensor([eta]))
        self.etaP = nn.Parameter(torch.cuda.FloatTensor([1.0]))

        if self.update == "phi_etadelta_kappa_mu":
            self.phi_etadelta_kappa_mu = phi_etadelta_kappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa_mu":
            self.phi_etakappa_mu = phi_etakappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa":
            self.phi_etakappa = phi_etakappa_Layer()
        elif self.update == "etakappa":
            self.etakappa = etakappa_Layer()

    def forward(self, phi_0, pbhist, img):
        phi = {}
        phi[0] = phi_0
        kappa = {}
        intermediate = {}
        # pbh=self.layer2(pbhist)
        for i in range(self.nIter):
            intermediate[i] = self.layer1(phi[i])  # +pbh
            kappa[i] = self.layer3(intermediate[i])
            if self.update == "phi_etadelta_kappa_mu":
                phi[i + 1] = self.phi_etadelta_kappa_mu(
                    kappa[i], phi[i], pbhist, self.eta
                )
            elif self.update == "phi_etakappa_mu":
                phi[i + 1] = self.phi_etakappa_mu(
                    kappa[i], phi[i], pbhist, self.eta * (self.etaP**i), i
                )
            elif self.update == "phi_etakappa":
                phi[i + 1] = self.phi_etakappa(kappa[i], phi[i], self.eta)
            if self.update == "etakappa":
                phi[i + 1] = self.etakappa(kappa[i], self.eta)

        return phi  # [self.nIter]


class CVNN_v4(nn.Module):
    def __init__(self, nIter, eta, Epsilon, update):
        super(CVNN_v4, self).__init__()
        self.eta = eta
        self.update = update
        self.nIter = nIter
        k = 3
        p = (k - 1) // 2
        k1 = 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
            nn.ELU(alpha=100.0, inplace=False),
            nn.Conv2d(k1, 1, kernel_size=1, stride=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
            nn.ELU(alpha=100.0, inplace=False),
            nn.Conv2d(k1, 1, kernel_size=1, stride=1),
        )

        self.eta = nn.Parameter(torch.cuda.FloatTensor([eta]))
        self.etaP = nn.Parameter(torch.cuda.FloatTensor([1.0]))
        self.eta2 = nn.Parameter(torch.cuda.FloatTensor([eta]))
        self.etaP2 = nn.Parameter(torch.cuda.FloatTensor([1.0]))

        if self.update == "phi_etadelta_kappa_mu":
            self.phi_etadelta_kappa_mu = phi_etadelta_kappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa_mu":
            self.phi_etakappa_mu = phi_etakappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa":
            self.phi_etakappa = phi_etakappa_Layer()
        elif self.update == "etakappa":
            self.etakappa = etakappa_Layer()

    def forward(self, phi_0, pbhist, img):
        phi = {}
        phi[0] = phi_0
        kappa = {}
        for i in range(self.nIter):
            kappa[i] = self.layer1(phi[i])
            if self.update == "phi_etadelta_kappa_mu":
                phi[i + 1] = self.phi_etadelta_kappa_mu(
                    kappa[i], phi[i], pbhist, self.eta
                )
            elif self.update == "phi_etakappa_mu":
                phi[i + 1] = self.phi_etakappa_mu(
                    kappa[i], phi[i], pbhist, self.eta * (self.etaP**i), i
                )
            elif self.update == "phi_etakappa":
                phi[i + 1] = self.phi_etakappa(kappa[i], phi[i], self.eta)
            if self.update == "etakappa":
                phi[i + 1] = self.etakappa(kappa[i], self.eta)

        for i in range(self.nIter, 2 * self.nIter):
            kappa[i] = self.layer2(phi[i])
            phi[i + 1] = self.phi_etakappa_mu(
                kappa[i],
                phi[i],
                img * torch.sigmoid(phi[i]),
                self.eta * (self.etaP2 ** (i - self.nIter)),
                i - self.nIter,
            )  # img*torch.sigmoid(phi[i+1])

        return phi  # [self.nIter]


class CV_block(nn.Module):
    def __init__(self, nIter, k1, eta, Epsilon, update):
        super(CV_block, self).__init__()
        self.eta = eta
        self.update = update
        self.nIter = nIter
        k = 3
        p = 1
        self.layer1 = nn.Sequential(
            # nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            # nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.ELU(alpha=2.0, inplace=False),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
            # nn.ReLU(),
            # nn.LeakyReLU(0.4, inplace=True),
            nn.ELU(alpha=2.0, inplace=False),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
        )
        if self.update == "phi_etadelta_kappa_mu":
            self.phi_etadelta_kappa_mu = phi_etadelta_kappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa_mu":
            self.phi_etakappa_mu = phi_etakappa_mu_Layer(Epsilon)
        elif self.update == "phi_etakappa":
            self.phi_etakappa = phi_etakappa_Layer()
        elif self.update == "etakappa":
            self.etakappa = etakappa_Layer()

    def forward(self, phi_0, pbhist):
        phi = {}
        phi[0] = phi_0
        kappa = {}
        for i in range(self.nIter):
            kappa[i] = self.layer1(phi[i])
            if self.update == "phi_etadelta_kappa_mu":
                phi[i + 1] = self.phi_etadelta_kappa_mu(
                    kappa[i], phi[i], pbhist, self.eta
                )
            elif self.update == "phi_etakappa_mu":
                phi[i + 1] = self.phi_etakappa_mu(
                    kappa[i], phi[i], pbhist, self.eta * (0.5 ** (max(0, i - 2))), i
                )
            elif self.update == "phi_etakappa":
                phi[i + 1] = self.phi_etakappa(kappa[i], phi[i], self.eta)
            if self.update == "etakappa":
                phi[i + 1] = self.etakappa(kappa[i], self.eta)

        return phi[self.nIter]


class ConvNet(nn.Module):
    def __init__(self, eta, Epsilon):
        super(ConvNet, self).__init__()
        self.eta = nn.Parameter(torch.cuda.FloatTensor([eta]))
        self.etaP = nn.Parameter(torch.cuda.FloatTensor([1.0]))
        k = 3
        p = (k - 1) // 2
        k1 = 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
            nn.ELU(alpha=100.0, inplace=False),
            nn.Conv2d(k1, 1, kernel_size=1, stride=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
            nn.ELU(alpha=100.0, inplace=False),
            nn.Conv2d(k1, 1, kernel_size=1, stride=1),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=k, stride=1, padding=p),
            nn.Conv2d(k1, k1, kernel_size=1, stride=1),
            nn.ELU(alpha=100.0, inplace=False),
            nn.Conv2d(k1, 1, kernel_size=1, stride=1),
        )
        self.phi_etakappa_mu = phi_etakappa_mu_Layer(Epsilon)

    def forward(self, phi_0, pbhist, img):
        kappa = {}
        phi = {}
        phi[0] = phi_0
        kappa[1] = self.layer1(phi[0])
        phi[1] = self.phi_etakappa_mu(
            kappa[1], phi[0], pbhist, self.eta * (self.etaP**0), 0
        )

        kappa[2] = self.layer1(phi[1])
        phi[2] = self.phi_etakappa_mu(
            kappa[2], phi[1], pbhist, self.eta * (self.etaP**1), 1
        )

        kappa[3] = self.layer1(phi[2])
        phi[3] = self.phi_etakappa_mu(
            kappa[3], phi[2], pbhist, self.eta * (self.etaP**2), 2
        )

        return kappa, phi
