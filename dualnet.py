from bluntools.layers import *


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        # Layer_1
        self.t1_conv_1x, self.t2_conv_1x = ConvBNReLU(1, 16), ConvBNReLU(1, 16)
        # Layer_2
        self.t1_conv_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(16, 32), ConvBNReLU(32, 32))
        self.t2_conv_2x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(16, 32), ConvBNReLU(32, 32))
        # Layer_3
        self.t1_conv_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(32, 64), ConvBNReLU(64, 64))
        self.t2_conv_4x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(32, 64), ConvBNReLU(64, 64))
        # Layer_4
        self.t1_conv_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        self.t2_conv_8x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        # Layer_5
        self.t1_conv_16x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(128, 256), ConvBNReLU(256, 256))
        self.t2_conv_16x = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(128, 256), ConvBNReLU(256, 256))

        # Bottom 32x
        self.t1_bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(256, 256), ConvBNReLU(256, 256))
        self.t2_bottom = nn.Sequential(nn.MaxPool2d(2, stride=2), ConvBNReLU(256, 256), ConvBNReLU(256, 256))

        self.fusion_16x = MultiRefine(256, 256, up=True)
        self.fusion_8x = MultiRefine(256, 128, up=True)
        self.fusion_4x = MultiRefine(128, 64, up=True)
        self.fusion_2x = MultiRefine(64, 32, up=True)
        self.fusion_1x = MultiRefine(32, 16, up=True)

        self.classify = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, t1, t2):
        t1_conv_1x, t2_conv_1x = self.t1_conv_1x(t1), self.t2_conv_1x(t2)
        t1_conv_2x, t2_conv_2x = self.t1_conv_2x(t1_conv_1x), self.t2_conv_2x(t2_conv_1x)
        t1_conv_4x, t2_conv_4x = self.t1_conv_4x(t1_conv_2x), self.t2_conv_4x(t2_conv_2x)
        t1_conv_8x, t2_conv_8x = self.t1_conv_8x(t1_conv_4x), self.t2_conv_8x(t2_conv_4x)
        t1_conv_16x, t2_conv_16x = self.t1_conv_16x(t1_conv_8x), self.t2_conv_16x(t2_conv_8x)

        t1_bottom = self.t1_bottom(t1_conv_16x)
        t2_bottom = self.t2_bottom(t2_conv_16x)

        fusion_16x = self.fusion_16x([t1_bottom, t2_bottom])
        fusion_8x = make_same(t1_conv_8x, self.fusion_8x([t1_conv_16x, t2_conv_16x, fusion_16x]))
        fusion_4x = make_same(t1_conv_4x, self.fusion_4x([t1_conv_8x, t2_conv_8x, fusion_8x]))
        fusion_2x = make_same(t1_conv_2x, self.fusion_2x([t1_conv_4x, t2_conv_4x, fusion_4x]))
        fusion_1x = make_same(t1_conv_1x, self.fusion_1x([t1_conv_2x, t2_conv_2x, fusion_2x]))

        return self.classify(fusion_1x)
