import torch
from torch import nn
from torch.nn import functional as F
import lib.extractors as extractors

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        # 自适应平均池化，只需要给定输出特征图的大小就好，其中通道数前后不发生变化
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # priors 是根据feats生成的特征图的列表，尺寸通道和feats一样；上采样还可以直接指定输出尺寸大小
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class PSPNet(nn.Module):
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet18',
                 pretrained=True):
        super(PSPNet, self).__init__()
        self.feats_depth = getattr(extractors, 'resnet18_depth')(pretrained) # 获取对象resnet18_depth
        self.psp_depth1 = PSPModule(64, 128, sizes)
        self.psp_depth2 = PSPModule(128, 256, sizes)
        self.psp_depth3 = PSPModule(256, 512, sizes)
        self.psp_depth4 = PSPModule(512, 1024, sizes)  # psp_size使用时输入为512


        self.feats = getattr(extractors, backend)(pretrained)  # 获取对象resnet18
        self.psp1 = PSPModule(64, 128, sizes)
        self.up_01 = PSPUpsample(256, 256)
        self.psp2 = PSPModule(128, 256, sizes)
        self.up_02 = PSPUpsample(512, 512)
        self.psp3 = PSPModule(256, 512, sizes)
        self.psp4 = PSPModule(psp_size, 1024, sizes)  # psp_size使用时输入为512
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.c_1 = nn.Conv2d(2048, 1024, 3, padding=1)  # 只改变层数不改变尺寸
        self.up_1 = PSPUpsample(2048, 1024)
        self.drop_2 = nn.Dropout2d(p=0.15)

        self.c_2 = nn.Conv2d(1024, 512, 3, padding=1)
        self.up_2 = PSPUpsample(1024, 512)

        self.c_3 = nn.Conv2d(512, 256, 3, padding=1)
        self.up_3 = PSPUpsample(512, 256)
        self.drop_3 = nn.Dropout2d(p=0.075)

        self.c_4 = nn.Conv2d(256, 128, 3, padding=1)
        self.drop_4 = nn.Dropout2d(p=0.05)

        self.c_5 = nn.Conv2d(128, 64, 3, padding=1)

        self.prelu = nn.ReLU()  # nn.ReLU()?
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.LogSoftmax()
        )  # 将网络层包装

    def forward(self, x, x_normal_map):

        df1,df2,df3,df4 = self.feats_depth(x_normal_map) #64 128 256 512
        df01 = self.psp_depth1(df1)  # [1,128,h,w]
        df02 = self.psp_depth2(df2)  # [1,256,h,w]
        df03 = self.psp_depth3(df3)  # [1,512,h,w]
        df04 = self.psp_depth4(df4)  # [1,1024,h,w]



        x1, x2, x3, x4 = self.feats(x)  # 尺度0.5x1=x2=x3=x4 层数64 128 256 512
        p01 = self.psp1(x1)  # [1,128,h,w]
        p01 = torch.cat((p01, df01), dim=1)  # 128+128
        p01 = self.up_01(p01)  # 256->256 h->2h
        p01 = self.drop_1(p01)

        p02 = self.psp2(x2)  # [1,256,h,w]
        p02 = torch.cat((p02, df02), dim=1) # 256+256
        p02 = self.up_02(p02)  # 512->512 h->2h
        p02 = self.drop_1(p02)

        p03 = self.psp3(x3)  # [1,512,h,w]
        p03 = torch.cat((p03, df03), dim=1) # 512+512
        p03 = self.drop_1(p03)

        p04 = self.psp4(x4)  # [1,1024,h,w]
        p04 = torch.cat((p04, df04), dim=1) # 1024+1024
        p04 = self.drop_1(p04)  # 经过drop_out尺寸不变

        p2 = self.c_1(p04)  # 2048->1024
        p2 = self.prelu(p2)
        p2 = torch.cat((p2, p03), dim=1)  # 1024+1024
        p2 = self.up_1(p2)  # 2048->1024 h->2h
        p2 = self.drop_2(p2)

        p3 = self.c_2(p2)  # 1024->512
        p3 = self.prelu(p3)
        p3 = torch.cat((p3, p02), dim=1)  # 512+512
        p3 = self.up_2(p3)  # 1024->512 2h->4h
        p3 = self.drop_2(p3)

        p4 = self.c_3(p3)  # 512->256
        p4 = self.prelu(p4)
        p4 = torch.cat((p4, p01), dim=1)  # 256+256
        p4 = self.up_3(p4)  # 512->256 4h->8h 8h=H?
        p4 = self.drop_3(p4)

        p5 = self.c_4(p4)  # 256->128 8h->8h
        p5 = self.prelu(p5)
        p5 = self.drop_4(p5)

        p6 = self.c_5(p5)  # 128->64 8h->8h
        p6 = self.prelu(p6)

        return self.final(p6)