import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
} # psp_size 是resnet最后一层的输出通道数，pspnet的输入通道数；deep_features_size是分类通道数目

class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()
        self.model = psp_models['resnet18'.lower()]()  #字符串小写
        self.model = nn.DataParallel(self.model)

    def forward(self, x, x_normal_map):
        x = self.model(x, x_normal_map)
        # x为输入的图片经过pspnet处理返回后的特征值; pspnet此时输入为rgb和depth，二者融合以后才输出
        return x


class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1) # 输入输出卷积核尺寸
        # 使用的卷积全是一维卷积，kernel=1，只对宽度进行卷积，对高度不卷积。
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points) # 用一维核进行平均池化
    
        self.num_points = num_points
    def forward(self, x, emb):
        x = F.relu(self.conv1(x)) # [bs, 3, 500]-->[bs,64,500]
        emb = F.relu(self.e_conv1(emb)) # [bs, 32, 500]-->[bs,64,500]
        pointfeat_1 = torch.cat((x, emb), dim=1) # 第一次concat 64+64=128  [bs,128,500]

        x = F.relu(self.conv2(x)) # [bs, 64, 500]-->[bs, 128, 500]
        emb = F.relu(self.e_conv2(emb)) # [bs, 64, 500]-->[bs, 128, 500]
        pointfeat_2 = torch.cat((x, emb), dim=1) # 第二次concat 128+128=256 [bs, 256, 500]

        x = F.relu(self.conv5(pointfeat_2)) # [bs, 256, 500]-->[bs, 512, 500]
        x = F.relu(self.conv6(x)) # [bs, 512, 500]-->[bs, 1024, 500]
        # pointfeat_4 = torch.cat((x, emb1), dim=1)  #1024+1024=2048

        ap_x = self.ap1(x)

        # ap_x[bs, 1024, 500]
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) # 192+256+768+1024=2240
        #第三次concat 128 + 256 + 1024 = 1408
        # pointfeat_1, pointfeat_2都是点和点之间的信息，ap_x是全局信息

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        """
        :param num_points:输入网络的点云数目
        :param num_obj: 目标物体的种类
        """
        super(PoseNet, self).__init__()
        self.num_points = num_points # 点云的数目
        self.cnn = ModifiedResnet()  # 修改过后的resnet RGB CNN特征提取模块
        self.feat = PoseNetFeat(num_points) #融合网络

        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1) #confidence

        self.num_obj = num_obj

    def forward(self, img, x, choose, obj, normal_map):
        """
        PoseNet的前向传播，进行姿态预测
        :param img: RGB图像的像素[bs,3,h,w]
        :param x: 点云数据[bs, 500, 3]
        :param choose: 选择点云的index下标[bs, 1, 500]
        :param obj: 目标物体的序列号[bs, 1]
        :param img_depth: 目标物体的深度图[bs,h,w]  ——>nn.Conv2d 的时候处理成bs*size size=c*h*w
        :return:对图像预测的姿态
        """
        #先将img_depth 处理成1*1*h*w 再和img再dim=1上concat
        #　merge_img= torch.cat([img,torch.unsqueeze(img_depth,dim=0)],dim=1)#将img_depth作为img的第四维
        # todo 此处可以再定义一个cnn单独将img_depth进行处理，是在cnn中进行，最后一层融合后再和点云进行以下的融合
        out_img = self.cnn(img, normal_map) #out_img[bs, 32, h, w]，hw不固定,color embeddings

        bs, di, _, _ = out_img.size()
        emb = out_img.view(bs, di, -1) #[1,32,12800] h w会变所以第三位数不固定
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()
        # torch.view等方法操作需要连续的Tensor https://zhuanlan.zhihu.com/p/64551412

        x = x.transpose(2, 1).contiguous()# [bs, 500, 3]-->[bs,3,500]
        ap_x = self.feat(x, emb)  # 融合后的feature

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points) #调整形状
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)
        
        b = 0 # 选择预测对应目标内标的矩阵参数
        out_rx = torch.index_select(rx[b], 0, obj[b]) # [bs, 4, 500] 索引对象，行、列，索引序号
        out_tx = torch.index_select(tx[b], 0, obj[b]) # [bs, 3, 500] https://blog.csdn.net/g_blink/article/details/102854188
        out_cx = torch.index_select(cx[b], 0, obj[b]) # [bs, 1, 500]
        # 500表示对每个像素都有进行预测，后面我们就要从这500个预测中，选出最好的一个结果最终的预测结果。

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous() # [bs, 500, 4]
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous() # [bs, 500, 3]
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous() # [bs, 500, 1]

        #emb1.detach()
        # emb2.detach()
        # emb3.detach()
        return out_rx, out_tx, out_cx, emb.detach() # detach()表示从图中分离出来，不做反向传播
 


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)
        
        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) #quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]
        
        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))   

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx
