from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from lib.knn.__init__ import KNearestNeighbor


def loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, num_point_mesh, sym_list):
    """
    :param pred_r: 预测的旋转参数[bs, 500, 4]，相对于摄像头
    :param pred_t: 预测的偏移参数[bs, 500, 3]，相对于摄像头
    :param pred_c: 预测的置信度参数[bs, 500, 1]，相对于摄像头
    :param target: 目标姿态，也就是预测图片，通过标准偏移矩阵，结合model_points求得图片对应得点云数据[bs,500,3]，这里点云数据，就是学习的目标数据
    :param model_points:目标模型的点云数据-第一帧[bs,500,3]
    :param idx:随机训练的一个索引
    :param points:由深度图计算出来的点云，也就是说该点云数据以摄像头为参考坐标
    :param refine:标记是否已经开始训练refine网络
    :param num_point_mesh:500
    :param sym_list:对称模型的序列号
    """
    knn = KNearestNeighbor(1)
    # [bs,500,1]
    bs, num_p, _ = pred_c.size()

    # 预测的旋转矩阵正则化
    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))

    # base[bs，500, 4] -->[500, 3, 3]，把预测的旋转参数，转化为旋转矩阵
    base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base  # 把相对于摄像头的偏移矩阵记录下来
    base = base.contiguous().transpose(2, 1).contiguous() # [3, 3, 500]

    # 复制num_p=500次，[bs,1,500,3]-->[500,500,3]，这里的复制操作，主要是因为每个ground truth（target）点云，
    # 需要与所有的predicted点云做距离差，
    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
    ori_target = target     # 把初始的目标点云（已经通过标准的pose进行了变换）记录下来
    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
    ori_t = pred_t     # 把原始预测的偏移矩阵记录下来，这里的t是相对摄像头的

    # 当前帧的点云，结合深度图计算而来，也就是说该点云信息是以摄像头为参考目标
    points = points.contiguous().view(bs * num_p, 1, 3)
    pred_c = pred_c.contiguous().view(bs * num_p)

    # 为批量矩阵相乘,model_points与旋转矩阵相乘加上偏移矩阵，得到当前帧对应的点云姿态，该点云的姿态是以model_points为参考的
    # pred[500,500,3]
    pred = torch.add(torch.bmm(model_points, base), points + pred_t) #  torch.bmm()是tensor中的一个相乘操作，类似于矩阵中的A*B。


    if not refine:
        if idx[0].item() in sym_list:  # 如果是对称的物体
            target = target[0].transpose(1, 0).contiguous().view(3, -1) # [500,500,3]-->[3,250000]
            pred = pred.permute(2, 0, 1).contiguous().view(3, -1) # [500,500,3]-->[3,250000]

            # [1, 1, 250000]，target的每个点云和pred的所有点云进行对比，找到每个target点云与pred的所有点云，距离最近点云的索引（pred）
            inds = knn(target.unsqueeze(0), pred.unsqueeze(0))

            # [3, 250000]，从target点云中，根据计算出来的min索引，全部挑选出来
            target = torch.index_select(target, 1, inds.view(-1).detach() - 1)
            # [500, 500, 3]
            target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
            # [500, 500, 3]
            pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()

    # 求得预测点云和目标点云的平均距离（每个点云）,按照论文，把置信度和点云距离关联起来
    dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
    loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)

    # 下面的操作都是为refine模型训练的准备
    pred_c = pred_c.view(bs, num_p)
    how_max, which_max = torch.max(pred_c, 1)
    dis = dis.view(bs, num_p)

    # 获得最好的偏移矩阵，这里的t是相对model_points的
    t = ori_t[which_max[0]] + points[which_max[0]]
    points = points.view(1, bs * num_p, 3)

    # 求得500中置信度最高的旋转矩阵，相对于摄像头的
    ori_base = ori_base[which_max[0]].view(1, 3, 3).contiguous()
    ori_t = t.repeat(bs * num_p, 1).contiguous().view(1, bs * num_p, 3)

    # 根据预测最好的旋转矩阵，求得新的当前帧对应的点云，注意这里是一个减号的操作，并且其中的ori_t是相对于摄像头的
    # （但是ori_t和ori_base都是预测出来的，就是返回去肯定存在偏差的）
    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    new_target = ori_target[0].view(1, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(num_point_mesh, 1).contiguous().view(1, num_point_mesh, 3)

    # 根据预测最好的旋转矩阵，求得新的当前帧对应的点云，注意这里是一个减号的操作，并且其中的ori_t是相对于摄像头的
    # （但是ori_t和ori_base都是预测出来的，就是返回去肯定存在偏差的，这里的偏差因为new_target是标准的，所以应该少一些）
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    # print('------------> ', dis[0][which_max[0]].item(), pred_c[0][which_max[0]].item(), idx[0].item())
    del knn

    # loss：根据每个点云计算出来的平均loss
    # 对应预测置信度度最高，target与预测点云之间的最小距离
    # new_points：根据最好的旋转矩阵，求得当前帧的点云
    # new_target：就是根据model_points求得的标椎点云
    return loss, dis[0][which_max[0]], new_points.detach(), new_target.detach()


class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine):

        return loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, self.num_pt_mesh, self.sym_list)
