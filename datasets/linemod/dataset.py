import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import cv2
sys.path.append('..')
import datasets.linemod.get_normal_map_pvn3d as gnmp


class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        '''
        :param mode: 可以选择train，test，eval
        :param num: mesh点的数目
        :param add_noise:是否加入噪声
        :param root:数据集的根目录
        :param noise_trans:噪声增强的相关参数
        :param refine:是否需要为refine模型提供相应的数据
        '''
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]  # 这里表示目标物体类别序列号
        self.mode = mode
        self.list_rgb = []   # 存储rg图像的路径
        self.list_depth = [] # 存储深度图的路径
        self.list_label = [] # 存储语义分割出来物体对应的mask
        self.list_obj = []   # obj和rank两个拼接起来,可以知道每张图片的路径，及物体类别，和图片下标
        self.list_rank = []
        self.meta = {}       # 拍摄图片时的旋转矩阵和偏移矩阵，以及物体box
        self.pt = {}         # 保存目标模型models点云数据，及models/obj_xx.ply文件中的数据
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine

        item_count = 0
        for item in self.objlist: # 对每个目标物的相关数据进行处理
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                if self.mode == 'eval':
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))
                
                self.list_obj.append(item)
                self.list_rank.append(int(input_line))

            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r')
            self.meta[item] = yaml.load(meta_file)
            # 这里保存的是目标物体，拍摄物体第一帧的点云数据，可以成为模型数据
            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))
            
            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)

        self.cam_cx = 325.26110  # 相机光轴在图像坐标系中的偏移量，单位为像素
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140  # 焦距 详细解释https://www.cnblogs.com/shaonianpi/p/12715282.html
        self.cam_fy = 573.57043

        self.xmap = np.array([[j for i in range(640)] for j in range(480)]) #列举图像的x、y坐标，先执行内括号再执行外括号
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        # 设定获取目标物体点云的数据
        self.num = num
        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05) # 随机改变亮度，对比度和饱和度的图像
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])# 均值和标准差归一化

        # 边界列表，可以想象把一个图片切割成了多个块
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 500    # 点云的最大和最小数目
        self.num_pt_mesh_small = 500
        self.symmetry_obj_idx = [7, 8]  # 标记对称物体的序列号

    # 迭代时调用该函数
    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index]) # 根据索引获得对应图片的RGB像素
        ori_img = np.array(img)
        depth = np.array(Image.open(self.list_depth[index])) # 根据索引获得对应图像的深度图像素
        label = np.array(Image.open(self.list_label[index])) # 根据索引获得对应图像的mask像素
        obj = self.list_obj[index] # 获得物体属于的类别的序列号
        rank = self.list_rank[index] # 获得该张图片物体图像的标号

        if obj == 2: # 如果该目标物体的序列为2，暂时不对序列为2的物体图像进行处理
            for i in range(0, len(self.meta[obj][rank])):
                if self.meta[obj][rank][i]['obj_id'] == 2:
                    meta = self.meta[obj][rank][i]
                    break
        else:
            meta = self.meta[obj][rank][0]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))

        # 标准的数据中的mask是3通道的，通过网络分割出来的mask，其是单通道的
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]

        # 对应元素相乘不是矩阵乘法,把mask和深度图结合到到一起，物体存在的区域像素为True，背景像素为False;此处还只是掩码
        mask = mask_label * mask_depth

        if self.add_noise: # 加入噪声
            img = self.trancolor(img)

        # [b,h,w,c]-->[b,c,h,w]
        img = np.array(img)[:, :, :3]
        img = np.transpose(img, (2, 0, 1))
        img_masked = img

        # 如果为eval模式，根据mask_label获得目标的box（rmin-rmax表示行的位置，cmin-cmax表示列的位置）
        if self.mode == 'eval':
            rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label)) # 表示的根据mask生成合适的box
        # 如果不是eval模式，则从gt.yml文件中，获取最标准的box
        else:
            rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb']) #行列最小值最大值

        # 根据确定的行和列，对图像进行截取，就是截取处包含了目标物体的图像
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]

        # 获得目标物体旋转矩阵的参数，以及偏移矩阵参数
        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        target_t = np.array(meta['cam_t_m2c'])
        # 对偏移矩阵添加噪声
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0] # 对mask图片的目标部分进行剪裁，变成拉平，变成一维
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc) # 如果剪切下来的部分面积为0，则直接返回五个0的元组，即表示没有目标物体

        # 如果剪切下来目标图片的像素，大于了点云的数目（一般都是这种情况）
        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int) # c_mask全部设置为0，大小和choose相同
            c_mask[:self.num] = 1  # 前self.num设置为1
            np.random.shuffle(c_mask)  # 随机打乱
            choose = choose[c_mask.nonzero()] # 选择c_mask不是0的部分，也就是说，只选择了500个像素，注意nonzero()返回的是索引
        # 如果剪切像素点的数目，小于了点云的数目
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap') # 使用0填补，调整到和点云数目一样大小（500）


        # 把深度图，对应着物体的部分也剪切下来，然后拉平，变成一维，挑选坐标
        depth_crop = depth[rmin:rmax, cmin:cmax]
        depth_masked = depth_crop.flatten()[choose][:, np.newaxis].astype(np.float32)
        # 把物体存在于原图的位置坐标剪切下来，拉平，然后进行挑选坐标挑选
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cld, n_choose = gnmp.dpt_2_cld(depth_crop)
        normal = gnmp.get_normal(cld)
        normal_map = gnmp.get_normal_map(normal, n_choose, rmin, rmax, cmin, cmax)
        normal_map = np.transpose(normal_map, (2, 0, 1))

        cam_scale = 1.0 # 摄像头缩放参数
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx  # 为对坐标进行正则化做准备
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)   # 把y，x，depth他们3个坐标合并在一起，变成点云数据，
        cloud = cloud / 1000.0  # 这里把点云数据除以1000，是为了根据深度进行正则化

        if self.add_noise:
            cloud = np.add(cloud, add_t)  # 对点云添加噪声


        # 存储在obj_xx.ply中的点云数据，对其进行正则化，也就是目标物体的点云信息
        model_points = self.pt[obj] / 1000.0
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)


        # 根据model_points（第一帧目标模型对应的点云信息），以及target（目前迭代这张图片）的旋转和偏移矩阵，计算出对应的点云数据
        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t / 1000.0 + add_t)
            out_t = target_t / 1000.0 + add_t
        else:
            target = np.add(target, target_t / 1000.0)
            out_t = target_t / 1000.0

        # 总结：
        # cloud：由深度图计算出来的点云，该点云数据以本摄像头为参考坐标
        # choose：所选择点云的索引
        # img_masked：通过box剪切下来的RGB图像 均值和标准差归一化数据提前已经定义好的，不能直接用在深度信息上
        # target：根据model_points点云信息，以及旋转偏移矩阵转换过的点云信息
        # model_points：目标初始帧（模型）对应的点云信息
        # [self.objlist.index(obj)]：目标物体的序列编号
        # 法线映射到RGB
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.objlist.index(obj)]), \
               torch.from_numpy(normal_map.astype(np.float32))
               # self.list_rgb[index] # 返回当前加载的图片路径

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small



border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680] #18个点分成17份
img_width = 480
img_length = 640


def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #找物体轮廓


    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)
