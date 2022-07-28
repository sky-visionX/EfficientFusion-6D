# -*- coding:utf-8 -*-

import  numpy as np
from  PIL import Image
import matplotlib.pyplot as plt
import cv2
import yaml


def get_normal_map(nrm, choose,rmin, rmax, cmin, cmax):
    # nrm_map = np.zeros((480, 640, 3), dtype=np.uint8)
    nrm_map = np.zeros((rmax - rmin, cmax - cmin, 3), dtype=np.uint8)
    nrm = nrm[:, :3]
    nrm[np.isnan(nrm)] = 0.0
    nrm[np.isinf(nrm)] = 0.0
    nrm_color = ((nrm + 1.0) * 127).astype(np.uint8)
    nrm_map = nrm_map.reshape(-1, 3)
    nrm_map[choose, :] = nrm_color
    # nrm_map = nrm_map.reshape((480, 640, 3))
    nrm_map = nrm_map.reshape((rmax - rmin, cmax - cmin, 3))
    return nrm_map

def get_normal(cld):
    import pcl
    cloud = pcl.PointCloud()
    cld = cld.astype(np.float32)
    cloud.from_array(cld)
    ne = cloud.make_NormalEstimation()
    kdtree = cloud.make_kdtree()
    ne.set_SearchMethod(kdtree)
    ne.set_KSearch(50)
    n = ne.compute()
    n = n.to_array()
    return n

def dpt_2_cld(dpt):
    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    msk_dp = dpt > 1e-6
    choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
    if len(choose) < 1:
        return None, None
    dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_mskd = np.array([[j for i in range(dpt.shape[1])] for j in range(dpt.shape[0])]).flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_mskd = np.array([[i for i in range(dpt.shape[1])] for j in range(dpt.shape[0])]).flatten()[choose][:, np.newaxis].astype(np.float32)
    cam_scale=1000.0
    pt2 = dpt_mskd / cam_scale
    cam_cx, cam_cy = 325.26110, 242.04899
    cam_fx, cam_fy = 572.41140, 573.57043
    # cam_cy=cam_cy-((rmax-rmin)/2)
    # cam_cx=cam_cx-((cmax-cmin)/2)
    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    cld = np.concatenate((pt0, pt1, pt2), axis=1)
    return cld, choose

def get_bbox(bbox):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3] #r行 c列
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


if __name__ == "__main__":

    meta_file = open('gt.yml', 'r')
    meta = yaml.load(meta_file)
    rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])
    dpt = (np.array(Image.open('obj_01_depth.png')))[rmin:rmax, cmin:cmax]
    cld, choose = dpt_2_cld(dpt)
    normal=get_normal(cld)
    color_map=get_normal_map(normal,choose,rmin, rmax, cmin, cmax)

    # plt.imshow(color_map)
    # plt.show()
    img=cv2.merge([color_map[:,:,0],color_map[:,:,1],color_map[:,:,2]])
    cv2.imshow('test',img)
    cv2.waitKey(0)
    cv2.destroyWindow('test')
    #可视化点云和法线