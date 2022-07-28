import sys
sys.path.append("..")
import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
# from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
# from datasets.warehouse.dataset import PoseDataset as PoseDataset_warehouse
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
from tensorboardX import SummaryWriter

torch.cuda.set_device(0)  #关于显卡的奇怪报错，尝试设置指定显卡
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'linemod', help='ycb or warehouse or linemod')
parser.add_argument('--dataset_root', type=str, default = '/home/wuyi/wym/EFN6D/DataSet/Linemod_preprocessed', help='dataset root dir (''YCB_Video_Dataset'' or ''Warehouse_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate,lr*lr_rate')
parser.add_argument('--w', default=0.015, help='learning rate,论文loss中的w')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w，开启调整lr和w阈值，只调节一次')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train') #default=500
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()
# refine_margin太小会导致过度拟合 https://github.com/j96w/DenseFusion/issues/49

opt.start_epoch = 1 #等于1的时候会清空log文件
opt.refine_margin=0.008
# opt.lr_rate = 0.5
# opt.nepoch = 60
# pt.resume_posenet = '../trained_models/linemod/pose_model_6_0.011413681305847252.pth'
# opt.resume_refinenet = '../trained_models/linemod/pose_refine_model_23_0.011202031748817558.pth'
# data_mini = '_mini' #小数据集测试代码，正式训练时屏蔽
data_mini = ''
opt.dataset_root = '/home/wuyi/wym/EFN6D/DataSet/Linemod_preprocessed'+data_mini 
writer = SummaryWriter(logdir='./log')  # 服务器上pytorch1.0 只能用tensorboard1.7以前，所以写法变成logdir，1.7以后是log_dir
# shownet_flag = False #是否可视化网络


def main():
    opt.manualSeed = random.randint(1, 100)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = '../trained_models/ycb' #folder to save trained models
        opt.log_dir = '../experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training
    elif opt.dataset == 'warehouse':
        opt.num_objects = 13
        opt.num_points = 1000
        opt.outf = '../trained_models/warehouse'
        opt.log_dir = '../experiments/logs/warehouse'
        opt.repeat_epoch = 1
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500 #500
        opt.outf = '../trained_models/linemod'+data_mini
        opt.log_dir = '../experiments/logs/linemod'+data_mini
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return

    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    refiner.cuda()
    torch.save(estimator, '{0}/pose_model_all.pth'.format(opt.outf)) # 保存带结构的完整模型
    torch.save(refiner, '{0}/pose_refine_model_all.pth'.format(opt.outf))

    if opt.resume_posenet != '':
        #estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
        estimator.load_state_dict(torch.load('{0}'.format(opt.resume_posenet)))
    if opt.resume_refinenet != '':
        #refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        refiner.load_state_dict(torch.load('{0}'.format(opt.resume_refinenet)))
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'warehouse':
        dataset = PoseDataset_warehouse('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers) #封装成batch
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'warehouse':
        test_dataset = PoseDataset_warehouse('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()
    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(dataloader, 0):
                # points：由深度图计算出来的点云，该点云数据以摄像头为参考坐标
                # choose：所选择点云的索引，
                # img：通过box剪切下来的RGB图像
                # target：根据model_points点云信息，以及旋转偏移矩阵转换过的点云信息
                # model_points：目标初始帧（模型）对应的点云信息
                # idx：目标物体的序列编号
                points, choose, img, target, model_points, idx, normal_map = data
                points, choose, img, target, model_points, idx, normal_map = Variable(points).cuda(),\
                                                                            Variable(choose).cuda(), \
                                                                            Variable(img).cuda(), \
                                                                            Variable(target).cuda(), \
                                                                            Variable(model_points).cuda(), \
                                                                            Variable(idx).cuda(), \
                                                                            Variable(normal_map).cuda()
                pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx, normal_map)
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                        dis.backward()
                else:
                    loss.backward()

                train_dis_avg += dis.item()
                train_count += 1

                if train_count % opt.batch_size == 0:
                    Frame_my=int(train_count / opt.batch_size)
                    Avg_dis_my=train_dis_avg / opt.batch_size
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, Frame_my, train_count, Avg_dis_my))
                    writer.add_scalar('train/Avg_dis', Avg_dis_my, epoch*100000+Frame_my) #添加tensorboard
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

                # if(shownet_flag==True):# tensorboard 可视化网络
                #     global shownet_flag
                #     shownet_flag=False
                #     writer.add_graph(estimator, (img, points, choose, idx,)) #最后一个参数决定是否打印在控制台
                #     #writer.add_graph(refiner, (new_points, emb, idx,))
                #     writer.close()
        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target, model_points, idx, normal_map = data
            points, choose, img, target, model_points, idx, normal_map = Variable(points).cuda(), \
                                                                    Variable(choose).cuda(), \
                                                                    Variable(img).cuda(), \
                                                                    Variable(target).cuda(), \
                                                                    Variable(model_points).cuda(), \
                                                                    Variable(idx).cuda(), \
                                                                    Variable(normal_map).cuda()
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx, normal_map)
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

            test_dis += dis.item()
            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))
            writer.add_scalar('test/dis', dis, epoch*100000+test_count)  # 添加tensorboard

            test_count += 1

        test_dis = test_dis / test_count
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        if test_dis <= best_test:
            best_test = test_dis
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            if opt.dataset == 'ycb':
                dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            if opt.dataset == 'warehouse':
                dataset = PoseDataset_warehouse('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            elif opt.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
            if opt.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            if opt.dataset == 'warehouse':
                test_dataset = PoseDataset_warehouse('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)


if __name__ == '__main__':
    main()
    writer.close()
