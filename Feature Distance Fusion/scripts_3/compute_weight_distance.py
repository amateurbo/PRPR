from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import sys
import os
import re
import glob
import os.path as osp

from torchreid.data import ImageDataset
from torchreid import metrics
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from torchreid.utils import (
    MetricMeter, AverageMeter, re_ranking, open_all_layers, save_checkpoint,
    open_specified_layers, visualize_ranked_results
)

from scripts.default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)


class MTA_reid(ImageDataset):
    dataset_dir = 'MTA_reid'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)
        self.train_dir = osp.join(
            self.dataset_dir, 'train'
        )
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'test'
        )

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)


        super(MTA_reid, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))

        imgnames = os.listdir(dir_path)
        pid_con = set()
        for imgname in imgnames:
            pid = int(imgname.split('_')[5].split('.')[0]) - 1
            pid_con.add(pid)
        pidlabel = {pid: label for label, pid in enumerate(pid_con)}

        list = []
        for imgname in imgnames:
            imgpath = osp.join(dir_path, imgname)
            pid = int(imgname.split('_')[5].split('.')[0]) - 1
            camid = int(imgname.split('_')[3])
            # if((dir_path.split('/')[-1] == 'query')):
            #     camid = camid - 1
            # assert 0 <= camid <= 1
            if relabel:
                pid = pidlabel[pid]
            list.append((imgpath, pid, camid))

        return list



        # pattern = re.compile(r'([-\d]+)_c(\d)')
        #
        # pid_container = set()
        # for img_path in img_paths:
        #     pid, _ = map(int, pattern.search(img_path).groups())
        #     pid_container.add(pid)
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}
        #
        # data = []
        # for img_path in img_paths:
        #     pid, camid = map(int, pattern.search(img_path).groups())
        #     assert 1 <= camid <= 8
        #     camid -= 1  # index starts from 0
        #     if relabel:
        #         pid = pid2label[pid]
        #     data.append((img_path, pid, camid))
        #
        # return data



import torchreid
torchreid.data.register_image_dataset('MTA_reid', MTA_reid)

class MLR_DukeMTMC(ImageDataset):
    dataset_dir = 'MLR_DukeMTMC'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)
        self.train_dir = osp.join(
            self.dataset_dir, 'bounding_box_train'
        )
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'bounding_box_test'
        )

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)


        super(MLR_DukeMTMC, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):

        imgnames = os.listdir(dir_path)
        imgnames.sort()
        pid_con = set()
        for imgname in imgnames:
            pid = int(imgname.split('_')[0]) - 1
            pid_con.add(pid)
        pidlabel = {pid: label for label, pid in enumerate(pid_con)}

        list = []
        for imgname in imgnames:
            imgpath = osp.join(dir_path, imgname)
            pid = int(imgname.split('_')[0]) - 1
            camid = int(imgname.split('_')[1].split('c')[1]) - 1
            # if((dir_path.split('/')[-1] == 'query')):
            #     camid = camid - 1
            # assert 0 <= camid <= 1
            if relabel:
                pid = pidlabel[pid]
            list.append((imgpath, pid, camid))
        return list

torchreid.data.register_image_dataset('MLR_DukeMTMC', MLR_DukeMTMC)

# datamanager = torchreid.data.ImageDataManager(
#     root='/mnt/data/mlr_datasets/',
#     sources='new_dataset'
# )

import sys
import time
import os.path as osp
import argparse
import torch
import torch.nn as nn

import torchreid
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from scripts.default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)

@torch.no_grad()
def _evaluate(
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=False,
        compute_weight=False,
        save_distance=False,
        model=None
):
    batch_time = AverageMeter()

    def _feature_extraction(data_loader):
        f_, pids_, camids_ = [], [], []

        # 获取对应尺度
        size1_, size2_, size_ = [], [], []
        imgpaths_ = []

        torch.manual_seed("0")

        for batch_idx, data in enumerate(data_loader):

            imgs = data['img']
            pids = data['pid']
            camids = data['camid']
            imgpaths = data['impath']
            # imgs, pids, camids, imgpaths = parse_data_for_eval(data)
            use_gpu = True
            use_gpu = (torch.cuda.is_available() and use_gpu)
            if use_gpu:
                imgs = imgs.cuda()
            end = time.time()

            for i in range(len(imgpaths)):
                img = cv2.imread(imgpaths[i])
                size1 = img.shape[1]
                size2 = img.shape[0]
                size = size1 * size2
                size_.append(size)
                size1_.append(size1)
                size2_.append(size2)

            imgpaths_.extend(imgpaths)

            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.cpu().clone()
            f_.append(features)
            pids_.extend(pids)
            camids_.extend(camids)
        f_ = torch.cat(f_, 0)
        pids_ = np.asarray(pids_)
        camids_ = np.asarray(camids_)
        # return f_, pids_, camids_
        return f_, pids_, camids_, size1_, size2_, size_, imgpaths_

    print('Extracting features from query set ...')
    # qf, q_pids, q_camids, = _feature_extraction(query_loader)
    qf, q_pids, q_camids, qsize1_, qsize2_, qsize_, qimgpaths = _feature_extraction(query_loader)
    print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

    print('Extracting features from gallery set ...')
    # gf, g_pids, g_camids, = _feature_extraction(gallery_loader)
    gf, g_pids, g_camids, gsize1_, gsize2_, gsize_, gimgpaths = _feature_extraction(gallery_loader)
    print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

    if normalize_feature:
        print('Normalzing features with L2 norm ...')
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    print(
        'Computing distance matrix with metric={} ...'.format(dist_metric)
    )
    distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
    distmat = distmat.numpy()

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
        distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
        distmat = re_ranking(distmat, distmat_qq, distmat_gg)

    return distmat, q_pids, g_pids, q_camids, g_camids, \
           qsize1_, qsize2_, qsize_, gsize_, \
           qf, gf #返回特征

def main(
        save_dir='/mnt/data/code/reidlog/MTA-reid/test',
        root='/mnt/data/datasets',
        model_load_weights='/mnt/data/code/reidlog/MTA-reid/2022-02-24-20-05-16model/model.pth.tar-59',
        dataset='MTA_reid',
        rerank=False
         ):

    use_gpu = torch.cuda.is_available()
    # print('Collecting env info ...')
    # print('** System info **\n{}\n'.format(collect_env_info()))
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    datamanager = torchreid.data.ImageDataManager(
        root=root,
        sources=dataset,
        height=256,
        width=128,
        batch_size_train=64,
        batch_size_test=300,
        transforms=[]
    )

    print(root)
    model_name = 'osnet_x1_0'
    loss_name = 'softmax'
    model_pretrained = True

    print('Building model: {}'.format(model_name))
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=datamanager.num_train_pids,
        loss=loss_name,
        pretrained=model_pretrained,
        use_gpu=use_gpu
    )

    data_height, data_width = 256, 128

    num_params, flops = compute_model_complexity(
        model, (1, 3, data_height, data_width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))
    if model_load_weights and check_isfile(model_load_weights):
        load_pretrained_weights(model, model_load_weights)
    if use_gpu:
        model = nn.DataParallel(model).cuda()


    model.eval()
    query_loader = datamanager.test_loader[dataset]['query']
    gallery_loader = datamanager.test_loader[dataset]['gallery']
    distmat, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = _evaluate(
        dataset_name=dataset,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        dist_metric='cosine',
        normalize_feature=True,
        visrank=False,
        visrank_topk=10,
        save_dir=save_dir,
        use_metric_cuhk03=False,
        ranks=[1, 5, 10, 20],
        rerank=rerank,
        compute_weight=False,
        save_distance=False,
        model = model
    )

    return distmat, q_pids, g_pids, q_camids, g_camids,\
           qsize1_, qsize2_, qsize_, gsize_, qf, gf



def computeCMC(distmat,q_pids,g_pids,q_camids,g_camids):
    print('Computing CMC and mAP ...')
    cmc, mAP = metrics.evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        use_metric_cuhk03=False
    )
    ranks = [1, 5, 10, 20]
    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))




if __name__ == '__main__':

    save_dir = '/mnt/data/code/reidlog/MTA-reid/test'
    log_name = 'test.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(save_dir, log_name))
    weight = True

    distmat, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = main(
        root='/mnt/data/datasets',
        model_load_weights='/mnt/data/code/reidlog/MTA-reid/2022-02-24-20-05-16model/model.pth.tar-59',
    )
    # print('dist0')
    # computeCMC(distmat, q_pids, g_pids, q_camids, g_camids,)
    # distmatx2, _, _, _, _, _, _, _, _, qfx2, gfx2 = main(
    #     root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_x2_5200_E',
    #     model_load_weights='/mnt/data/code/reidlog/MTA_reid/x2/osnet_x1_0_x2_5200_E_x2_MTA_reid_softmax_cosinelr/2022-03-01-10-20-14model/model.pth.tar-55',
    # )
    #
    # distmatx3, _, _, _, _, _, _, _, _, qfx3, gfx3 = main(
    #     root='/mnt/data/code/SwinIR-main_2/results/swinir_classical_sr_x3_2400_E',
    #     model_load_weights='/mnt/data/code/reidlog/MTA_reid/x3/osnet_x1_0_x3_2400_E_x3_MTA_reid_softmax_cosinelr/2022-03-02-17-37-48model/model.pth.tar-60',
    # )
    #
    # distmatx4, _, _, _, _, _, _, _, _, qfx4, gfx4 = main(
    #     root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_x4_3600_E',
    #     model_load_weights='/mnt/data/code/reidlog/MTA_reid/x4/osnet_x1_0_x4_3600_E_x4_MTA_reid_softmax_cosinelr/2022-03-01-12-24-17model/model.pth.tar-60',
    # )
    #
    # distmatx8, _, _, _, _, _, _, _, _, qfx8, gfx8 = main(
    #     root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_x8_2200_E',
    #     model_load_weights='/mnt/data/code/reidlog/MTA_reid/x8/osnet_x1_0_x8_2200_E_x8_MTA_reid_softmax_cosinelr/2022-03-05-12-21-02model/model.pth.tar-60',
    # )


    distmatx2, _, _, _, _, _, _, _, _, qfx2, gfx2 = main(
        root='/mnt/data/code/SwinIR_results/results_MTA_reid/swinir_classical_sr_x2_5200_E',
        model_load_weights='/mnt/data/code/reidlog/MTA_reid/x2/osnet_x1_0_x2_5200_E_x2_MTA_reid_softmax_cosinelr/2022-03-01-10-20-14model/model.pth.tar-55',
    )

    distmatx3, _, _, _, _, _, _, _, _, qfx3, gfx3 = main(
        # root='/mnt/data/code/SwinIR-main_2/results/swinir_classical_sr_x3_2400_E',
        # model_load_weights='/mnt/data/code/reidlog/MTA_reid/x3/osnet_x1_0_x3_2400_E_x3_MTA_reid_softmax_cosinelr/2022-03-02-17-37-48model/model.pth.tar-60',
        root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_MTA_reid_x3_4600_E',
        model_load_weights='/mnt/data/code/reidlog/MTA_reid/x3/osnet_x1_0_x3_4600_E_x3_MTA_reid_softmax_cosinelr/2022-03-11-14-05-23model/model.pth.tar-56'
    )

    distmatx4, _, _, _, _, _, _, _, _, qfx4, gfx4 = main(
        # root='/mnt/data/code/SwinIR_results/results_MTA_reid/swinir_classical_sr_x4_3600_E',
        # model_load_weights='/mnt/data/code/reidlog/MTA_reid/x4/osnet_x1_0_x4_3600_E_x4_MTA_reid_softmax_cosinelr/2022-03-01-12-24-17model/model.pth.tar-60',
        root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_MTA_reid_x4_4800_E',
        model_load_weights='/mnt/data/code/reidlog/MTA_reid/x4/osnet_x1_0_x4_4800_E_x4_MTA_reid_softmax_cosinelr/2022-03-11-14-01-46model/model.pth.tar-54',
    )

    distmatx8, _, _, _, _, _, _, _, _, qfx8, gfx8 = main(
        # root='/mnt/data/code/SwinIR_results/results_MTA_reid/swinir_classical_sr_x8_2200_E',
        # model_load_weights='/mnt/data/code/reidlog/MTA_reid/x8/osnet_x1_0_x8_2200_E_x8_MTA_reid_softmax_cosinelr/2022-03-05-12-21-02model/model.pth.tar-60',
        root='/mnt/data/code/SwinIR_results/results_MTA_reid/swinir_classical_sr_x8_4000_E',
        model_load_weights='/mnt/data/code/reidlog/MTA_reid/x8/osnet_x1_0_x8_4000_E_x8_MTA_reid_softmax_cosinelr/2022-03-04-23-33-09model/model.pth.tar-60',
    )
    print('\nComputing distmat_qq and distmat_gg')

    start = time.time()
    distmat_qq = metrics.compute_distance_matrix(qf, qf, 'cosine')
    end = time.time()
    print('单次计算distmat_qq耗费时间：',end-start)
    start = time.time()
    distmat_gg = metrics.compute_distance_matrix(gf, gf, 'cosine')
    end = time.time()
    print('单次计算distmat_gg耗费时间：',end-start)
    print('还需对distmat_qq、distmat_gg分别计算4次')
    distmat_qqx2 = metrics.compute_distance_matrix(qfx2, qfx2, 'cosine')
    distmat_ggx2 = metrics.compute_distance_matrix(gfx2, gfx2, 'cosine')
    distmat_qqx3 = metrics.compute_distance_matrix(qfx3, qfx3, 'cosine')
    distmat_ggx3 = metrics.compute_distance_matrix(gfx3, gfx3, 'cosine')
    distmat_qqx4 = metrics.compute_distance_matrix(qfx4, qfx4, 'cosine')
    distmat_ggx4 = metrics.compute_distance_matrix(gfx4, gfx4, 'cosine')
    distmat_qqx8 = metrics.compute_distance_matrix(qfx8, qfx8, 'cosine')
    distmat_ggx8 = metrics.compute_distance_matrix(gfx8, gfx8, 'cosine')

    print('\n直接加dist0 x2348 单独 Applying person re-ranking ...')
    start = time.time()
    distmat02348 = re_ranking(distmat + distmatx2 + distmatx3 + distmatx4 + distmatx8,
                                  distmat_qq + distmat_qqx2 + distmat_qqx3 + distmat_qqx4 + distmat_qqx8,
                                  distmat_gg + distmat_ggx2 + distmat_ggx3 + distmat_ggx4 + distmat_ggx8)
    end1 = time.time()
    print('单次rerank耗费时间：',end1-start)
    computeCMC(distmat02348, q_pids, g_pids, q_camids, g_camids, )
    end2 = time.time()
    print('单次计算CMC曲线耗费时间：',end2-end1)


    #distmat0
    print('dist0')
    computeCMC(distmat, q_pids, g_pids, q_camids, g_camids,)
    #distmatx2
    print('distx2')
    computeCMC(distmatx2, q_pids, g_pids, q_camids, g_camids)
    #distmatx3
    print('distx3')
    computeCMC(distmatx3, q_pids, g_pids, q_camids, g_camids)
    #distmatx4
    print('distx4')
    computeCMC(distmatx4, q_pids, g_pids, q_camids, g_camids)
    #distmatx8
    print('distx8')
    computeCMC(distmatx8, q_pids, g_pids, q_camids, g_camids)

    # qfx234 = np.concatenate((qfx2, qfx3, qfx4), axis=1)
    # gfx234 = np.concatenate((gfx2, gfx3, gfx4), axis=1)
    #
    # qf0x234 = np.concatenate((qf, qfx2, qfx3, qfx4), axis=1)
    # gf0x234 = np.concatenate((gf, gfx2, gfx3, gfx4), axis=1)
    #
    # qf0x2348 = np.concatenate((qf, qfx2, qfx3, qfx4, qfx8), axis=1)
    # gf0x2348 = np.concatenate((gf, gfx2, gfx3, gfx4, gfx8), axis=1)
    #
    # concat_feature_x234_distance = metrics.compute_distance_matrix(torch.from_numpy(qfx234), torch.from_numpy(gfx234), 'cosine')
    # concat_feature_x234_distance = concat_feature_x234_distance.numpy()
    # print('concat_feature_x234_distance_cosine')
    # computeCMC(concat_feature_x234_distance, q_pids, g_pids, q_camids, g_camids, )
    #
    # concat_feature_0x234_distance = metrics.compute_distance_matrix(torch.from_numpy(qf0x234), torch.from_numpy(gf0x234), 'cosine')
    # concat_feature_0x234_distance = concat_feature_0x234_distance.numpy()
    # print('\nconcat_feature_0x234_distance_cosine')
    # computeCMC(concat_feature_0x234_distance, q_pids, g_pids, q_camids, g_camids, )
    #
    # concat_feature_0x2348_distance = metrics.compute_distance_matrix(torch.from_numpy(qf0x2348), torch.from_numpy(gf0x2348), 'cosine')
    # concat_feature_0x2348_distance = concat_feature_0x2348_distance.numpy()
    # print('\nconcat_feature_0x2348_distance_cosine')
    # computeCMC(concat_feature_0x2348_distance, q_pids, g_pids, q_camids, g_camids, )
    #
    #
    #
    # concat_feature_x234_distance = metrics.compute_distance_matrix(torch.from_numpy(qfx234), torch.from_numpy(gfx234), 'euclidean')
    # concat_feature_x234_distance = concat_feature_x234_distance.numpy()
    # print('concat_feature_x234_distance_euclidean')
    # computeCMC(concat_feature_x234_distance, q_pids, g_pids, q_camids, g_camids, )
    #
    # concat_feature_0x234_distance = metrics.compute_distance_matrix(torch.from_numpy(qf0x234), torch.from_numpy(gf0x234), 'euclidean')
    # concat_feature_0x234_distance = concat_feature_0x234_distance.numpy()
    # print('\nconcat_feature_0x234_distance_euclidean')
    # computeCMC(concat_feature_0x234_distance, q_pids, g_pids, q_camids, g_camids, )
    #
    # concat_feature_0x2348_distance = metrics.compute_distance_matrix(torch.from_numpy(qf0x2348), torch.from_numpy(gf0x2348), 'euclidean')
    # concat_feature_0x2348_distance = concat_feature_0x2348_distance.numpy()
    # print('\nconcat_feature_0x2348_distance_cosine')
    # computeCMC(concat_feature_0x2348_distance, q_pids, g_pids, q_camids, g_camids, )




    #
    # #distmat0
    # print('dist0')
    # computeCMC(distmat, q_pids, g_pids, q_camids, g_camids,)
    # #distmatx2
    # print('distx2')
    # computeCMC(distmatx2, q_pids, g_pids, q_camids, g_camids)
    # #distmatx3
    # print('distx3')
    # computeCMC(distmatx3, q_pids, g_pids, q_camids, g_camids)
    # #distmatx4
    # print('distx4')
    # computeCMC(distmatx4, q_pids, g_pids, q_camids, g_camids)
    # #distmatx8
    # print('distx8')
    # computeCMC(distmatx8, q_pids, g_pids, q_camids, g_camids)
    #
    # #距离向量直接相加
    #distmatx2 +0
    print('distx2+0')
    computeCMC(distmatx2 + distmat, q_pids, g_pids, q_camids, g_camids)
    #distmatx3
    print('distx3+0')
    computeCMC(distmatx3 + distmat, q_pids, g_pids, q_camids, g_camids)
    #distmatx4
    print('distx4+0')
    computeCMC(distmatx4 + distmat, q_pids, g_pids, q_camids, g_camids)
    #distmatx8
    print('distx8+0')
    computeCMC(distmatx8 + distmat, q_pids, g_pids, q_camids, g_camids)
    print('\n直接加dist0 x23')
    computeCMC(distmat + distmatx2 + distmatx3, q_pids, g_pids, q_camids, g_camids, )
    print('\n直接加dist0 x234')
    computeCMC(distmat + distmatx2 + distmatx3 + distmatx4, q_pids, g_pids, q_camids, g_camids, )
    print('\n直接加dist0 x2348')
    computeCMC(distmat + distmatx2 + distmatx3 + distmatx4 + distmatx8, q_pids, g_pids, q_camids, g_camids, )

    # #距离向量直接相加
    # print('\n直接加distx234')
    # computeCMC(distmatx2 + distmatx3 + distmatx4, q_pids, g_pids, q_camids, g_camids, )
    # print('\n直接加dist0 x234')
    # computeCMC(distmat + distmatx2 + distmatx3 + distmatx4, q_pids, g_pids, q_camids, g_camids, )
    # print('\n直接加dist0 x2348')
    # computeCMC(distmat + distmatx2 + distmatx3 + distmatx4 + distmatx8, q_pids, g_pids, q_camids, g_camids, )
    #
    # gsize_ = np.array(gsize_)
    # qsize1_ = np.array(qsize1_)
    # qsize2_ = np.array(qsize2_)
    # qsize_ = np.array(qsize_)
    # average = np.average(gsize_)
    # bili = qsize_ / average
    # r = np.sqrt(qsize_ / average)
    # r_m1 = r - 1
    # r_m2 = r - 1 / 2
    # r_m3 = r - 1 / 3
    # r_m4 = r - 1 / 4
    # r_m8 = r - 1 / 8
    #
    # #原始公式
    # dt = 0.2
    # print('计算权重，dt=' + str(dt))
    # w1 = np.exp(-pow(dt, 2) * pow(r_m1, 2))
    # w2 = np.exp(-pow(dt, 2) * pow(r_m2, 2))
    # w3 = np.exp(-pow(dt, 2) * pow(r_m3, 2))
    # w4 = np.exp(-pow(dt, 2) * pow(r_m4, 2))
    # w8 = np.exp(-pow(dt, 2) * pow(r_m8, 2))
    # print('\n\n原始方法距离向量加权......')
    # dist = distmat * w1.reshape(-1, 1)
    # distx2 = distmatx2 * w2.reshape(-1, 1)
    # distx3 = distmatx3 * w3.reshape(-1, 1)
    # distx4 = distmatx4 * w4.reshape(-1, 1)
    # distx8 = distmatx8 * w8.reshape(-1, 1)
    #
    # print('加权distx234')
    # computeCMC(distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids, )
    # print('\n加权dist0 x234')
    # computeCMC(dist + distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids, )
    # print('\n加权dist0 x2348')
    # computeCMC(dist + distx2 + distx3 + distx4 + distx8, q_pids, g_pids, q_camids, g_camids, )

    # #只保留最接近的权重
    # r_m1 = np.maximum(r_m1, -r_m1)
    # r_m2 = np.maximum(r_m2, -r_m2)
    # r_m3 = np.maximum(r_m3, -r_m3)
    # r_m4 = np.maximum(r_m4, -r_m4)
    # all_r = np.vstack((r_m1, r_m2, r_m3, r_m4))
    # min_r = np.min(all_r, axis=0).reshape(1, -1)
    # all_w = all_r * (all_r == min_r)
    # print('\n\n只保留最接近的权重加权......')
    # dist = distmat * all_w[0, :].reshape(-1, 1)
    # distx2 = distmatx2 * all_w[1, :].reshape(-1, 1)
    # distx3 = distmatx3 * all_w[2, :].reshape(-1, 1)
    # distx4 = distmatx4 * all_w[3, :].reshape(-1, 1)
    # print('加权distx234')
    # computeCMC(distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids, )
    # print('\n加权dist0 x234')
    # computeCMC(dist + distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids, )


    # #更改公式后加权
    # for i in range(1, 11):
    #     # --------更改公式
    #     dt = 0.02 * i
    #     print('\n\n更改公式后计算权重，dt=' + str(dt))
    #     r_m1 = np.maximum(r_m1, -r_m1)
    #     r_m2 = np.maximum(r_m2, -r_m2)
    #     r_m3 = np.maximum(r_m3, -r_m3)
    #     r_m4 = np.maximum(r_m4, -r_m4)
    #     # w1 = np.exp(-pow(dt, 2) * r_m1)
    #     # w2 = np.exp(-pow(dt, 2) * r_m2)
    #     # w3 = np.exp(-pow(dt, 2) * r_m3)
    #     # w4 = np.exp(-pow(dt, 2) * r_m4)
    #     w1 = np.exp(-dt * r_m1)
    #     w2 = np.exp(-dt * r_m2)
    #     w3 = np.exp(-dt * r_m3)
    #     w4 = np.exp(-dt * r_m4)
    #
    #     dist = distmat * w1.reshape(-1, 1)
    #     distx2 = distmatx2 * w2.reshape(-1, 1)
    #     distx3 = distmatx3 * w3.reshape(-1, 1)
    #     distx4 = distmatx4 * w4.reshape(-1, 1)
    #
    #     print('加权distx234')
    #     computeCMC(distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids, )
    #     print('\n加权dist0 x234')
    #     computeCMC(dist + distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids,)



