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

import torchreid
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



