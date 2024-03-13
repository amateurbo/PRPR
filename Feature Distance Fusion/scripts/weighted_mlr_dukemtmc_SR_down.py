import os
import cv2
import sys
import os.path as osp
import numpy as np
import time
import torch
import torchreid

import compute_weight_distance as cwd
from compute_weight_distance import computeCMC
from mlr_dukemtmc_new import MLR_DukeMTMC
# from MLR_Market1501 import MLR_Market1501
from torchreid import metrics
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

save_dir = '/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/test_new'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True

def compute(
        datapath='',
        modelpath='',
):
    distmathr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
        root=osp.join(datapath, 'hr'),
        model_load_weights=osp.join(modelpath, 'hr/model/model-best.pth.tar'),
        dataset='MLR_DukeMTMC',
        # rerank=True
    )

    distmatlr1, _, _, _, _, _, _, _, _, qflr1, gflr1 = cwd.main(
        root=osp.join(datapath, 'lr1'),
        model_load_weights=osp.join(modelpath, 'lr1/model/model-best.pth.tar'),
        dataset='MLR_DukeMTMC',
        # rerank=True
    )

    distmatlr2, _, _, _, _, _, _, _, _, qflr2, gflr2 = cwd.main(
        root=osp.join(datapath, 'lr2'),
        model_load_weights=osp.join(modelpath, 'lr2/model/model-best.pth.tar'),
        dataset='MLR_DukeMTMC',
        # rerank=True
    )

    distmatlr3, _, _, _, _, _, _, _, _, qflr3, gflr3 = cwd.main(
        root=osp.join(datapath, 'lr3'),
        model_load_weights=osp.join(modelpath, 'lr3/model/model-best.pth.tar'),
        dataset='MLR_DukeMTMC',
        # rerank=True
    )
    print('dist hr')
    cwd.computeCMC(distmathr, q_pids, g_pids, q_camids, g_camids, )
    print('\n\n')
    # distmatlr1
    print('dist lr1')
    cwd.computeCMC(distmatlr1, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    # distmatlr2
    print('dist lr2')
    cwd.computeCMC(distmatlr2, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    # distmatlr3
    print('dist lr3')
    cwd.computeCMC(distmatlr3, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print("直接加距离向量（考虑未划分前距离向量）：")
    print('dist ori+hr')
    computeCMC(distmatori + distmathr, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1')
    computeCMC(distmatori + distmathr + distmatlr1, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1+lr2')
    computeCMC(distmatori + distmathr + distmatlr1 + distmatlr2, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1+lr2+lr3')
    computeCMC(distmatori + distmathr + distmatlr1 + distmatlr2 + distmatlr3, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')

    return distmatori + distmathr + distmatlr1 + distmatlr2 + distmatlr3, distmathr + distmatlr1 + distmatlr2 + distmatlr3

if __name__ == '__main__':


    distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
        root='/mnt/data/datasets/DukeMTMC-reID/MLR_DukeMTMC',
        model_load_weights='/mnt/data/code/reidlog/MLR_DukeMTMC/x0/2022-03-13-14-46-24model/model.pth.tar-219',
        dataset='MLR_DukeMTMC'
        # rerank=True
    )
    distmat_hr, distmat_hr_2 = compute(datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/hr',
                                       modelpath='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/hr')

    distmat_lr1, distmat_lr1_2 = compute(datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr1',
                                       modelpath='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/lr1')

    distmat_lr2, distmat_lr2_2 = compute(datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr2',
                                       modelpath='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/lr2')

    distmat_lr3, distmat_lr3_2 = compute(datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr3',
                                       modelpath='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/lr3')


    print('\n\n----------------------------------------------------------------------------\n')
    print('返回的是单个超分数据下面的 distmatori + distmathr + distmatlr1 + distmatlr2 + distmatlr3')
    print('dist ori+hr')
    computeCMC(distmatori + distmat_hr, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1')
    computeCMC(distmatori + distmat_hr + distmat_lr1, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1+lr2')
    computeCMC(distmatori + distmat_hr + distmat_lr1 + distmat_lr2, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1+lr2+lr3')
    computeCMC(distmatori + distmat_hr + distmat_lr1 + distmat_lr2 + distmat_lr3, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('----------------------------------------------------------------------------\n')
    print('返回的是单个超分数据下面的 distmatori + distmathr + distmatlr1 + distmatlr2 + distmatlr3 / 5')
    print('dist ori+hr')
    computeCMC(distmatori + distmat_hr/5, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1')
    computeCMC(distmatori + distmat_hr/5 + distmat_lr1/5, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1+lr2')
    computeCMC(distmatori + distmat_hr/5 + distmat_lr1/5 + distmat_lr2/5, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1+lr2+lr3')
    computeCMC(distmatori + distmat_hr/5 + distmat_lr1/5 + distmat_lr2/5 + distmat_lr3/5, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')



    print('\n\n----------------------------------------------------------------------------\n')
    print('返回的是单个超分数据下面的 distmathr + distmatlr1 + distmatlr2 + distmatlr3')
    print('dist ori+hr')
    computeCMC(distmatori + distmat_hr_2, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1')
    computeCMC(distmatori + distmat_hr_2 + distmat_lr1_2, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1+lr2')
    computeCMC(distmatori + distmat_hr_2 + distmat_lr1_2 + distmat_lr2_2, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1+lr2+lr3')
    computeCMC(distmatori + distmat_hr_2 + distmat_lr1_2 + distmat_lr2_2 + distmat_lr3_2, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('----------------------------------------------------------------------------\n')
    print('返回的是单个超分数据下面的 distmathr + distmatlr1 + distmatlr2 + distmatlr3 / 4')
    print('dist ori+hr')
    computeCMC(distmatori + distmat_hr_2/4, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1')
    computeCMC(distmatori + distmat_hr_2/4 + distmat_lr1_2/4, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1+lr2')
    computeCMC(distmatori + distmat_hr_2/4 + distmat_lr1_2/4 + distmat_lr2_2/4, q_pids, g_pids, q_camids, g_camids)
    print('\n\n')
    print('dist ori+hr+lr1+lr2+lr3')
    computeCMC(distmatori + distmat_hr_2/4 + distmat_lr1_2/4 + distmat_lr2_2/4 + distmat_lr3_2/4, q_pids, g_pids, q_camids,
               g_camids)
    print('\n\n')

# distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
#     root='/mnt/data/datasets/DukeMTMC-reID/MLR_DukeMTMC',
#     model_load_weights='/mnt/data/code/reidlog/MLR_DukeMTMC/x0/2022-03-13-14-46-24model/model.pth.tar-219',
#     dataset='MLR_DukeMTMC'
#     # rerank=True
# )
#
# distmathr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
#     root='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/hr',
#     model_load_weights='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/hr/2022-08-30-09-51-39model/model-best.pth.tar',
#     dataset='MLR_DukeMTMC',
#     # rerank=True
# )
#
# distmatlr1, _, _, _, _, _, _, _, _, qflr1, gflr1 = cwd.main(
#     root='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr1',
#     model_load_weights='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/lr1/2022-08-30-13-26-46model/model-best.pth.tar',
#     dataset='MLR_DukeMTMC',
#     # rerank=True
# )
#
# distmatlr2, _, _, _, _, _, _, _, _, qflr2, gflr2 = cwd.main(
#     root='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr2',
#     model_load_weights='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/lr2/2022-08-30-16-24-36model/model-best.pth.tar',
#     dataset='MLR_DukeMTMC',
#     # rerank=True
# )
#
# distmatlr3, _, _, _, _, _, _, _, _, qflr3, gflr3 = cwd.main(
#     root='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr3',
#     model_load_weights='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/lr3/2022-08-30-19-17-09model/model-best.pth.tar',
#     dataset='MLR_DukeMTMC',
#     # rerank=True
# )
#
# # distmat
# print('dist ori')
# cwd.computeCMC(distmatori, q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')
# # distmathr
# print('dist hr')
# cwd.computeCMC(distmathr, q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')
# #distmatlr1
# print('dist lr1')
# cwd.computeCMC(distmatlr1, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# #distmatlr2
# print('dist lr2')
# cwd.computeCMC(distmatlr2, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# #distmatlr3
# print('dist lr3')
# cwd.computeCMC(distmatlr3, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
#
#
# print("直接加距离向量：")
# print('dist hr+lr1')
# computeCMC(distmathr + distmatlr1, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist hr+lr1+lr2')
# computeCMC(distmathr + distmatlr1 + distmatlr2, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist hr+lr1+lr2+lr3')
# computeCMC(distmathr + distmatlr1 + distmatlr2 + distmatlr3, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
#
# print("直接加距离向量（考虑未划分前距离向量）：")
# print('dist ori+hr')
# computeCMC(distmatori + distmathr, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+hr+lr1')
# computeCMC(distmatori + distmathr + distmatlr1, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+hr+lr1+lr2')
# computeCMC(distmatori + distmathr + distmatlr1 + distmatlr2, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+hr+lr1+lr2+lr3')
# computeCMC(distmatori + distmathr + distmatlr1 + distmatlr2 + distmatlr3, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
#
# print("直接加距离向量（考虑未划分前距离向量）：")
# print('dist ori+hr')
# computeCMC(distmatori + distmathr, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+lr1')
# computeCMC(distmatori + distmatlr1, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+lr2')
# computeCMC(distmatori + distmatlr2, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+lr3')
# computeCMC(distmatori + distmatlr3, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
#
