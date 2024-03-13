import os
import cv2
import sys
import os.path as osp
import numpy as np
import time
import torch
import torchreid
from scripts.mydatasets import MTA_reid_new
torchreid.data.register_image_dataset('MTA_reid_new', MTA_reid_new)
import scripts.compute_weight_distance as cwd
from scripts.compute_weight_distance import computeCMC
from scripts.mlr_market1501_new import mlr_market1501
# from MLR_Market1501 import MLR_Market1501
from torchreid import metrics
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

save_dir = '/mnt/data/code/reidlog/MTA_reid_HAN/test_a'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True

# distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
#     root='/mnt/data/datasets/MTA_reid',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid/model/model-best.pth.tar',
#     dataset='MTA_reid_new',
#     # rerank=True
# )

# distmat_hr22, _, _, _, _, _, _, _, _, qflr2, gflr2 = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_HAN_2/x2',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_HAN_2/x2/model/model-best.pth.tar',
#     dataset='MTA_reid_new',
# )
# #distmatlr22
# print('dist hr22')
# cwd.computeCMC(distmat_hr22, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')


# distmat_hr4, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_HAN/x4',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_HAN/x4/model/model-best.pth.tar',
#     dataset='MTA_reid_new',
# )

# distmat_hr3, _, _, _, _, _, _, _, _, qflr1, gflr1 = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_HAN/x3',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_HAN_2/x3/model/model-best.pth.tar',
#     dataset='MTA_reid_new',
# )
#
# distmat_hr2, _, _, _, _, _, _, _, _, qflr2, gflr2 = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_HAN/x2',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_HAN_2/x2/model/model-best.pth.tar',
#     dataset='MTA_reid_new',
# )


# distmat_hr4, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_HAN/x4',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_HAN/x4/model/model-best.pth.tar',
#     dataset='MTA_reid_new',
# )

# distmat_hr3, _, _, _, _, _, _, _, _, qflr1, gflr1 = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_HAN_2/x3',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_HAN_2/x3/model/model-best.pth.tar',
#     dataset='MTA_reid_new',
# )

distmat_hr2, q_pids, g_pids, q_camids, g_camids, _, _, _, _, qflr2, gflr2 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_HAN_2/x2',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_HAN_2/x2/model/model-best.pth.tar',
    dataset='MTA_reid_new',
)
print('dist distmat_hr2')
computeCMC(distmat_hr2, q_pids, g_pids, q_camids, g_camids)
print('dist distmat_hr3')
computeCMC(distmat_hr3, q_pids, g_pids, q_camids, g_camids)
print('dist distmat_hr4')
computeCMC(distmat_hr4, q_pids, g_pids, q_camids, g_camids)

# print("直接加距离向量（考虑未划分前距离向量）：")
# print('dist ori+hr+lr1+lr2')
# computeCMC(distmatori + distmat_hr4 + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori*2+hr+lr1+lr2')
# computeCMC(distmatori*2 + distmat_hr4 + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids)
#
# print('dist ori*2+hr+lr1+lr2+lr22')
# computeCMC(distmatori*2 + distmat_hr4 + distmat_hr3 + distmat_hr2 + distmat_hr22, q_pids, g_pids, q_camids, g_camids)
#
# print('\n\n')
# print('dist ori+hr')
# computeCMC(distmatori + distmat_hr4, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+hr+lr1')
# computeCMC(distmatori + distmat_hr4 + distmat_hr3, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
#
# print("直接加距离向量（考虑未划分前距离向量）：")
# print('dist ori+hr')
# computeCMC(distmatori + distmat_hr4, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+lr1')
# computeCMC(distmatori + distmat_hr3, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+lr2')
# computeCMC(distmatori + distmat_hr2, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
#
# #distmatlr2
# print('dist hr22')
# cwd.computeCMC(distmat_hr22, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# # distmat
# print('dist ori')
# cwd.computeCMC(distmatori, q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')
# # distmat_hr4
# print('dist hr4')
# cwd.computeCMC(distmat_hr4, q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')
# #distmatlr1
# print('dist hr3')
# cwd.computeCMC(distmat_hr3, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# #distmatlr2
# print('dist hr2')
# cwd.computeCMC(distmat_hr2, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
#
# print("直接加距离向量：")
# print('dist hr+lr1')
# computeCMC(distmat_hr4 + distmat_hr3, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist hr+lr1+lr2')
# computeCMC(distmat_hr4 + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')