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

save_dir = '/mnt/data/code/reidlog/MTA_reid_l1/test'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True

distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/MTA_reid',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid/model/model-best.pth.tar',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_hr4, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_l1/x4',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_l1/x4/model/model-best.pth.tar',
    dataset='MTA_reid_new',
)

distmat_hr3, _, _, _, _, _, _, _, _, qflr1, gflr1 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_l1/x3',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_l1/x3/model/model-best.pth.tar',
    dataset='MTA_reid_new',
)

distmat_hr2, _, _, _, _, _, _, _, _, qflr2, gflr2 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_l1/x2',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_l1/x2/model/model-best.pth.tar',
    dataset='MTA_reid_new',
)


# distmat
print('dist ori')
cwd.computeCMC(distmatori, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
# distmat_hr4
print('dist hr4')
cwd.computeCMC(distmat_hr4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
#distmatlr1
print('dist hr3')
cwd.computeCMC(distmat_hr3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
#distmatlr2
print('dist hr2')
cwd.computeCMC(distmat_hr2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')

print("直接加距离向量：")
print('dist hr+lr1')
computeCMC(distmat_hr4 + distmat_hr3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist hr+lr1+lr2')
computeCMC(distmat_hr4 + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')

print("直接加距离向量（考虑未划分前距离向量）：")
print('dist ori+hr')
computeCMC(distmatori + distmat_hr4, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+hr+lr1')
computeCMC(distmatori + distmat_hr4 + distmat_hr3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+hr+lr1+lr2')
computeCMC(distmatori + distmat_hr4 + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')


print("直接加距离向量（考虑未划分前距离向量）：")
print('dist ori+hr')
computeCMC(distmatori + distmat_hr4, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+lr1')
computeCMC(distmatori + distmat_hr3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+lr2')
computeCMC(distmatori + distmat_hr2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')


#-----------ori----------
print('----------------------------------------------------')
print('---------------------   ori   ----------------------')
print('----------------------------------------------------')
for i in range(5):
    k1 = i * 0.2
    k2 = 1
    print('dist ori*{} + hr_x4*{} + hr_x3*{} + hr_x2*{}'.format(k1, k2, k2, k2))
    computeCMC(distmatori * k1 + distmat_hr4 * k2 + distmat_hr3 * k2 + distmat_hr2 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(4):
    k1 = 1 + i * 0.5
    k2 = 1
    print('dist ori*{} + hr_x4*{} + hr_x3*{} + hr_x2*{}'.format(k1, k2, k2, k2))
    computeCMC(distmatori * k1 + distmat_hr4 * k2 + distmat_hr3 * k2 + distmat_hr2 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(8):
    k1 = 3 + i * 1
    k2 = 1
    print('dist ori*{} + hr_x4*{} + hr_x3*{} + hr_x2*{}'.format(k1, k2, k2, k2))
    computeCMC(distmatori * k1 + distmat_hr4 * k2 + distmat_hr3 * k2 + distmat_hr2 * k2,
        q_pids, g_pids, q_camids, g_camids)


print('----------------------------------------------------')
print('----------------------   x4   ----------------------')
print('----------------------------------------------------')
for i in range(5):
    k1 = i * 0.2
    k2 = 1
    print('dist ori*{} + hr_x4*{} + hr_x3*{} + hr_x2*{}'.format(k2, k1, k2, k2))
    computeCMC(distmatori * k2 + distmat_hr4 * k1 + distmat_hr3 * k2 + distmat_hr2 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(4):
    k1 = 1 + i * 0.5
    k2 = 1
    print('dist ori*{} + hr_x4*{} + hr_x3*{} + hr_x2*{}'.format(k2, k1, k2, k2))
    computeCMC(distmatori * k2 + distmat_hr4 * k1 + distmat_hr3 * k2 + distmat_hr2 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(8):
    k1 = 3 + i * 1
    k2 = 1
    print('dist ori*{} + hr_x4*{} + hr_x3*{} + hr_x2*{}'.format(k2, k1, k2, k2))
    computeCMC(distmatori * k2 + distmat_hr4 * k1 + distmat_hr3 * k2 + distmat_hr2 * k2,
        q_pids, g_pids, q_camids, g_camids)


print('----------------------------------------------------')
print('----------------------   x3   ----------------------')
print('----------------------------------------------------')
for i in range(5):
    k1 = i * 0.2
    k2 = 1
    print('dist ori*{} + hr_x4*{} + hr_x3*{} + hr_x2*{}'.format(k2, k2, k1, k2))
    computeCMC(distmatori * k2 + distmat_hr4 * k2 + distmat_hr3 * k1 + distmat_hr2 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(4):
    k1 = 1 + i * 0.5
    k2 = 1
    print('dist ori*{} + hr_x4*{} + hr_x3*{} + hr_x2*{}'.format(k2, k2, k1, k2))
    computeCMC(distmatori * k2 + distmat_hr4 * k2 + distmat_hr3 * k1 + distmat_hr2 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(8):
    k1 = 3 + i * 1
    k2 = 1
    print('dist ori*{} + hr_x4*{} + hr_x3*{} + hr_x2*{}'.format(k2, k2, k1, k2))
    computeCMC(distmatori * k2 + distmat_hr4 * k2 + distmat_hr3 * k1 + distmat_hr2 * k2,
        q_pids, g_pids, q_camids, g_camids)



print('----------------------------------------------------')
print('----------------------   x2   ----------------------')
print('----------------------------------------------------')
for i in range(5):
    k1 = i * 0.2
    k2 = 1
    print('dist ori*{} + hr_x4*{} + hr_x3*{} + hr_x2*{}'.format(k2, k2, k2, k1))
    computeCMC(distmatori * k2 + distmat_hr4 * k2 + distmat_hr3 * k2 + distmat_hr2 * k1,
        q_pids, g_pids, q_camids, g_camids)
for i in range(4):
    k1 = 1 + i * 0.5
    k2 = 1
    print('dist ori*{} + hr_x4*{} + hr_x3*{} + hr_x2*{}'.format(k2, k2, k2, k1))
    computeCMC(distmatori * k2 + distmat_hr4 * k2 + distmat_hr3 * k2 + distmat_hr2 * k1,
        q_pids, g_pids, q_camids, g_camids)
for i in range(8):
    k1 = 3 + i * 1
    k2 = 1
    print('dist ori*{} + hr_x4*{} + hr_x3*{} + hr_x2*{}'.format(k2, k2, k2, k1))
    computeCMC(distmatori * k2 + distmat_hr4 * k2 + distmat_hr3 * k2 + distmat_hr2 * k1,
        q_pids, g_pids, q_camids, g_camids)


