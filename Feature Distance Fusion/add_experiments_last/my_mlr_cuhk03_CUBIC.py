import os
import cv2
import sys
import os.path as osp
import numpy as np
import time
import torch
import torchreid

import scripts.compute_weight_distance as cwd
from scripts.compute_weight_distance import computeCMC
from scripts.MLR_CUHK03 import MLR_CUHK03
# from MLR_CUHK03 import MLR_CUHK03
from torchreid import metrics
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

save_dir = '/mnt/data/code/reidlog/mlr_cuhk03_CUBIC/test_lastepoch'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True

distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/CUHK03',
    model_load_weights='/mnt/data/code/reidlog/MLR_CUHK03/x0/2022-03-13-15-48-23model/model.pth.tar-250',
    dataset='MLR_CUHK03',
    # rerank=True
)

distmat_hr4, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/datasets/CUHK03/mlr_cuhk03_CUBIC/x4',
    model_load_weights='/mnt/data/code/reidlog/mlr_cuhk03_CUBIC/x4/model/model.pth.tar-250',
    dataset='MLR_CUHK03',
)

distmat_hr3, _, _, _, _, _, _, _, _, qflr1, gflr1 = cwd.main(
    root='/mnt/data/datasets/CUHK03/mlr_cuhk03_CUBIC/x3',
    model_load_weights='/mnt/data/code/reidlog/mlr_cuhk03_CUBIC/x3/model/model.pth.tar-250',
    dataset='MLR_CUHK03',
)

distmat_hr2, _, _, _, _, _, _, _, _, qflr2, gflr2 = cwd.main(
    root='/mnt/data/datasets/CUHK03/mlr_cuhk03_CUBIC/x2',
    model_load_weights='/mnt/data/code/reidlog/mlr_cuhk03_CUBIC/x2/model/model.pth.tar-250',
    dataset='MLR_CUHK03',
)


# distmat
print('dist ori')
cwd.computeCMC(distmatori, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
# distmathr
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
print('dist ori*2+hr+lr1+lr2')
computeCMC(distmatori*2 + distmat_hr4 + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')





