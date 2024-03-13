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
from mlr_market1501_new import mlr_market1501
# from MLR_Market1501 import MLR_Market1501
from torchreid import metrics
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

save_dir = '/mnt/data/code/reidlog/mlr_market1501_new_2_new/test'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True

distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/market1501',
    model_load_weights='/mnt/data/code/deep-person-reid-master/testlog/2022-01-18-14-57-33model/model.pth.tar-222',
    dataset='mlr_market1501',
    # rerank=True
)

distmathr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_sr_down_2_new/hr',
    model_load_weights='/mnt/data/code/reidlog/mlr_market1501_new_2_new/hr/2022-08-31-16-54-51model/model-best.pth.tar',
    dataset='mlr_market1501',
    # rerank=True
)

distmatx2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
    root='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_sr_down_2_new/x2',
    model_load_weights='/mnt/data/code/reidlog/mlr_market1501_new_2_new/x2/2022-08-31-19-08-58model/model-best.pth.tar',
    dataset='mlr_market1501',
    # rerank=True
)

distmatx3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
    root='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_sr_down_2_new/x3',
    model_load_weights='/mnt/data/code/reidlog/mlr_market1501_new_2_new/x3/2022-08-31-21-21-27model/model-best.pth.tar',
    dataset='mlr_market1501',
    # rerank=True
)

distmatx4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
    root='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_sr_down_2_new/x4',
    model_load_weights='/mnt/data/code/reidlog/mlr_market1501_new_2_new/x4/2022-08-31-23-33-58model/model-best.pth.tar',
    dataset='mlr_market1501',
    # rerank=True
)

distmatx8, _, _, _, _, _, _, _, _, qfx8, gfx8 = cwd.main(
    root='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_sr_down_2_new/x8',
    model_load_weights='/mnt/data/code/reidlog/mlr_market1501_new_2_new/x8/2022-09-01-01-46-49model/model-best.pth.tar',
    dataset='mlr_market1501'
    # rerank=True
)
# distmat
print('dist ori')
cwd.computeCMC(distmatori, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
# distmathr
print('dist hr')
cwd.computeCMC(distmathr, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
#distmatx2
print('dist x2')
cwd.computeCMC(distmatx2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
#distmatx3
print('dist x3')
cwd.computeCMC(distmatx3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
#distmatx4
print('dist x4')
cwd.computeCMC(distmatx4, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
#distmatx8
print('dist x8')
cwd.computeCMC(distmatx8, q_pids, g_pids, q_camids, g_camids)
print('\n\n')


print("直接加距离向量：")
print('dist hr+x2')
computeCMC(distmathr + distmatx2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist hr+x2+x3')
computeCMC(distmathr + distmatx2 + distmatx3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist hr+x2+x3+x4')
computeCMC(distmathr + distmatx2 + distmatx3 + distmatx4, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist hr+x2+x3+x4+x8')
computeCMC(distmathr + distmatx2 + distmatx3 + distmatx4 +distmatx8, q_pids, g_pids, q_camids, g_camids)
print('\n\n')

print("直接加距离向量（考虑未划分前距离向量）：")
print('dist ori+hr')
computeCMC(distmatori + distmathr, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+hr+x2')
computeCMC(distmatori + distmathr + distmatx2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+hr+x2+x3')
computeCMC(distmatori + distmathr + distmatx2 + distmatx3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+hr+x2+x3+x4')
computeCMC(distmatori + distmathr + distmatx2 + distmatx3 + distmatx4, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+hr+x2+x3+x4+x8')
computeCMC(distmatori + distmathr + distmatx2 + distmatx3 + distmatx4 +distmatx8, q_pids, g_pids, q_camids, g_camids)
print('\n\n')

print("直接加距离向量（考虑未划分前距离向量）：")
print('dist ori+hr')
computeCMC(distmatori + distmathr, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+x2')
computeCMC(distmatori + distmatx2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+x3')
computeCMC(distmatori + distmatx3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+x4')
computeCMC(distmatori + distmatx4, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+x8')
computeCMC(distmatori + distmatx8, q_pids, g_pids, q_camids, g_camids)
print('\n\n')

print("直接加距离向量（考虑未划分前距离向量）：")
print('dist ori')
computeCMC(distmatori, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+x2')
computeCMC(distmatori + distmatx2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+x2+x3')
computeCMC(distmatori + distmatx2 + distmatx3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+x2+x3+x4')
computeCMC(distmatori + distmatx2 + distmatx3 + distmatx4, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+x2+x3+x4+x8')
computeCMC(distmatori + distmatx2 + distmatx3 + distmatx4 +distmatx8, q_pids, g_pids, q_camids, g_camids)
print('\n\n')