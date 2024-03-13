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

save_dir = '/mnt/data/code/reidlog/mlr_market1501_new_3/test'
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
    root='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_sr_down_3/hr',
    model_load_weights='/mnt/data/code/reidlog/mlr_market1501_new_3/hr/2022-09-12-22-50-59model/model-best.pth.tar',
    dataset='mlr_market1501',
    # rerank=True
)

distmatlr1, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
    root='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_sr_down_3/x2',
    model_load_weights='/mnt/data/code/reidlog/mlr_market1501_new_3/x2/2022-09-13-01-07-10model/model-best.pth.tar',
    dataset='mlr_market1501',
    # rerank=True
)

distmatlr2, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
    root='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_sr_down_3/x3',
    model_load_weights='/mnt/data/code/reidlog/mlr_market1501_new_3/x3/2022-09-13-03-23-47model/model-best.pth.tar',
    dataset='mlr_market1501',
    # rerank=True
)

distmatlr3, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
    root='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_sr_down_3/x4',
    model_load_weights='/mnt/data/code/reidlog/mlr_market1501_new_3/x4/2022-09-13-05-40-56model/model-best.pth.tar',
    dataset='mlr_market1501',
    # rerank=True
)
#
# distmatx8, _, _, _, _, _, _, _, _, qfx8, gfx8 = cwd.main(
#     root='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_sr_down_2/x8',
#     model_load_weights='/mnt/data/code/reidlog/mlr_market1501_new_2/x8/2022-05-20-06-01-44model/model-best.pth.tar',
#     dataset='mlr_market1501'
#     # rerank=True
# )
# distmat
print('dist ori')
cwd.computeCMC(distmatori, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
# distmathr
print('dist hr')
cwd.computeCMC(distmathr, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
#distmatlr1
print('dist lr1')
cwd.computeCMC(distmatlr1, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
#distmatlr2
print('dist lr2')
cwd.computeCMC(distmatlr2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
#distmatlr3
print('dist lr3')
cwd.computeCMC(distmatlr3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')

print("直接加距离向量：")
print('dist hr+lr1')
computeCMC(distmathr + distmatlr1, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist hr+lr1+lr2')
computeCMC(distmathr + distmatlr1 + distmatlr2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist hr+lr1+lr2+lr3')
computeCMC(distmathr + distmatlr1 + distmatlr2 + distmatlr3, q_pids, g_pids, q_camids, g_camids)
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

print("直接加距离向量（考虑未划分前距离向量）：")
print('dist ori+hr')
computeCMC(distmatori + distmathr, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+lr1')
computeCMC(distmatori + distmatlr1, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+lr2')
computeCMC(distmatori + distmatlr2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+lr3')
computeCMC(distmatori + distmatlr3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')

