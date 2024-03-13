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
from scripts.mlr_dukemtmc_new import MLR_DukeMTMC
# from MLR_Market1501 import MLR_Market1501
from torchreid import metrics
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

save_dir = '/mnt/data/code/reidlog/mlr_dukemtmc_new_sr_myself_l2_nofreeze_x8/test'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True

distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/DukeMTMC-reID/MLR_DukeMTMC',
    model_load_weights='/mnt/data/code/reidlog/MLR_DukeMTMC/x0/2022-03-13-14-46-24model/model.pth.tar-219',
    dataset='MLR_DukeMTMC'
    # rerank=True
)



distmat_hr8, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
    root='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_DukeMTMC-reID_new_sr_myself_l2_nofreeze_x8/x8',
    model_load_weights='/mnt/data/code/reidlog/mlr_dukemtmc_new_sr_myself_l2_nofreeze_x8/x8/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC',
    # rerank=True
)

distmat_hr4, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
    root='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_DukeMTMC-reID_new_sr_myself_l2_nofreeze_x8/x4',
    model_load_weights='/mnt/data/code/reidlog/mlr_dukemtmc_new_sr_myself_l2_nofreeze_x8/x4/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC',
    # rerank=True
)

distmat_hr3, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
    root='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_DukeMTMC-reID_new_sr_myself_l2_nofreeze_x8/x3',
    model_load_weights='/mnt/data/code/reidlog/mlr_dukemtmc_new_sr_myself_l2_nofreeze_x8/x3/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC',
    # rerank=True
)

distmat_hr2, _, _, _, _, _, _, _, _, qfx8, gfx8 = cwd.main(
    root='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_DukeMTMC-reID_new_sr_myself_l2_nofreeze_x8/x2',
    model_load_weights='/mnt/data/code/reidlog/mlr_dukemtmc_new_sr_myself_l2_nofreeze_x8/x2/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC'
    # rerank=True
)
# distmat
print('dist ori')
cwd.computeCMC(distmatori, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
# distmathr
print('dist hr8')
cwd.computeCMC(distmat_hr8, q_pids, g_pids, q_camids, g_camids,)
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
print('dist h8+h4')
computeCMC(distmat_hr8 + distmat_hr4, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist h8+h4+h3')
computeCMC(distmat_hr8 + distmat_hr4 + distmat_hr3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist h8+h4+h3+h2')
computeCMC(distmat_hr8 + distmat_hr4 + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')

print("直接加距离向量（考虑未划分前距离向量）：")
print('dist ori+h8')
computeCMC(distmatori + distmat_hr8, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+h8+h4')
computeCMC(distmatori + distmat_hr8 + distmat_hr4, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+h8+h4+h3')
computeCMC(distmatori + distmat_hr8 + distmat_hr4 + distmat_hr3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+h8+h4+h3+h2')
computeCMC(distmatori + distmat_hr8 + distmat_hr4 + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')


print("直接加距离向量（考虑未划分前距离向量）：")
print('dist ori+h8')
computeCMC(distmatori + distmat_hr8, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+h4')
computeCMC(distmatori + distmat_hr4, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+h3')
computeCMC(distmatori + distmat_hr3, q_pids, g_pids, q_camids, g_camids)
print('\n\n')
print('dist ori+h2')
computeCMC(distmatori + distmat_hr2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')

