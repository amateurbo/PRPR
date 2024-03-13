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

save_dir = '/mnt/data/code/reidlog/mlr_dukemtmc_HAN/test'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True

distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/DukeMTMC-reID/MLR_DukeMTMC',
    model_load_weights='/mnt/data/code/reidlog/MLR_DukeMTMC/x0/2022-03-13-14-46-24model/model.pth.tar-219',
    dataset='MLR_DukeMTMC'
)

distmat_hr4_HAN, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_dukemtmc_HAN/x4',
    model_load_weights='/mnt/data/code/reidlog/mlr_dukemtmc_HAN/x4/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC',
)

distmat_hr3_HAN, _, _, _, _, _, _, _, _, qflr1, gflr1 = cwd.main(
    root='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_dukemtmc_HAN/x3',
    model_load_weights='/mnt/data/code/reidlog/mlr_dukemtmc_HAN/x3/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC',
)

distmat_hr2_HAN, _, _, _, _, _, _, _, _, qflr2, gflr2 = cwd.main(
    root='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_dukemtmc_HAN/x2',
    model_load_weights='/mnt/data/code/reidlog/mlr_dukemtmc_HAN/x2/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC',
)

distmat_hr4, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_DukeMTMC-reID_new_sr_myself_l2_nofreeze_r_100/x4',
    model_load_weights='/mnt/data/code/reidlog/mlr_dukemtmc_new_sr_myself_l2_nofreeze_r_100/x4/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC',
)

distmat_hr3, _, _, _, _, _, _, _, _, qflr1, gflr1 = cwd.main(
    root='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_DukeMTMC-reID_new_sr_myself_l2_nofreeze_r_100/x3',
    model_load_weights='/mnt/data/code/reidlog/mlr_dukemtmc_new_sr_myself_l2_nofreeze_r_100/x3/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC',
)

distmat_hr2, _, _, _, _, _, _, _, _, qflr2, gflr2 = cwd.main(
    root='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_DukeMTMC-reID_new_sr_myself_l2_nofreeze_r_100/x2',
    model_load_weights='/mnt/data/code/reidlog/mlr_dukemtmc_new_sr_myself_l2_nofreeze_r_100/x2/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC',
)


# distmat
print('dist ori')
cwd.computeCMC(distmatori, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')

# distmathr
print('distmatori + distmat_hr4_HAN')
cwd.computeCMC(distmatori + distmat_hr4_HAN, q_pids, g_pids, q_camids, g_camids,)
print('distmatori + distmat_hr4')
cwd.computeCMC(distmatori + distmat_hr4, q_pids, g_pids, q_camids, g_camids,)
print('distmat_hr4_HAN + distmat_hr4')
cwd.computeCMC(distmat_hr4_HAN + distmat_hr4, q_pids, g_pids, q_camids, g_camids,)
print('distmatori + distmat_hr4_HAN + distmat_hr4')
cwd.computeCMC(distmatori + distmat_hr4_HAN + distmat_hr4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')

# distmathr
print('distmatori + distmat_hr3_HAN')
cwd.computeCMC(distmatori + distmat_hr3_HAN, q_pids, g_pids, q_camids, g_camids,)
print('distmatori + distmat_hr3')
cwd.computeCMC(distmatori + distmat_hr3, q_pids, g_pids, q_camids, g_camids,)
print('distmat_hr3_HAN + distmat_hr3')
cwd.computeCMC(distmat_hr3_HAN + distmat_hr3, q_pids, g_pids, q_camids, g_camids,)
print('distmatori + distmat_hr3_HAN + distmat_hr3')
cwd.computeCMC(distmatori + distmat_hr3_HAN + distmat_hr3, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')


print('distmatori + distmat_hr2_HAN')
cwd.computeCMC(distmatori + distmat_hr2_HAN, q_pids, g_pids, q_camids, g_camids,)
print('distmatori + distmat_hr2')
cwd.computeCMC(distmatori + distmat_hr3, q_pids, g_pids, q_camids, g_camids,)
print('distmat_hr2_HAN + distmat_hr2')
cwd.computeCMC(distmat_hr2_HAN + distmat_hr3, q_pids, g_pids, q_camids, g_camids,)
print('distmatori + distmat_hr2_HAN + distmat_hr2')
cwd.computeCMC(distmatori + distmat_hr2_HAN + distmat_hr2, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')


print('distmatori + distmat_hr4_HAN + distmat_hr4 + distmat_hr3_HAN + distmat_hr3 ')
cwd.computeCMC(distmatori + distmat_hr4_HAN + distmat_hr4 + distmat_hr3_HAN + distmat_hr3, q_pids, g_pids, q_camids, g_camids,)
print('distmatori + distmat_hr4_HAN + distmat_hr4 + distmat_hr2_HAN + distmat_hr2')
cwd.computeCMC(distmatori + distmat_hr4_HAN + distmat_hr4 + distmat_hr2_HAN + distmat_hr2, q_pids, g_pids, q_camids, g_camids,)
print('distmatori + distmat_hr3_HAN + distmat_hr3 + distmat_hr2_HAN + distmat_hr2')
cwd.computeCMC(distmatori + distmat_hr3_HAN + distmat_hr3 + distmat_hr2_HAN + distmat_hr2, q_pids, g_pids, q_camids, g_camids,)
print('distmatori + distmat_hr4_HAN + distmat_hr4 + distmat_hr3_HAN + distmat_hr3 + distmat_hr2_HAN + distmat_hr2')
cwd.computeCMC(distmatori + distmat_hr4_HAN + distmat_hr4 + distmat_hr3_HAN + distmat_hr3 + distmat_hr2_HAN + distmat_hr2, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')


print('distmatori * 2 + distmat_hr4_HAN + distmat_hr4')
cwd.computeCMC(distmatori * 2 + distmat_hr4_HAN + distmat_hr4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmatori * 3 + distmat_hr4_HAN + distmat_hr4 + distmat_hr3_HAN + distmat_hr3 + distmat_hr2_HAN + distmat_hr2')
cwd.computeCMC(distmatori * 3 + distmat_hr4_HAN + distmat_hr4 + distmat_hr3_HAN + distmat_hr3, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmatori * 4 + distmat_hr4_HAN + distmat_hr4 + distmat_hr3_HAN + distmat_hr3 + distmat_hr2_HAN + distmat_hr2')
cwd.computeCMC(distmatori * 4 + distmat_hr4_HAN + distmat_hr4 + distmat_hr3_HAN + distmat_hr3 + distmat_hr2_HAN + distmat_hr2, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')