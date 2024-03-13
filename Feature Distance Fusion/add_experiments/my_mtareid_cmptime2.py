import os
import cv2
import sys
import os.path as osp
import numpy as np
import time
import torch
import torchreid
import timeit
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

save_dir = '/mnt/data/code/reidlog/MTA_reid_CUBIC/test_time'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True


start_time = time.time()
st1 = time.time()
distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/MTA_reid',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid/model/model-best.pth.tar',
    dataset='MTA_reid_new',
    # rerank=True
)
et1 = time.time()
print(f"ori程序耗时:{et1-st1:.4f}秒")


st1 = time.time()
distmat_hr4, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/datasets/MTA_reid/CUBIC_Result/x4',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_CUBIC/x4/model-best.pth.tar',
    dataset='MTA_reid_new',
)
et1 = time.time()
print(f"hr4程序耗时:{et1-st1:.4f}秒")

st1 = time.time()
distmat_hr3, _, _, _, _, _, _, _, _, qflr1, gflr1 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/CUBIC_Result/x3',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_CUBIC/x3/model-best.pth.tar',
    dataset='MTA_reid_new',
)
et1 = time.time()
print(f"hr3程序耗时:{et1-st1:.4f}秒")

st1 = time.time()
distmat_hr2, _, _, _, _, _, _, _, _, qflr2, gflr2 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/CUBIC_Result/x2',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_CUBIC/x2/model-best.pth.tar',
    dataset='MTA_reid_new',
)
et1 = time.time()
print(f"hr2程序耗时:{et1-st1:.4f}秒")

st1 = time.time()
# distmat
print('cmp distmat_hr4 + distmat_hr2')
# cwd.computeCMC(distmatori * 2 + distmat_hr4  + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids,)
cwd.computeCMC(distmat_hr4 + distmat_hr2 , q_pids, g_pids, q_camids, g_camids,)
et1 = time.time()
print(f"cmpCMC程序耗时:{et1-st1:.4f}秒")

st1 = time.time()
# distmat
print('cmp distmat_hr4 + distmat_hr3')
# cwd.computeCMC(distmatori * 2 + distmat_hr4  + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids,)
cwd.computeCMC(distmat_hr4 + distmat_hr3 , q_pids, g_pids, q_camids, g_camids,)
et1 = time.time()
print(f"cmpCMC程序耗时:{et1-st1:.4f}秒")

st1 = time.time()
# distmat
print('cmp distmat_hr4 + distmat_hr3 + distmat_hr2')
# cwd.computeCMC(distmatori * 2 + distmat_hr4  + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids,)
cwd.computeCMC(distmat_hr4 + distmat_hr3 + distmat_hr2 , q_pids, g_pids, q_camids, g_camids,)
et1 = time.time()
print(f"cmpCMC程序耗时:{et1-st1:.4f}秒")

st1 = time.time()
# distmat
print('cmp distmatori + distmat_hr4 + distmat_hr3 + distmat_hr2')
# cwd.computeCMC(distmatori * 2 + distmat_hr4  + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids,)
cwd.computeCMC(distmatori + distmat_hr4 + distmat_hr3 + distmat_hr2 , q_pids, g_pids, q_camids, g_camids,)
et1 = time.time()
print(f"cmpCMC程序耗时:{et1-st1:.4f}秒")

st1 = time.time()
# distmat
print('cmp distmatori * 2 + distmat_hr4 + distmat_hr3 + distmat_hr2')
# cwd.computeCMC(distmatori * 2 + distmat_hr4  + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids,)
cwd.computeCMC(distmatori * 2 + distmat_hr4 + distmat_hr3 + distmat_hr2 , q_pids, g_pids, q_camids, g_camids,)
et1 = time.time()
print(f"cmpCMC程序耗时:{et1-st1:.4f}秒")


elapsed_time = end_time - start_time
print(f"程序总耗时:{end_time - start_time:.4f}秒")
