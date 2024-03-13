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
from mydatasets import MTA_reid_new
torchreid.data.register_image_dataset('MTA_reid_new', MTA_reid_new)
from torchreid import metrics
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

save_dir = '/mnt/data/code/reidlog/MTA_reid_new2/test'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True

 # 对应尺度匹配
distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/MTA_reid',
    model_load_weights='/mnt/data/code/reidlog/MTA-reid/2022-02-24-20-05-16model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmathr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr_down_2/hr',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new_2/hr/2022-05-17-21-23-10model/model-best.pth.tar',
    dataset='MTA_reid_new',
    # rerank=True
)

distmatx2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr_down_2/x2',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new_2/x2/2022-05-18-00-27-26model/model-best.pth.tar',
    dataset='MTA_reid_new',
    # rerank=True
)

distmatx3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr_down_2/x3',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new_2/x3/2022-05-18-03-34-13model/model-best.pth.tar',
    dataset='MTA_reid_new',
    # rerank=True
)

distmatx4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr_down_2/x4',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new_2/x4/2022-05-18-06-43-22model/model-best.pth.tar',
    dataset='MTA_reid_new',
    # rerank=True
)

distmatx8, _, _, _, _, _, _, _, _, qfx8, gfx8 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr_down_2/x8',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new_2/x8/2022-05-18-09-53-37model/model-best.pth.tar',
    dataset='MTA_reid_new'
    # rerank=True
)

# # distmat
# print('dist ori')
# cwd.computeCMC(distmatori, q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')
# # distmathr
# print('dist hr')
# cwd.computeCMC(distmathr, q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')
# #distmatx2
# print('dist x2')
# cwd.computeCMC(distmatx2, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# #distmatx3
# print('dist x3')
# cwd.computeCMC(distmatx3, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# #distmatx4
# print('dist x4')
# cwd.computeCMC(distmatx4, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# #distmatx8
# print('dist x8')
# cwd.computeCMC(distmatx8, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
#
#
# print("直接加距离向量：")
# print('dist hr+x2')
# computeCMC(distmathr + distmatx2, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist hr+x2+x3')
# computeCMC(distmathr + distmatx2 + distmatx3, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist hr+x2+x3+x4')
# computeCMC(distmathr + distmatx2 + distmatx3 + distmatx4, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist hr+x2+x3+x4+x8')
# computeCMC(distmathr + distmatx2 + distmatx3 + distmatx4 +distmatx8, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
#
# print("直接加距离向量（考虑未划分前距离向量）：")
# print('dist ori+hr')
# computeCMC(distmatori + distmathr, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+hr+x2')
# computeCMC(distmatori + distmathr + distmatx2, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+hr+x2+x3')
# computeCMC(distmatori + distmathr + distmatx2 + distmatx3, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+hr+x2+x3+x4')
# computeCMC(distmatori + distmathr + distmatx2 + distmatx3 + distmatx4, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+hr+x2+x3+x4+x8')
# computeCMC(distmatori + distmathr + distmatx2 + distmatx3 + distmatx4 +distmatx8, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
#
#
# print("直接加距离向量（考虑未划分前距离向量）：")
# print('dist ori+hr')
# computeCMC(distmatori + distmathr, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+x2')
# computeCMC(distmatori + distmatx2, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+x3')
# computeCMC(distmatori + distmatx3, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+x4')
# computeCMC(distmatori + distmatx4, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')
# print('dist ori+x8')
# computeCMC(distmatori + distmatx8, q_pids, g_pids, q_camids, g_camids)
# print('\n\n')




# for i in range(10):
#     k2 = i * 0.1;
#     k1 = 1;
#     print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k1, k2, k2, k2, k2))
#     computeCMC(distmatori * k1 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k2,
#         q_pids, g_pids, q_camids, g_camids)
#
# for i in range(10):
#     k1 = i * 0.1;
#     k2 = 1;
#     print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k2, k2, k1))
#     computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k1,
#         q_pids, g_pids, q_camids, g_camids)
#
#     print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k2, k1, k2))
#     computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k1 + distmatx4 * k2,
#         q_pids, g_pids, q_camids, g_camids)
#
#     print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k2, k1, k2))
#     computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k1 + distmatx4 * k2,
#         q_pids, g_pids, q_camids, g_camids)
#
#     print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k1, k2, k2))
#     computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k1 + distmatx3 * k2 + distmatx4 * k2,
#         q_pids, g_pids, q_camids, g_camids)
#
#     print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k1, k2, k2, k2))
#     computeCMC(distmatori * k2 + distmathr * k1 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k2,
#         q_pids, g_pids, q_camids, g_camids)
#
#     print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k1, k2, k2, k2, k2))
#     computeCMC(distmatori * k1 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k2,
#         q_pids, g_pids, q_camids, g_camids)

#-----------ori----------
print('----------------------------------------------------')
print('---------------------   ori   ----------------------')
print('----------------------------------------------------')
for i in range(5):
    k1 = i * 0.2;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k1, k2, k2, k2, k2))
    computeCMC(distmatori * k1 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(4):
    k1 = 1 + i * 0.5;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k1, k2, k2, k2, k2))
    computeCMC(distmatori * k1 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(8):
    k1 = 3 + i * 1;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k1, k2, k2, k2, k2))
    computeCMC(distmatori * k1 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k2,
        q_pids, g_pids, q_camids, g_camids)


print('----------------------------------------------------')
print('----------------------   hr   ----------------------')
print('----------------------------------------------------')
for i in range(5):
    k1 = i * 0.2;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k1, k2, k2, k2))
    computeCMC(distmatori * k2 + distmathr * k1 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(4):
    k1 = 1 + i * 0.5;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k1, k2, k2, k2))
    computeCMC(distmatori * k2 + distmathr * k1 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(8):
    k1 = 3 + i * 1;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k1, k2, k2, k2))
    computeCMC(distmatori * k2 + distmathr * k1 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k2,
        q_pids, g_pids, q_camids, g_camids)


print('----------------------------------------------------')
print('----------------------   x2   ----------------------')
print('----------------------------------------------------')
for i in range(5):
    k1 = i * 0.2;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k1, k2, k2))
    computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k1 + distmatx3 * k2 + distmatx4 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(4):
    k1 = 1 + i * 0.5;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k1, k2, k2))
    computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k1 + distmatx3 * k2 + distmatx4 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(8):
    k1 = 3 + i * 1;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k1, k2, k2))
    computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k1 + distmatx3 * k2 + distmatx4 * k2,
        q_pids, g_pids, q_camids, g_camids)



print('----------------------------------------------------')
print('----------------------   x3   ----------------------')
print('----------------------------------------------------')
for i in range(5):
    k1 = i * 0.2;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k2, k1, k2))
    computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k1 + distmatx4 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(4):
    k1 = 1 + i * 0.5;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k2, k1, k2))
    computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k1 + distmatx4 * k2,
        q_pids, g_pids, q_camids, g_camids)
for i in range(8):
    k1 = 3 + i * 1;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k2, k1, k2))
    computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k1 + distmatx4 * k2,
        q_pids, g_pids, q_camids, g_camids)


print('----------------------------------------------------')
print('----------------------   x4   ----------------------')
print('----------------------------------------------------')
for i in range(5):
    k1 = i * 0.2;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k2, k2, k1))
    computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k1,
        q_pids, g_pids, q_camids, g_camids)
for i in range(4):
    k1 = 1 + i * 0.5;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k2, k2, k1))
    computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k1,
        q_pids, g_pids, q_camids, g_camids)
for i in range(8):
    k1 = 3 + i * 1;
    k2 = 1;
    print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{}'.format(k2, k2, k2, k2, k1))
    computeCMC(distmatori * k2 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k1,
        q_pids, g_pids, q_camids, g_camids)


