import os
import cv2
import sys
import os.path as osp
import numpy as np
import time
import torch
import torchreid

import compute_weight_distance as cwd
from MLR_VIPeR import MLR_VIPeR
from torchreid import metrics
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

save_dir = '/mnt/data/code/reidlog/MLR_VIPeR/test'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True

distmat, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/VIPeR',
    model_load_weights='/mnt/data/code/reidlog/MLR_VIPeR/x0/2022-03-21-14-44-48model/model.pth.tar-40',
    dataset='MLR_VIPeR'
)

distmatx2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
    root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_MLR_VIPeR_x2_4400_E',
    model_load_weights='/mnt/data/code/reidlog/MLR_VIPeR/x2/osnet_x1_0_x2_4400_E_x2_MLR_VIPeR_softmax_cosinelr/2022-03-21-15-10-10model/model.pth.tar-41',
    dataset='MLR_VIPeR'
)

distmatx3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
    root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_MLR_VIPeR_x3_4000_E',
    model_load_weights='/mnt/data/code/reidlog/MLR_VIPeR/x3/osnet_x1_0_x3_4000_E_x3_MLR_VIPeR_softmax_cosinelr/2022-03-21-15-12-40model/model.pth.tar-46',
    dataset='MLR_VIPeR'
)

distmatx4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
    root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_MLR_VIPeR_x4_4600_E',
    model_load_weights='/mnt/data/code/reidlog/MLR_VIPeR/x4/osnet_x1_0_x4_4600_E_x4_MLR_VIPeR_softmax_cosinelr/2022-03-21-15-15-12model/model.pth.tar-46',
    dataset='MLR_VIPeR'
)

distmatx8, _, _, _, _, _, _, _, _, qfx8, gfx8 = cwd.main(
    root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_MLR_VIPeR_x8_4600_E',
    model_load_weights='/mnt/data/code/reidlog/MLR_VIPeR/x8/osnet_x1_0_x8_4600_E_x8_MLR_VIPeR_softmax_cosinelr/2022-03-21-15-17-44model/model.pth.tar-38',
    dataset='MLR_VIPeR'
)

#distmat0
print('dist0')
cwd.computeCMC(distmat, q_pids, g_pids, q_camids, g_camids,)
#distmatx2
print('distx2')
cwd.computeCMC(distmatx2, q_pids, g_pids, q_camids, g_camids)
#distmatx3
print('distx3')
cwd.computeCMC(distmatx3, q_pids, g_pids, q_camids, g_camids)
#distmatx4
print('distx4')
cwd.computeCMC(distmatx4, q_pids, g_pids, q_camids, g_camids)
#distmatx8
print('distx8')
cwd.computeCMC(distmatx8, q_pids, g_pids, q_camids, g_camids)

qfx234 = np.concatenate((qfx2, qfx3, qfx4), axis=1)
gfx234 = np.concatenate((gfx2, gfx3, gfx4), axis=1)

qf0x234 = np.concatenate((qf, qfx2, qfx3, qfx4), axis=1)
gf0x234 = np.concatenate((gf, gfx2, gfx3, gfx4), axis=1)

qf0x2348 = np.concatenate((qf, qfx2, qfx3, qfx4, qfx8), axis=1)
gf0x2348 = np.concatenate((gf, gfx2, gfx3, gfx4, gfx8), axis=1)

concat_feature_x234_distance = metrics.compute_distance_matrix(torch.from_numpy(qfx234), torch.from_numpy(gfx234), 'cosine')
concat_feature_x234_distance = concat_feature_x234_distance.numpy()
print('concat_feature_x234_distance_cosine')
cwd.computeCMC(concat_feature_x234_distance, q_pids, g_pids, q_camids, g_camids, )

concat_feature_0x234_distance = metrics.compute_distance_matrix(torch.from_numpy(qf0x234), torch.from_numpy(gf0x234), 'cosine')
concat_feature_0x234_distance = concat_feature_0x234_distance.numpy()
print('\nconcat_feature_0x234_distance_cosine')
cwd.computeCMC(concat_feature_0x234_distance, q_pids, g_pids, q_camids, g_camids, )

concat_feature_0x2348_distance = metrics.compute_distance_matrix(torch.from_numpy(qf0x2348), torch.from_numpy(gf0x2348), 'cosine')
concat_feature_0x2348_distance = concat_feature_0x2348_distance.numpy()
print('\nconcat_feature_0x2348_distance_cosine')
cwd.computeCMC(concat_feature_0x2348_distance, q_pids, g_pids, q_camids, g_camids, )



concat_feature_x234_distance = metrics.compute_distance_matrix(torch.from_numpy(qfx234), torch.from_numpy(gfx234), 'euclidean')
concat_feature_x234_distance = concat_feature_x234_distance.numpy()
print('concat_feature_x234_distance_euclidean')
cwd.computeCMC(concat_feature_x234_distance, q_pids, g_pids, q_camids, g_camids, )

concat_feature_0x234_distance = metrics.compute_distance_matrix(torch.from_numpy(qf0x234), torch.from_numpy(gf0x234), 'euclidean')
concat_feature_0x234_distance = concat_feature_0x234_distance.numpy()
print('\nconcat_feature_0x234_distance_euclidean')
cwd.computeCMC(concat_feature_0x234_distance, q_pids, g_pids, q_camids, g_camids, )

concat_feature_0x2348_distance = metrics.compute_distance_matrix(torch.from_numpy(qf0x2348), torch.from_numpy(gf0x2348), 'euclidean')
concat_feature_0x2348_distance = concat_feature_0x2348_distance.numpy()
print('\nconcat_feature_0x2348_distance_cosine')
cwd.computeCMC(concat_feature_0x2348_distance, q_pids, g_pids, q_camids, g_camids, )



#距离向量直接相加
print('\n直接加distx234')
cwd.computeCMC(distmatx2 + distmatx3 + distmatx4, q_pids, g_pids, q_camids, g_camids, )
print('\n直接加dist0 x234')
cwd.computeCMC(distmat + distmatx2 + distmatx3 + distmatx4, q_pids, g_pids, q_camids, g_camids, )
print('\n直接加dist0 x2348')
cwd.computeCMC(distmat + distmatx2 + distmatx3 + distmatx4 + distmatx8, q_pids, g_pids, q_camids, g_camids, )

gsize_ = np.array(gsize_)
qsize1_ = np.array(qsize1_)
qsize2_ = np.array(qsize2_)
qsize_ = np.array(qsize_)
average = np.average(gsize_)
bili = qsize_ / average
r = np.sqrt(qsize_ / average)
r_m1 = r - 1
r_m2 = r - 1 / 2
r_m3 = r - 1 / 3
r_m4 = r - 1 / 4
r_m8 = r - 1 / 8

#原始公式
dt = 0.2
print('计算权重，dt=' + str(dt))
w1 = np.exp(-pow(dt, 2) * pow(r_m1, 2))
w2 = np.exp(-pow(dt, 2) * pow(r_m2, 2))
w3 = np.exp(-pow(dt, 2) * pow(r_m3, 2))
w4 = np.exp(-pow(dt, 2) * pow(r_m4, 2))
w8 = np.exp(-pow(dt, 2) * pow(r_m8, 2))
print('\n\n原始方法距离向量加权......')
dist = distmat * w1.reshape(-1, 1)
distx2 = distmatx2 * w2.reshape(-1, 1)
distx3 = distmatx3 * w3.reshape(-1, 1)
distx4 = distmatx4 * w4.reshape(-1, 1)
distx8 = distmatx8 * w8.reshape(-1, 1)

print('加权distx234')
cwd.computeCMC(distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids, )
print('\n加权dist0 x234')
cwd.computeCMC(dist + distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids, )
print('\n加权dist0 x2348')
cwd.computeCMC(dist + distx2 + distx3 + distx4 + distx8, q_pids, g_pids, q_camids, g_camids, )
#
#只保留最接近的权重
r_m1 = np.maximum(r_m1, -r_m1)
r_m2 = np.maximum(r_m2, -r_m2)
r_m3 = np.maximum(r_m3, -r_m3)
r_m4 = np.maximum(r_m4, -r_m4)
r_m8 = np.maximum(r_m8, -r_m8)
all_r = np.vstack((r_m1, r_m2, r_m3, r_m4, r_m8))
min_r = np.min(all_r, axis=0).reshape(1, -1)
all_w = all_r * (all_r == min_r)
print('\n\n只保留最接近的权重加权......')
dist = distmat * all_w[0, :].reshape(-1, 1)
distx2 = distmatx2 * all_w[1, :].reshape(-1, 1)
distx3 = distmatx3 * all_w[2, :].reshape(-1, 1)
distx4 = distmatx4 * all_w[3, :].reshape(-1, 1)
distx8 = distmatx4 * all_w[4, :].reshape(-1, 1)
print('加权distx234')
cwd.computeCMC(distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids, )
print('\n加权dist0 x234')
cwd.computeCMC(dist + distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids, )
print('\n加权dist0 x2348')
cwd.computeCMC(dist + distx2 + distx3 + distx4 + distx8, q_pids, g_pids, q_camids, g_camids, )

#更改公式后加权
for i in range(1, 20):
    # --------更改公式
    dt = 0.1 * i
    print('\n\n更改公式后计算权重，dt=' + str(dt))
    r_m1 = np.maximum(r_m1, -r_m1)
    r_m2 = np.maximum(r_m2, -r_m2)
    r_m3 = np.maximum(r_m3, -r_m3)
    r_m4 = np.maximum(r_m4, -r_m4)
    r_m8 = np.maximum(r_m8, -r_m8)
    # w1 = np.exp(-pow(dt, 2) * r_m1)
    # w2 = np.exp(-pow(dt, 2) * r_m2)
    # w3 = np.exp(-pow(dt, 2) * r_m3)
    # w4 = np.exp(-pow(dt, 2) * r_m4)
    w1 = np.exp(-dt * r_m1)
    w2 = np.exp(-dt * r_m2)
    w3 = np.exp(-dt * r_m3)
    w4 = np.exp(-dt * r_m4)
    w8 = np.exp(-dt * r_m8)

    dist = distmat * w1.reshape(-1, 1)
    distx2 = distmatx2 * w2.reshape(-1, 1)
    distx3 = distmatx3 * w3.reshape(-1, 1)
    distx4 = distmatx4 * w4.reshape(-1, 1)
    distx8 = distmatx8 * w8.reshape(-1, 1)

    print('加权distx234')
    cwd.computeCMC(distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids, )
    print('\n加权dist0 x234')
    cwd.computeCMC(dist + distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids,)
    print('\n加权dist0 x2348')
    cwd.computeCMC(dist + distx2 + distx3 + distx4 + distx8, q_pids, g_pids, q_camids, g_camids,)


#更改公式后加权
for i in range(1, 20):
    # --------更改公式
    dt = 0.1 * i
    print('\n\n更改公式后计算权重，dt=' + str(dt))
    r_m1 = np.maximum(r_m1, -r_m1)
    r_m2 = np.maximum(r_m2, -r_m2)
    r_m3 = np.maximum(r_m3, -r_m3)
    r_m4 = np.maximum(r_m4, -r_m4)
    r_m8 = np.maximum(r_m8, -r_m8)
    w1 = np.exp(-pow(dt, 2) * r_m1)
    w2 = np.exp(-pow(dt, 2) * r_m2)
    w3 = np.exp(-pow(dt, 2) * r_m3)
    w4 = np.exp(-pow(dt, 2) * r_m4)
    w8 = np.exp(-pow(dt, 2) * r_m8)
    # w1 = np.exp(-dt * r_m1)
    # w2 = np.exp(-dt * r_m2)
    # w3 = np.exp(-dt * r_m3)
    # w4 = np.exp(-dt * r_m4)
    # w8 = np.exp(-dt * r_m8)

    dist = distmat * w1.reshape(-1, 1)
    distx2 = distmatx2 * w2.reshape(-1, 1)
    distx3 = distmatx3 * w3.reshape(-1, 1)
    distx4 = distmatx4 * w4.reshape(-1, 1)
    distx8 = distmatx8 * w8.reshape(-1, 1)

    print('加权distx234')
    cwd.computeCMC(distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids, )
    print('\n加权dist0 x234')
    cwd.computeCMC(dist + distx2 + distx3 + distx4, q_pids, g_pids, q_camids, g_camids,)
    print('\n加权dist0 x2348')
    cwd.computeCMC(dist + distx2 + distx3 + distx4 + distx8, q_pids, g_pids, q_camids, g_camids,)