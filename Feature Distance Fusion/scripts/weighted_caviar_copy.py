import os
import cv2
import sys
import os.path as osp
import numpy as np
import time
import torch
import torchreid

import compute_weight_distance as cwd
from CAVIAR import CAVIAR
from torchreid import metrics
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

save_dir = '/mnt/data/code/reidlog/CAVIAR/test'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True

#distmat0
distmat, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/CAVIAR',
    model_load_weights='/mnt/data/code/reidlog/CAVIAR/x3/osnet_x1_0_x3_5200_E_x3_CAVIAR_softmax_cosinelr/2022-03-29-09-57-57model/model.pth.tar-42',
    dataset='CAVIAR',
    # rerank=True
)

distmatx2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
    root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_CAVIAR_x2_4400_E',
    model_load_weights='/mnt/data/code/reidlog/CAVIAR/x2/osnet_x1_0_x2_4400_E_x2_CAVIAR_softmax_cosinelr/2022-03-29-10-05-16model/model.pth.tar-39',
    dataset='CAVIAR',
    # rerank=True
)

distmatx3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
    root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_CAVIAR_x3_5200_E',
    model_load_weights='/mnt/data/code/reidlog/CAVIAR/x3/osnet_x1_0_x3_5200_E_x3_CAVIAR_softmax_cosinelr/2022-03-29-10-24-03model/model.pth.tar-34',
    dataset='CAVIAR',
    # rerank=True
)

distmatx4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
    root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_CAVIAR_x4_3600_E',
    model_load_weights='/mnt/data/code/reidlog/CAVIAR/x4/osnet_x1_0_x4_3600_E_x4_CAVIAR_softmax_cosinelr/2022-03-29-10-08-56model/model.pth.tar-43',
    dataset='CAVIAR',
    # rerank=True
)

distmatx8, _, _, _, _, _, _, _, _, qfx8, gfx8 = cwd.main(
    root='/mnt/data/code/SwinIR-main/results/swinir_classical_sr_CAVIAR_x8_2800_E',
    model_load_weights='/mnt/data/code/reidlog/CAVIAR/x8/osnet_x1_0_x8_2800_E_x8_CAVIAR_softmax_cosinelr/2022-03-29-10-37-22model/model.pth.tar-32',
    dataset='CAVIAR'
    # rerank=True
)
# print('dist0')
# cwd.computeCMC(distmat, q_pids, g_pids, q_camids, g_camids,)
# print('\nApplying person re-ranking ...')
distmat_qq = metrics.compute_distance_matrix(qf, qf, 'cosine')
distmat_gg = metrics.compute_distance_matrix(gf, gf, 'cosine')
# distmat = cwd.re_ranking(distmat, distmat_qq, distmat_gg)
# cwd.computeCMC(distmat, q_pids, g_pids, q_camids, g_camids,)
# #distmatx2
# print('distx2')
# cwd.computeCMC(distmatx2, q_pids, g_pids, q_camids, g_camids)
# print('\nApplying person re-ranking ...')
distmat_qqx2 = metrics.compute_distance_matrix(qfx2, qfx2, 'cosine')
distmat_ggx2 = metrics.compute_distance_matrix(gfx2, gfx2, 'cosine')
# distmatx2 = cwd.re_ranking(distmatx2, distmat_qqx2, distmat_ggx2)
# cwd.computeCMC(distmatx2, q_pids, g_pids, q_camids, g_camids,)
# #distmatx3
# print('distx3')
# cwd.computeCMC(distmatx3, q_pids, g_pids, q_camids, g_camids)
# print('\nApplying person re-ranking ...')
distmat_qqx3 = metrics.compute_distance_matrix(qfx3, qfx3, 'cosine')
distmat_ggx3 = metrics.compute_distance_matrix(gfx3, gfx3, 'cosine')
# distmatx3 = cwd.re_ranking(distmatx3, distmat_qqx3, distmat_ggx3)
# cwd.computeCMC(distmatx3, q_pids, g_pids, q_camids, g_camids,)
# #distmatx4
# print('distx4')
# cwd.computeCMC(distmatx4, q_pids, g_pids, q_camids, g_camids)
# print('\nApplying person re-ranking ...')
distmat_qqx4 = metrics.compute_distance_matrix(qfx4, qfx4, 'cosine')
distmat_ggx4 = metrics.compute_distance_matrix(gfx4, gfx4, 'cosine')
# distmatx4 = cwd.re_ranking(distmatx4, distmat_qqx4, distmat_ggx4)
# cwd.computeCMC(distmatx4, q_pids, g_pids, q_camids, g_camids,)
# #distmatx8
# print('distx8')
# cwd.computeCMC(distmatx8, q_pids, g_pids, q_camids, g_camids)
# print('\nApplying person re-ranking ...')
distmat_qqx8 = metrics.compute_distance_matrix(qfx8, qfx8, 'cosine')
distmat_ggx8 = metrics.compute_distance_matrix(gfx8, gfx8, 'cosine')
# distmatx8 = cwd.re_ranking(distmatx8, distmat_qqx8, distmat_ggx8)
# cwd.computeCMC(distmatx8, q_pids, g_pids, q_camids, g_camids,)

#距离向量直接相加
print('\n直接加distx234')
cwd.computeCMC(distmatx2 + distmatx3 + distmatx4, q_pids, g_pids, q_camids, g_camids, )
print('\n直接加dist0 x234')
cwd.computeCMC(distmat + distmatx2 + distmatx3 + distmatx4, q_pids, g_pids, q_camids, g_camids, )
print('\n直接加distx2348')
cwd.computeCMC(distmatx2 + distmatx3 + distmatx4 + distmatx8, q_pids, g_pids, q_camids, g_camids, )
print('\n直接加dist0 x2348')
cwd.computeCMC(distmat + distmatx2 + distmatx3 + distmatx4 + distmatx8, q_pids, g_pids, q_camids, g_camids, )

print('\n\n\n\n\nApplying person re-ranking ...')

distmat_qq = metrics.compute_distance_matrix(qf, qf, 'cosine')
distmat_gg = metrics.compute_distance_matrix(gf, gf, 'cosine')
distmat_qqx2 = metrics.compute_distance_matrix(qfx2, qfx2, 'cosine')
distmat_ggx2 = metrics.compute_distance_matrix(gfx2, gfx2, 'cosine')
distmat_qqx3 = metrics.compute_distance_matrix(qfx3, qfx3, 'cosine')
distmat_ggx3 = metrics.compute_distance_matrix(gfx3, gfx3, 'cosine')
distmat_qqx4 = metrics.compute_distance_matrix(qfx4, qfx4, 'cosine')
distmat_ggx4 = metrics.compute_distance_matrix(gfx4, gfx4, 'cosine')
distmat_qqx8 = metrics.compute_distance_matrix(qfx8, qfx8, 'cosine')
distmat_ggx8 = metrics.compute_distance_matrix(gfx8, gfx8, 'cosine')
print('\n直接加dist0 单独 Applying person re-ranking ...')
distmat0 = cwd.re_ranking(distmat,
                              distmat_qq,
                              distmat_gg)
cwd.computeCMC(distmat0, q_pids, g_pids, q_camids, g_camids, )
print('\n直接加dist0 x2 单独 Applying person re-ranking ...')
distmat02 = cwd.re_ranking(distmat + distmatx2,
                              distmat_qq + distmat_qqx2,
                              distmat_gg + distmat_ggx2)
cwd.computeCMC(distmat02, q_pids, g_pids, q_camids, g_camids, )
print('\n直接加dist0 x23 单独 Applying person re-ranking ...')
distmat023 = cwd.re_ranking(distmat + distmatx2 + distmatx3,
                              distmat_qq + distmat_qqx2 + distmat_qqx3,
                              distmat_gg + distmat_ggx2 + distmat_ggx3)
cwd.computeCMC(distmat023, q_pids, g_pids, q_camids, g_camids, )
print('\n直接加dist x234 单独 Applying person re-ranking ...')
distmat234 = cwd.re_ranking(distmatx2 + distmatx3 + distmatx4,
                              distmat_qqx2 + distmat_qqx3 + distmat_qqx4,
                              distmat_ggx2 + distmat_ggx3 + distmat_ggx4)
cwd.computeCMC(distmat234, q_pids, g_pids, q_camids, g_camids, )
print('\n直接加dist0 x234 单独 Applying person re-ranking ...')
distmat0234 = cwd.re_ranking(distmat + distmatx2 + distmatx3 + distmatx4,
                              distmat_qq + distmat_qqx2 + distmat_qqx3 + distmat_qqx4,
                              distmat_gg + distmat_ggx2  + distmat_ggx3 + distmat_ggx4)
cwd.computeCMC(distmat0234, q_pids, g_pids, q_camids, g_camids, )
print('\n直接加dist x2348 单独 Applying person re-ranking ...')
distmat2348 = cwd.re_ranking(distmatx2 + distmatx3 + distmatx4 + distmatx8,
                              distmat_qqx2 + distmat_qqx3 + distmat_qqx4 + distmat_qqx8,
                              distmat_ggx2 + distmat_ggx3 + distmat_ggx4 + distmat_ggx8)
cwd.computeCMC(distmat2348, q_pids, g_pids, q_camids, g_camids, )

print('\n直接加dist0 x2348 单独 Applying person re-ranking ...')
distmat02348 = cwd.re_ranking(distmat + distmatx2 + distmatx3 + distmatx4 + distmatx8,
                              distmat_qq + distmat_qqx2 + distmat_qqx3 + distmat_qqx4 + distmat_qqx8,
                              distmat_gg + distmat_ggx2 + distmat_ggx3 + distmat_ggx4 + distmat_ggx8)
cwd.computeCMC(distmat02348, q_pids, g_pids, q_camids, g_camids, )

