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

from caviar_new import CAVIAR
from torchreid import metrics
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

save_dir = '/mnt/data/code/reidlog/caviar_new_2/test'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True

distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/CAVIAR/CAVIAR',
    model_load_weights='/mnt/data/code/reidlog/CAVIAR/x3/osnet_x1_0_x3_5200_E_x3_CAVIAR_softmax_cosinelr/2022-03-29-09-57-57model/model.pth.tar-42',
    dataset='CAVIAR',
    # rerank=True
)


#------------sr_by_cuhk03----------------
# distmathr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
#     root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_cuhk03_sr_model/hr',
#     model_load_weights='/mnt/data/code/reidlog/caviar_new_2/hr/2022-05-06-15-12-03model/model-best.pth.tar',
#     dataset='CAVIAR',
#     # rerank=True
# )
#
# distmatx2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
#     root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_cuhk03_sr_model/x2',
#     model_load_weights='/mnt/data/code/reidlog/caviar_new_2/x2/2022-05-06-15-15-03model/model-best.pth.tar',
#     dataset='CAVIAR',
#     # rerank=True
# )
#
# distmatx3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
#     root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_cuhk03_sr_model/x3',
#     model_load_weights='/mnt/data/code/reidlog/caviar_new_2/x3/2022-05-06-15-18-00model/model-best.pth.tar',
#     dataset='CAVIAR',
#     # rerank=True
# )
#
# distmatx4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
#     root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_cuhk03_sr_model/x4',
#     model_load_weights='/mnt/data/code/reidlog/caviar_new_2/x4/2022-05-06-15-20-51model/model-best.pth.tar',
#     dataset='CAVIAR',
#     # rerank=True
# )
#
# distmatx8, _, _, _, _, _, _, _, _, qfx8, gfx8 = cwd.main(
#     root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_cuhk03_sr_model/x8',
#     model_load_weights='/mnt/data/code/reidlog/caviar_new_2/x8/2022-05-06-15-23-45model/model-best.pth.tar',
#     dataset='CAVIAR'
#     # rerank=True
# )



# #------------sr_by_dukemtmc----------------
# distmathr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
#     root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_dukemtmc_sr_model/hr',
#     model_load_weights='/mnt/data/code/reidlog/caviar_new_2/hr/model_by_duke/model-best.pth.tar',
#     dataset='CAVIAR',
#     # rerank=True
# )
#
# distmatx2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
#     root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_dukemtmc_sr_model/x2',
#     model_load_weights='/mnt/data/code/reidlog/caviar_new_2/x2/model_by_duke/model-best.pth.tar',
#     dataset='CAVIAR',
#     # rerank=True
# )
#
# distmatx3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
#     root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_dukemtmc_sr_model/x3',
#     model_load_weights='/mnt/data/code/reidlog/caviar_new_2/x3/model_by_duke/model-best.pth.tar',
#     dataset='CAVIAR',
#     # rerank=True
# )
#
# distmatx4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
#     root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_dukemtmc_sr_model/x4',
#     model_load_weights='/mnt/data/code/reidlog/caviar_new_2/x4/model_by_duke/model-best.pth.tar',
#     dataset='CAVIAR',
#     # rerank=True
# )
#
# distmatx8, _, _, _, _, _, _, _, _, qfx8, gfx8 = cwd.main(
#     root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_dukemtmc_sr_model/x8',
#     model_load_weights='/mnt/data/code/reidlog/caviar_new_2/x8/model_by_duke/model-best.pth.tar',
#     dataset='CAVIAR'
#     # rerank=True
# )

#------------sr_by_market----------------
distmathr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_market1501_sr_model_2/hr',
    model_load_weights='/mnt/data/code/reidlog/caviar_new_2/hr/model_by_market/model-best.pth.tar',
    dataset='CAVIAR',
    # rerank=True
)

distmatx2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
    root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_market1501_sr_model_2/x2',
    model_load_weights='/mnt/data/code/reidlog/caviar_new_2/x2/model_by_market/model-best.pth.tar',
    dataset='CAVIAR',
    # rerank=True
)

distmatx3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
    root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_market1501_sr_model_2/x3',
    model_load_weights='/mnt/data/code/reidlog/caviar_new_2/x3/model_by_market/model-best.pth.tar',
    dataset='CAVIAR',
    # rerank=True
)

distmatx4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
    root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_market1501_sr_model_2/x4',
    model_load_weights='/mnt/data/code/reidlog/caviar_new_2/x4/model_by_market/model-best.pth.tar',
    dataset='CAVIAR',
    # rerank=True
)

distmatx8, _, _, _, _, _, _, _, _, qfx8, gfx8 = cwd.main(
    root='/mnt/data/datasets/CAVIAR/caviar_new_sr_down_by_market1501_sr_model_2/x8',
    model_load_weights='/mnt/data/code/reidlog/caviar_new_2/x8/model_by_market/model-best.pth.tar',
    dataset='CAVIAR'
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