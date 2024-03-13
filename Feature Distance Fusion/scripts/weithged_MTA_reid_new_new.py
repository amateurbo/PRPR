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

save_dir = '/mnt/data/code/reidlog/MTA_reid_new/test'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True


# distmat_ori_ori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
#     root='/mnt/data/datasets/MTA_reid',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
#     dataset='MTA_reid_new',
#     # rerank=True
# )
#
# distmat_hr_hr, _, _, _, _, _, _, _, _, _, _ = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr/hr',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/hr/2022-04-19-23-46-23model/model.pth.tar-60',
#     dataset='MTA_reid_new',
#     # rerank=True
# )

# distmat_x2_ori,  q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf  = cwd.main(
#     root='/mnt/data/datasets/MTA_reid',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/x2/2022-04-20-05-55-17model/model.pth.tar-58',
#     dataset='MTA_reid_new',
#     # rerank=True
# )
#
# distmat_x2_hr, _, _, _, _, _, _, _, _,  _, _, = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr/hr',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/x2/2022-04-20-05-55-17model/model.pth.tar-58',
#     dataset='MTA_reid_new',
#     # rerank=True
# )
#
# distmat_x2_x2, _, _, _, _, _, _, _, _,  _, _,  = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr/x2',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/x2/2022-04-20-05-55-17model/model.pth.tar-58',
#     dataset='MTA_reid_new',
#     # rerank=True
# )
#
# distmat_x2_x3,  _, _, _, _, _, _, _, _,  _, _, = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr/x3',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/x2/2022-04-20-05-55-17model/model.pth.tar-58',
#     dataset='MTA_reid_new',
#     # rerank=True
# )
#
# distmat_x2_x4, _, _, _, _, _, _, _, _,  _, _, = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr/x4',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/x2/2022-04-20-05-55-17model/model.pth.tar-58',
#     dataset='MTA_reid_new',
#     # rerank=True
# )
# print('distmat_x2_ori+x2')
# cwd.computeCMC(distmat_x2_ori + distmat_x2_x2, q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')
# print('distmat_x2_hr+x2')
# cwd.computeCMC(distmat_x2_hr + distmat_x2_x2, q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')
#
# print('distmat_x2_ori+hr+x2')
# cwd.computeCMC(distmat_x2_ori + distmat_x2_hr + distmat_x2_x2, q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')
# print('distmat_x2_ori+hr+x2+x3')
# cwd.computeCMC(distmat_x2_ori + distmat_x2_hr + distmat_x2_x2 + distmat_x2_x3, q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')
# print('distmat_x2_ori+hr+x2+x3+x4')
# cwd.computeCMC(distmat_x2_ori + distmat_x2_hr + distmat_x2_x2 + distmat_x2_x3 + distmat_x2_x4, q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')
# print('distmat_x2_hr+x2+x3+x4')
# cwd.computeCMC(distmat_x2_hr + distmat_x2_x2 + distmat_x2_x3 + distmat_x2_x4, q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')
# print('distmat_x2_hr+x2+x3')
# cwd.computeCMC(distmat_x2_hr + distmat_x2_x2 + distmat_x2_x3 , q_pids, g_pids, q_camids, g_camids,)
# print('\n\n')



# distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/ori_ori',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
#     dataset='MTA_reid_new',
#     # rerank=True
# )

# distmathr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr/hr',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
#     dataset='MTA_reid_new',
#     # rerank=True
# )
#
# distmatx2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr/x2',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
#     dataset='MTA_reid_new',
#     # rerank=True
# )
#
# distmatx3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr/x3',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
#     dataset='MTA_reid_new',
#     # rerank=True
# )
#
# distmatx4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr/x4',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
#     dataset='MTA_reid_new',
#     # rerank=True
# )
#
# distmatx8, _, _, _, _, _, _, _, _, qfx8, gfx8 = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr/x8',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
#     dataset='MTA_reid_new'
#     # rerank=True
# )
# distmat
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

#
#
#
# for i in range(20):
#     k1 = i*0.01
#     k2 = (1-k1)/5
#     print('dist ori*{} + hr*{} + x2*{} + x3*{} + x4*{} + x8*{}'.format(k2, k2, k2, k2, k2, k1))
#     computeCMC(
#         distmatori * k2 + distmathr * k2 + distmatx2 * k2 + distmatx3 * k2 + distmatx4 * k2 + distmatx8 * k1,
#         q_pids, g_pids, q_camids, g_camids)
#     print('\n\n')
#
# for i in range(10):
#     k = i*0.1
#     print('dist ori + hr + x2 + x3 + x4 + x8*{}'.format(k))
#     computeCMC(distmatori + distmathr + distmatx2 + distmatx3 + distmatx4 +distmatx8*k, q_pids, g_pids, q_camids, g_camids)
#     print('\n\n')
# start = time.time()
# distmat_combine, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
#     root='/mnt/data/datasets/MTA_reid/MTA_reid_combine_sr',
#     model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
#     dataset='MTA_reid_new',
#     # rerank=True
# )
# computeCMC(distmat_combine, q_pids, g_pids, q_camids, g_camids)
# end = time.time()
# print('time:', end-start)

distmat_ori_ori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/ori_ori',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_ori_hr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/ori_hr',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_ori_x2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/ori_x2',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_ori_x3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/ori_x3',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_ori_x4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/ori_x4',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)



distmat_hr_ori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/hr_ori',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_hr_hr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/hr_hr',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_hr_x2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/hr_x2',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_hr_x3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/hr_x3',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_hr_x4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/hr_x4',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)







distmat_x2_ori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x2_ori',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_x2_hr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x2_hr',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_x2_x2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x2_x2',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_x2_x3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x2_x3',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_x2_x4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x2_x4',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)




distmat_x3_ori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x3_ori',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_x3_hr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x3_hr',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_x3_x2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x3_x2',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_x3_x3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x3_x3',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_x3_x4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x3_x4',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)










distmat_x4_ori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x4_ori',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_x4_hr, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x4_hr',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_x4_x2, _, _, _, _, _, _, _, _, qfx2, gfx2 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x4_x2',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_x4_x3, _, _, _, _, _, _, _, _, qfx3, gfx3 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x4_x3',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)

distmat_x4_x4, _, _, _, _, _, _, _, _, qfx4, gfx4 = cwd.main(
    root='/mnt/data/datasets/MTA_reid/MTA_reid_crossmatch/x4_x4',
    model_load_weights='/mnt/data/code/reidlog/MTA_reid_new/combine_hr_x234/2022-04-26-22-26-59model/model.pth.tar-59',
    dataset='MTA_reid_new',
    # rerank=True
)




print('distmat_ori_ori')
cwd.computeCMC(distmat_ori_ori, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_ori_hr')
cwd.computeCMC(distmat_ori_hr, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_ori_x2')
cwd.computeCMC(distmat_ori_x2, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_ori_x3')
cwd.computeCMC(distmat_ori_x3, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_ori_x4')
cwd.computeCMC(distmat_ori_x4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_ori_ori+hr+x2+x3+x4')
cwd.computeCMC(distmat_ori_ori + distmat_ori_hr + distmat_ori_x2 + distmat_ori_x3 + distmat_ori_x4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')






print('distmat_hr_ori')
cwd.computeCMC(distmat_hr_ori, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_hr_hr')
cwd.computeCMC(distmat_hr_hr, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_hr_x2')
cwd.computeCMC(distmat_hr_x2, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_hr_x3')
cwd.computeCMC(distmat_hr_x3, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_hr_x4')
cwd.computeCMC(distmat_hr_x4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_hr_ori+hr+x2+x3+x4')
cwd.computeCMC(distmat_hr_ori + distmat_hr_hr + distmat_hr_x2 + distmat_hr_x3 + distmat_hr_x4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')





print('distmat_x2_ori')
cwd.computeCMC(distmat_x2_ori, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x2_hr')
cwd.computeCMC(distmat_x2_hr, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x2_x2')
cwd.computeCMC(distmat_x2_x2, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x2_x3')
cwd.computeCMC(distmat_x2_x3, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x2_x4')
cwd.computeCMC(distmat_x2_x4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x2_ori+hr+x2+x3+x4')
cwd.computeCMC(distmat_x2_ori + distmat_x2_hr + distmat_x2_x2 + distmat_x2_x3 + distmat_x2_x4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')





print('distmat_x3_ori')
cwd.computeCMC(distmat_x3_ori, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x3_hr')
cwd.computeCMC(distmat_x3_hr, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x3_x2')
cwd.computeCMC(distmat_x3_x2, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x3_x3')
cwd.computeCMC(distmat_x3_x3, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x3_x4')
cwd.computeCMC(distmat_x3_x4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x3_ori+hr+x2+x3+x4')
cwd.computeCMC(distmat_x3_ori + distmat_x3_hr + distmat_x3_x2 + distmat_x3_x3 + distmat_x3_x4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')




print('distmat_x4_ori')
cwd.computeCMC(distmat_x4_ori, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x4_hr')
cwd.computeCMC(distmat_x4_hr, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x4_x2')
cwd.computeCMC(distmat_x4_x2, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x4_x3')
cwd.computeCMC(distmat_x4_x3, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x4_x4')
cwd.computeCMC(distmat_x4_x4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_x4_ori+hr+x2+x3+x4')
cwd.computeCMC(distmat_x4_ori + distmat_x4_hr + distmat_x4_x2 + distmat_x4_x3 + distmat_x4_x4, q_pids, g_pids, q_camids, g_camids,)
print('\n\n')



print('distmat_all_ori+hr')
cwd.computeCMC( distmat_ori_ori + distmat_ori_hr + distmat_ori_x2 + distmat_ori_x3 + distmat_ori_x4 +
                distmat_hr_ori + distmat_hr_hr + distmat_hr_x2 + distmat_hr_x3 + distmat_hr_x4
               , q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_all_ori+hr+x2')
cwd.computeCMC( distmat_ori_ori + distmat_ori_hr + distmat_ori_x2 + distmat_ori_x3 + distmat_ori_x4 +
                distmat_hr_ori + distmat_hr_hr + distmat_hr_x2 + distmat_hr_x3 + distmat_hr_x4 +
                distmat_x2_ori + distmat_x2_hr + distmat_x2_x2 + distmat_x2_x3 + distmat_x2_x4
               , q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_all_ori+hr+x2+x3')
cwd.computeCMC( distmat_ori_ori + distmat_ori_hr + distmat_ori_x2 + distmat_ori_x3 + distmat_ori_x4 +
                distmat_hr_ori + distmat_hr_hr + distmat_hr_x2 + distmat_hr_x3 + distmat_hr_x4 +
                distmat_x2_ori + distmat_x2_hr + distmat_x2_x2 + distmat_x2_x3 + distmat_x2_x4 +
                distmat_x3_ori + distmat_x3_hr + distmat_x3_x2 + distmat_x3_x3 + distmat_x3_x4
               , q_pids, g_pids, q_camids, g_camids,)
print('\n\n')
print('distmat_all_ori+hr+x2+x3+x4')
cwd.computeCMC( distmat_ori_ori + distmat_ori_hr + distmat_ori_x2 + distmat_ori_x3 + distmat_ori_x4 +
                distmat_hr_ori + distmat_hr_hr + distmat_hr_x2 + distmat_hr_x3 + distmat_hr_x4 +
                distmat_x2_ori + distmat_x2_hr + distmat_x2_x2 + distmat_x2_x3 + distmat_x2_x4 +
                distmat_x3_ori + distmat_x3_hr + distmat_x3_x2 + distmat_x3_x3 + distmat_x3_x4 +
                distmat_x4_ori + distmat_x4_hr + distmat_x4_x2 + distmat_x4_x3 + distmat_x4_x4
               , q_pids, g_pids, q_camids, g_camids,)
print('\n\n')






