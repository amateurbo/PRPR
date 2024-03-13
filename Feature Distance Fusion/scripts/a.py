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