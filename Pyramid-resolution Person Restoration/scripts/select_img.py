import os
import shutil

pathx4 = '/mnt/data/datasets/MTA_reid/MTA_reid_l1/x4/train'
pathx3 = '/mnt/data/datasets/MTA_reid/MTA_reid_l1/x3/train'
pathx2 = '/mnt/data/datasets/MTA_reid/MTA_reid_l1/x2/train'
pathori = '/mnt/data/datasets/MTA_reid/train'

path = '/mnt/data/datasets/MTA_reid/fig4_select/x4'
imgnames = os.listdir(pathx4)

for imgname in imgnames:
    imgx4 = os.path.join(pathx4, imgname)
    shutil.copy(imgx4, imgx4.replace('MTA_reid_l1/x4/train', 'fig4_select/x4'))
    imgx3 = os.path.join(pathx3, imgname)
    shutil.copy(imgx3, imgx3.replace('MTA_reid_l1/x3/train', 'fig4_select/x3'))
    imgx2 = os.path.join(pathx2, imgname)
    shutil.copy(imgx2, imgx2.replace('MTA_reid_l1/x2/train', 'fig4_select/x2'))
    imgori = os.path.join(pathori, imgname)
    shutil.copy(imgori, imgori.replace('train', 'fig4_select/ori'))