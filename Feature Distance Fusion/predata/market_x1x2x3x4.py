import os
import random
import shutil

import cv2
import numpy as np
def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


testpath = '/mnt/data/market1501/bounding_box_test'
trainpath = '/mnt/data/market1501/bounding_box_train'
querypath = '/mnt/data/market1501/query'

savetrainpath = trainpath.replace('market1501', 'market1501/x1234_market1501')


cptrainx1 = 'cp -r  /mnt/data/market1501/bounding_box_train /mnt/data/market1501/x1234_market1501'
cptest = 'cp -r  /mnt/data/market1501/bounding_box_test /mnt/data/market1501/x1234_market1501'
cpquery = 'cp -r  /mnt/data/mlr_datasets/mlr_data/mlr_market1501/query /mnt/data/market1501/x1234_market1501'

os.system(cptrainx1)
os.system(cptest)
os.system(cpquery)

imgdir = trainpath
savedir = savetrainpath

imglist = os.listdir(imgdir)
for i in range(len(imglist)):
    if (imglist[i] == 'Thumbs.db'):
        continue
    checkdir(savedir)
    imgpath = imgdir + '/' + imglist[i]
    input = cv2.imread(imgpath)
    savepath_x2 = savedir + '/' + imglist[i].split('.')[0] + 'x2.jpg'
    output = cv2.resize(input, (32, 64), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(savepath_x2, output)

    savepath_x3 = savedir + '/' + imglist[i].split('.')[0] + 'x3.jpg'
    output = cv2.resize(input, (21, 42), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(savepath_x3, output)

    savepath_x4 = savedir + '/' + imglist[i].split('.')[0] + 'x4.jpg'
    output = cv2.resize(input, (16, 32), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(savepath_x4, output)

#python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root data_market/0_E_hr+sr_c1_mlr_market1501/  data.save_dir log_data_market/osnet_x1_0_hr+sr_0_E_mlr_market1501_softmax_cosinelr
#
# python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root data_market/sr_600E_mlrc1_market1501/  data.save_dir log_data_market/osnet_x1_0_sr_600E_mlrc1_market1501_softmax_cosinelr
#
# python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root data_market/sr_600E_noresize_mlrc1_market1501/  data.save_dir log_data_market/osnet_x1_0_sr_600E_noresize_mlrc1_market1501_softmax_cosinelr
#
# python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root data_market/sr+hr_600E_mlrc1_market1501/  data.save_dir log_data_market/osnet_x1_0_sr+hr_600E_mlrc1_market1501_softmax_cosinelr