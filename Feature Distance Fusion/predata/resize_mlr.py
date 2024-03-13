import os
import random
import shutil

import cv2
import numpy as np

#设置源数据路径及存储路径
testpath = '/mnt/data/mlr_datasets/market1501/c1/mlrc1_market1501/bounding_box_test'
trainpath = '/mnt/data/mlr_datasets/market1501/c1/mlrc1_market1501/bounding_box_train'
querypath = '/mnt/data/mlr_datasets/market1501/c1/mlrc1_market1501/query'


querysavepath = '/mnt/data/mlr_datasets/market1501/c1/resize_mlrc1_market1501/query'
testsavepath = '/mnt/data/mlr_datasets/market1501/c1/resize_mlrc1_market1501/bounding_box_test'
trainsavepath = '/mnt/data/mlr_datasets/market1501/c1/resize_mlrc1_market1501/bounding_box_train'
if not os.path.exists(querysavepath):
    os.makedirs(querysavepath)
if not os.path.exists(testsavepath):
    os.makedirs(testsavepath)
if not os.path.exists(trainsavepath):
    os.makedirs(trainsavepath)

imglist = os.listdir(trainpath)
for i in range(len(imglist)):
    if(imglist[i] == 'Thumbs.db'):
        continue
    imgpath = trainpath + '/' + imglist[i]
    orimg = cv2.imread(imgpath)
    size = (32, 64)
    downimg = cv2.resize(orimg, size, interpolation=cv2.INTER_LINEAR)
    savepath = trainsavepath + '/' + imglist[i]
    # cv2.imwrite(imgpath, downimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(savepath, downimg)

imglist = os.listdir(querypath)
for i in range(len(imglist)):
    if(imglist[i] == 'Thumbs.db'):
        continue
    imgpath = querypath + '/' + imglist[i]
    orimg = cv2.imread(imgpath)
    size = (32, 64)
    downimg = cv2.resize(orimg, size, interpolation=cv2.INTER_LINEAR)
    savepath = querysavepath + '/' + imglist[i]
    # cv2.imwrite(imgpath, downimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(savepath, downimg)

# python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root data/mlr_market1501/  data.save_dir log_mlr/osnet_x1_0_ori_mlr_market1501_default_softmax_cosinelr
