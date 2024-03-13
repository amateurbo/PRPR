import os
import random
import shutil

import cv2
import numpy as np

#设置源数据路径及存储路径
testpath = '/mnt/data/mlr_datasets/market1501/bounding_box_test'
trainpath = '/mnt/data/mlr_datasets/market1501/bounding_box_train'
querypath = '/mnt/data/mlr_datasets/market1501/query'


querysavepath = '/mnt/data/mlr_datasets/market1501/c1/mlr_market1501/query'
testsavepath = '/mnt/data/mlr_datasets/market1501/c1/mlr_market1501/bounding_box_test'
trainsavepath = '/mnt/data/mlr_datasets/market1501/c1/mlr_market1501/bounding_box_train'
if not os.path.exists(querysavepath):
    os.makedirs(querysavepath)
if not os.path.exists(testsavepath):
    os.makedirs(testsavepath)
if not os.path.exists(trainsavepath):
    os.makedirs(trainsavepath)

imglist = os.listdir(testpath)
for i in range(len(imglist)):
    if(imglist[i] == 'Thumbs.db'):
        continue
    imgpath = testpath + '/' + imglist[i]
    camera = imglist[i].split('_')[1].split('s')[0]
    id = imglist[i].split('_')[0]
    #找出相机编号为C1的图片放入query,不为C1放入gallery
    if (int(id) > 0) & (camera == 'c1'):
        shutil.copy(imgpath, querysavepath)
    else:
        shutil.copy(imgpath, testsavepath)

imglist = os.listdir(querypath)
for i in range(len(imglist)):
    if(imglist[i] == 'Thumbs.db'):
        continue
    imgpath = querypath + '/' + imglist[i]
    camera = imglist[i].split('_')[1].split('s')[0]
    id = imglist[i].split('_')[0]
    #找出相机编号为C1的图片放入query,不为C1放入gallery
    if (int(id) > 0) & (camera == 'c1'):
        shutil.copy(imgpath, querysavepath)
    else:
        shutil.copy(imgpath, testsavepath)


#复制train
copytrain = 'cp -r ' + trainpath + ' ' + trainsavepath.split('bounding')[0]
os.system(copytrain)

#降采样query
imglist = os.listdir(querysavepath)
for i in range(len(imglist)):
    if(imglist[i] == 'Thumbs.db'):
        continue
    imgpath = querysavepath + '/' + imglist[i]
    n = random.choice([2, 3, 4])
    orimg = cv2.imread(imgpath)
    size = (int(orimg.shape[1]/n),int(orimg.shape[0]/n))
    downimg = cv2.resize(orimg, size, interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite(imgpath, downimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(imgpath, downimg)

#降采样train
imglist = os.listdir(trainsavepath)
for i in range(len(imglist)):
    if(imglist[i] == 'Thumbs.db'):
        continue
    imgpath = trainsavepath + '/' + imglist[i]
    n = random.choice([2, 3, 4])
    camera = imglist[i].split('_')[1].split('s')[0]
    id = imglist[i].split('_')[0]
    # 找出相机编号为C1的图片降采样
    if (int(id) > 0) & (camera == 'c1'):
        orimg = cv2.imread(imgpath)
        size = (int(orimg.shape[1] / n), int(orimg.shape[0] / n))
        downimg = cv2.resize(orimg, size, interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(imgpath, downimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(imgpath, downimg)