import shutil

import cv2
import os.path

import numpy as np
from PIL import Image
import random
import torch
from torchvision import utils as vutils
path = '/mnt/data/mlr_datasets/mlr_data/mlr_market1501_x2/bounding_box_train'
dir_list = os.listdir(path)
s = len(dir_list)
for i in  range(len(dir_list)):
    if(dir_list[i] == 'Thumbs.db'):
        continue
    imgdirpath = path + '/' + dir_list[i]
    p1 = imgdirpath.split('/')[7].split('_')[0]
    # 前面676个身份作为训练 后75作为测试 0、1干扰项
    if(int(p1) > 0):
        # if (int(p1) < 1365): #pre_test
        if (int(p1) < 1332):
            traindir = path.replace('bounding_box_train','finetuning_data/train/LR')
            if not os.path.exists(traindir): #判断路径是否存在，若不存在创建
                os.makedirs(traindir)
            shutil.copy(imgdirpath,traindir+'/'+dir_list[i]) #复制图片
        else:
            traindir = path.replace('bounding_box_train','finetuning_data/test/LR')
            if not os.path.exists(traindir): #判断路径是否存在，若不存在创建
                os.makedirs(traindir)
            shutil.copy(imgdirpath,traindir+'/'+dir_list[i]) #复制图片

path = '/mnt/data/mlr_datasets/mlr_data/market1501/bounding_box_train'
dir_list = os.listdir(path)
s = len(dir_list)
for i in range(len(dir_list)):
    if (dir_list[i] == 'Thumbs.db'):
        continue
    imgdirpath = path + '/' + dir_list[i]
    p1 = imgdirpath.split('/')[7].split('_')[0]
    # 前面676个身份作为训练 后75作为测试 0、1干扰项
    if (int(p1) > 0):
        # if (int(p1) < 1365): #pre_test
        if (int(p1) < 1332):
            traindir = path.replace('bounding_box_train', 'finetuning_data/train/HR').replace('market1501','mlr_market1501_x2')
            if not os.path.exists(traindir):  # 判断路径是否存在，若不存在创建
                os.makedirs(traindir)
            shutil.copy(imgdirpath, traindir + '/' + dir_list[i])  # 复制图片
        else:
            traindir = path.replace('bounding_box_train', 'finetuning_data/test/HR').replace('market1501','mlr_market1501_x2')
            if not os.path.exists(traindir):  # 判断路径是否存在，若不存在创建
                os.makedirs(traindir)
            shutil.copy(imgdirpath, traindir + '/' + dir_list[i])  # 复制图片

    # img_list = os.listdir(imgdirpath)
    # for j in range(len(img_list)):
    #     imgpath = imgdirpath + '/' + img_list[j]
    #     img = cv2.imread(imgpath)
    #     if(img.shape[0]>500 & img.shape[1]>200):
    #         savepath = '/mnt/data/sr_datasets/premsmt/test/HR/' + img_list[j]
           