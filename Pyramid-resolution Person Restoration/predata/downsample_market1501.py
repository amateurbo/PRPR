import cv2
import os.path

import numpy as np
from PIL import Image
import random
import torch
from torchvision import utils as vutils
path1 = '/mnt/data/mlr_datasets/market1501/bounding_box_train/'
dir_list = os.listdir(path1)
s = len(dir_list)
for i in range(len(dir_list)):
    imgpath = path1 + dir_list[i]
    ori_img = cv2.imread(imgpath)
    x2dir = imgpath.replace('market1501', 'market1501_x2')
    if not os.path.exists(x2dir):  # 判断路径是否存在，若不存在创建
        os.makedirs(x2dir)
    x3dir = imgpath.replace('market1501', 'market1501_x3')
    if not os.path.exists(x3dir):  # 判断路径是否存在，若不存在创建
        os.makedirs(x3dir)
    x4dir = imgpath.replace('market1501', 'market1501_x4')
    if not os.path.exists(x4dir):  # 判断路径是否存在，若不存在创建
        os.makedirs(x4dir)
    #双线性插值
    # x2降采样
    img2 = cv2.resize(ori_img, (int(ori_img.shape[1]/2),int(ori_img.shape[0]/2)), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(x2dir, img2)
    #x3降采样
    img3 = cv2.resize(ori_img, (int(ori_img.shape[1]/3),int(ori_img.shape[0]/3)), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(x3dir, img3)
    #x4降采样
    img4 = cv2.resize(ori_img, (int(ori_img.shape[1]/4),int(ori_img.shape[0]/4)), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(x4dir, img4)


path2 = '/mnt/data/mlr_datasets/market1501/query/'
dir_list = os.listdir(path2)
s = len(dir_list)
for i in range(len(dir_list)):
    imgpath = path2 + dir_list[i]
    ori_img = cv2.imread(imgpath)
    x2dir = imgpath.replace('market1501', 'market1501_x2')
    if not os.path.exists(x2dir):  # 判断路径是否存在，若不存在创建
        os.makedirs(x2dir)
    x3dir = imgpath.replace('market1501', 'market1501_x3')
    if not os.path.exists(x3dir):  # 判断路径是否存在，若不存在创建
        os.makedirs(x3dir)
    x4dir = imgpath.replace('market1501', 'market1501_x4')
    if not os.path.exists(x4dir):  # 判断路径是否存在，若不存在创建
        os.makedirs(x4dir)
    # 双线性插值
    # x2降采样
    img2 = cv2.resize(ori_img, (int(ori_img.shape[1] / 2), int(ori_img.shape[0] / 2)),
                      interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(x2dir, img2)
    # x3降采样
    img3 = cv2.resize(ori_img, (int(ori_img.shape[1] / 3), int(ori_img.shape[0] / 3)),
                      interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(x3dir, img3)
    # x4降采样
    img4 = cv2.resize(ori_img, (int(ori_img.shape[1] / 4), int(ori_img.shape[0] / 4)),
                      interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(x4dir, img4)

os.system('cp -r /mnt/data/mlr_datasets/market1501/bounding_box_test/ /mnt/data/mlr_datasets/market1501_x2')
os.system('cp -r /mnt/data/mlr_datasets/market1501/bounding_box_test/ /mnt/data/mlr_datasets/market1501_x3')
os.system('cp -r /mnt/data/mlr_datasets/market1501/bounding_box_test/ /mnt/data/mlr_datasets/market1501_x4')