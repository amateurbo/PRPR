import cv2
import os

import numpy as np
from PIL import Image
import random
import torch
from torchvision import utils as vutils
path = '/mnt/data/mlr_datasets/mlr_data/mlr_market1501/bounding_box_train'
dir_list = os.listdir(path)
save_dir = path.replace('mlr_market1501', 'mlr_market1501_x2')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for i in  range(len(dir_list)):
    imgpath = path + '/' + dir_list[i]
    ori_img = cv2.imread(imgpath)
    img2 = cv2.resize(ori_img, (32, 64), interpolation=cv2.INTER_LINEAR)
    save_path = imgpath.replace('mlr_market1501', 'mlr_market1501_x2')
    cv2.imwrite(save_path, img2)

path = '/mnt/data/mlr_datasets/mlr_data/mlr_market1501/query'
dir_list = os.listdir(path)
save_dir = path.replace('mlr_market1501', 'mlr_market1501_x2')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for i in  range(len(dir_list)):
    imgpath = path + '/' + dir_list[i]
    ori_img = cv2.imread(imgpath)
    img2 = cv2.resize(ori_img, (32, 64), interpolation=cv2.INTER_LINEAR)
    save_path = imgpath.replace('mlr_market1501', 'mlr_market1501_x2')
    cv2.imwrite(save_path, img2)

os.system('cp -r /mnt/data/mlr_datasets/mlr_data/mlr_market1501/bounding_box_test /mnt/data/mlr_datasets/mlr_data/mlr_market1501_x2/bounding_box_test')