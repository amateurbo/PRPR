import cv2
import os.path

import numpy as np
from PIL import Image
import random
import torch
from torchvision import utils as vutils
path = '/mnt/data/sr_datasets/bigtest/train/HR'
dir_list = os.listdir(path)
s = len(dir_list)
for i in  range(len(dir_list)):
    imgpath = path + '/' + dir_list[i]
    ori_img = cv2.imread(imgpath)
    # #x2降采样 正方形
    # img2 = cv2.resize(ori_img, (int(ori_img.shape[1]/2),int(ori_img.shape[1]/2)), interpolation=cv2.INTER_CUBIC)
    # save_path2 = imgpath.replace('HR','LR_bicubic/X2').replace('.jpg','x2.jpg')
    # cv2.imwrite(save_path2, img2)
    # n = random.choice([2, 3, 4])
    # width = int(ori_img.shape[1]/2)
    # height = int(ori_img.shape[0]/2)
    # dim = (width,height)

    # x2降采样
    img2 = cv2.resize(ori_img, (int(ori_img.shape[1]/2),int(ori_img.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
    save_path2 = imgpath.replace('HR','LR_bicubic/X2').replace('.jpg','x2.jpg')
    cv2.imwrite(save_path2, img2)
    #x3降采样
    img3 = cv2.resize(ori_img, (int(ori_img.shape[1]/3),int(ori_img.shape[0]/3)), interpolation=cv2.INTER_CUBIC)
    save_path3 = imgpath.replace('HR','LR_bicubic/X3').replace('.jpg','x3.jpg')
    cv2.imwrite(save_path3, img3)
    #x4降采样
    img4 = cv2.resize(ori_img, (int(ori_img.shape[1]/4),int(ori_img.shape[0]/4)), interpolation=cv2.INTER_CUBIC)
    save_path4 = imgpath.replace('HR','LR_bicubic/X4').replace('.jpg','x4.jpg')
    cv2.imwrite(save_path4, img4)


    # #生成png
    # #原始图像保存
    #
    # # ori_img = np.rot90(ori_img)
    # save_path1 = imgpath.replace('self','png').replace('jpg','png')
    # cv2.imwrite(save_path1, ori_img)
    #
    # #x2降采样
    # img2 = cv2.resize(ori_img, (int(ori_img.shape[1]/2),int(ori_img.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
    # save_path2 = imgpath.replace('HR','LR_bicubic/X2').replace('.jpg','x2.png').replace('self','png')
    # cv2.imwrite(save_path2, img2)
    # #x3降采样
    # img3 = cv2.resize(ori_img, (int(ori_img.shape[1]/3),int(ori_img.shape[0]/3)), interpolation=cv2.INTER_CUBIC)
    # save_path3 = imgpath.replace('HR','LR_bicubic/X3').replace('.jpg','x3.png').replace('self','png')
    # cv2.imwrite(save_path3, img3)
    # #x4降采样
    # img4 = cv2.resize(ori_img, (int(ori_img.shape[1]/4),int(ori_img.shape[0]/4)), interpolation=cv2.INTER_CUBIC)
    # save_path4 = imgpath.replace('HR','LR_bicubic/X4').replace('.jpg','x4.png').replace('self','png')
    # cv2.imwrite(save_path4, img4)
