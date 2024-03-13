import os
import random
import shutil

import cv2
import numpy as np



#路径不存在创建文件夹
def checkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


#随机下采样【2，3，4】此路径下的所有图像
def mlr_down(imgdir, savedir):
    imglist = os.listdir(imgdir)
    for i in range(len(imglist)):
        if (imglist[i] == 'Thumbs.db'):
            continue
        checkdir(savedir)

        imgpath = imgdir + '/' + imglist[i]
        savepath = savedir + '/' + imglist[i]
        n = random.choice([2, 3, 4])
        input = cv2.imread(imgpath)
        size = (int(input.shape[1] / n), int(input.shape[0] / n))
        output = cv2.resize(input, size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(savepath, output)


#resize此路径下的所有图像
def resize(imgdir, savedir, size):
    imglist = os.listdir(imgdir)
    for i in range(len(imglist)):
        if (imglist[i] == 'Thumbs.db'):
            continue
        checkdir(savedir)

        imgpath = imgdir + '/' + imglist[i]
        savepath = savedir + '/' + imglist[i]
        input = cv2.imread(imgpath)
        output = cv2.resize(input, size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(savepath, output)

