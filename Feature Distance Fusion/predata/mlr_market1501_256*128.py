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

def main():
    # 设置源数据路径及存储路径
    testpath = '/mnt/data/market1501/bounding_box_test'
    trainpath = '/mnt/data/market1501/bounding_box_train'
    querypath = '/mnt/data/market1501/query'

    #准备sr微调数据集
    mlrtrainpath = trainpath.replace('market1501', 'market1501/256*128/mlr_market1501')
    mlrtestpath = testpath.replace('market1501', 'market1501/256*128/mlr_market1501')
    mlrquerypath = querypath.replace('market1501', 'market1501/256*128/mlr_market1501')
    mlr_down(trainpath, mlrtrainpath)
    mlr_down(querypath, mlrquerypath)
    copytest = 'cp -r ' + testpath + ' ' + mlrtestpath
    os.system(copytest)

    #mlr数据集resize生成LR
    trainpath64_128 = trainpath.replace('market1501', 'market1501/256*128/finetuningdata/LR')
    querypath64_128 = querypath.replace('market1501', 'market1501/256*128/finetuningdata/LR')
    resize(mlrquerypath, querypath64_128, (64, 128))
    resize(mlrtrainpath, trainpath64_128, (64, 128))


    #原始数据集resize生成HR
    trainpath128_256 = trainpath.replace('market1501', 'market1501/256*128/finetuningdata/HR')
    querypath128_256 = querypath.replace('market1501', 'market1501/256*128/finetuningdata/HR')
    resize(querypath, querypath128_256, (128, 256))
    resize(trainpath, trainpath128_256, (128, 256))


if __name__ == '__main__':
    main()

