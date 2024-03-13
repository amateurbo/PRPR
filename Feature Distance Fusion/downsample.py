import cv2
import os.path
import random
import torch
from torchvision import utils as vutils
path = '/mnt/data/mlr_datasets/market1501/c1/ori_mlr_market1501/query'
dir_list = os.listdir(path)
s = len(dir_list)
for i in  range(len(dir_list)):
    imgpath = path + '/' + dir_list[i]
    ori_img = cv2.imread(imgpath)
    n = random.choice([2, 3, 4])
    width = int(ori_img.shape[1]/n)
    height = int(ori_img.shape[0]/n)
    dim = (width,height)
    img = cv2.resize(ori_img, dim)
    cv2.imwrite(imgpath, img)
    print('下采样'+str(n)+' 第'+str(i)+'张图片'+'success')#
# for i in range(len(dir_list)):
#     img_list = os.listdir(path+dir_list[i])
#     for j in range (len(img_list)):
#         imgpath = path + dir_list[i] + '/' + img_list[j]
#         ori_img = cv2.imread(imgpath)
#         n = random.choice([1, 1, 1, 1, 2, 3, 4])
#         width = int(ori_img.shape[1]/n)
#         height = int(ori_img.shape[0]/n)
#         dim = (width,height)
#         img = cv2.resize(ori_img, dim)
#         cv2.imwrite(imgpath, img)






        # y = cv2.imread(imgpath)
        # x = torch.from_numpy(y)
        # end = torch.nn.functional.interpolate(x, size=None, scale_factor=0.5, mode='nearest', align_corners=None)
        # vutils.save_image(img, imgpath, normalize=True)

    # x = Variable(torch.randn([1, 3, 64, 64]))
    # y0 = torch.interpolate(x, scale_factor=0.5)
    # y1 = torch.interpolate(x, size=[32, 32])
    #
    # y2 = torch.interpolate(x, size=[128, 128], mode="bilinear")

    # print(y1.shape)
    # print(y2.shape)
# def countFile(dir):
#     # 输入文件夹
#     tmp = 0
#     for item in os.listdir(dir):
#         if os.path.isfile(os.path.join(dir, item)):
#             tmp += 1
#         else:
#             tmp += countFile(os.path.join(dir, item))
#     return tmp
#
# filenum = countFile('/mnt/data/datasets/downtest/0002')    # 返回的是图片的张数
# print(filenum)
#
# # filenum
# n = 4
# index = 1   #保存图片编号
# num = 0     #处理图片计数
# for i in range(1, filenum + 1):
#     ########################################################
#     # 1.读取原始图片
#
#     if index < 10:
#         filename = "D:\\model\\super-resolution\\.div2k\images\\DIV2K_train_LR_bicubic\\X4\\000" + str(i) + ".png"
#     elif index < 100:
#         filename = "D:\\model\\super-resolution\\.div2k\images\\DIV2K_train_LR_bicubic\\X4\\00" + str(i) + ".png"
#     else:
#         filename = "D:\\model\\super-resolution\\.div2k\images\\DIV2K_train_LR_bicubic\\X4\\0" + str(i) + ".png"
#     print(filename)
#     original_image = cv2.imread(filename)
#     # 2.下采样
#     if n == 2:
#         img_1 = cv2.pyrDown(original_image)
#     if n == 4:
#         img_1 = cv2.pyrDown(original_image)
#         img_1 = cv2.pyrDown(img_1)
#     # 3.将下采样图片保存到指定路径当中
#     if index < 10:
#         cv2.imwrite("D:\\model\\super-resolution\\.div2k\\1\\000" + str(index) + ".png", img_1)
#     elif index < 100:
#         cv2.imwrite("D:\\model\\super-resolution\\.div2k\\1\\00" + str(index) + ".png", img_1)
#     else:
#         cv2.imwrite("D:\\model\\super-resolution\\.div2k\\1\\0" + str(index) + ".png", img_1)
#
#     num = num + 1
#     print("正在为第" + str(num) + "图片采样......")
#     index = index + 1
