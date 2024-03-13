from torch.utils.tensorboard import SummaryWriter
import numpy as np
import re
import os


path1 = '/mnt/data/code/SwinIR-main/log_mlr_rcduke_s64w8_x2'
path2 = '/mnt/data/code/SwinIR-main_2/log_mlr_rcduke_s64w8_x2'
writer = SummaryWriter('/mnt/data/code/SwinIR-main/maxreid')
n = 50
k = 0
data = np.zeros((n, 6))


def maxlog(path, k):
    dir_list = os.listdir(path)

    for i in range(len(dir_list)):
        train_dir = path + '/' + dir_list[i]

        train_log = os.listdir(train_dir)
        for line in train_log:
            log = re.findall(r"train.log", line)
            if log:
                train_log_dir = line
        f = open(train_dir + '/' + train_log_dir, 'r')
        results = f.readlines()
        map = []
        rank1, rank5, rank10, rank20 = [], [], [], []
        for line in results:
            mAP = re.findall(r"mAP:", line)
            Rank1 = re.findall(r"Rank-1 ", line)
            Rank5 = re.findall(r"Rank-5", line)
            Rank10 = re.findall(r"Rank-10 ", line)
            Rank20 = re.findall(r"Rank-20", line)
            if mAP:
                map.append(line)
            if Rank1:
                rank1.append(line)
            if Rank5:
                rank5.append(line)
            if Rank10:
                rank10.append(line)
            if Rank20:
                rank20.append(line)
        map = sorted(map, key=lambda  s: s.split(': ')[1].split('%')[0], reverse=True)
        rank1 = sorted(rank1, key=lambda  s: s.split(': ')[1].split('%')[0], reverse=True)
        rank5 = sorted(rank5, key=lambda s: s.split(': ')[1].split('%')[0], reverse=True)
        rank10 = sorted(rank10, key=lambda s: s.split(': ')[1].split('%')[0], reverse=True)
        rank20 = sorted(rank20, key=lambda s: s.split(': ')[1].split('%')[0], reverse=True)

        iter = dir_list[i].split('_')[7]
        maxmap = map[0].split(': ')[1].split('%')[0]
        maxrank1 = rank1[0].split(': ')[1].split('%')[0]
        maxrank5 = rank5[0].split(': ')[1].split('%')[0]
        maxrank10 = rank10[0].split(': ')[1].split('%')[0]
        maxrank20 = rank20[0].split(': ')[1].split('%')[0]


        data[k][0] = iter
        data[k][1] = maxmap
        data[k][2] = maxrank1
        data[k][3] = maxrank5
        data[k][4] = maxrank10
        data[k][5] = maxrank20
        k += 1
    return k

k = maxlog(path1, k)
k = maxlog(path2, k)


data = data[data[:,0].argsort()]
j = 0
for i in data:
    writer.add_scalar('ReID/mAP/iter', data[j][1], data[j][0])
    writer.add_scalar('ReID/Rank1/iter', data[j][2], data[j][0])
    writer.add_scalar('ReID/Rank5/iter', data[j][3], data[j][0])
    writer.add_scalar('ReID/Rank10/iter', data[j][4], data[j][0])
    writer.add_scalar('ReID/Rank20/iter', data[j][5], data[j][0])
    j += 1

l = n-1
writer.add_scalar('ReID/mAP/iter', data[l][1], data[l][0])
writer.add_scalar('ReID/Rank1/iter', data[l][2], data[l][0])
writer.add_scalar('ReID/Rank5/iter', data[l][3], data[l][0])
writer.add_scalar('ReID/Rank10/iter', data[l][4], data[l][0])
writer.add_scalar('ReID/Rank20/iter', data[l][5], data[l][0])

writer.add_scalar('ReID/mAP/iter', data[l][1], data[l][0])
writer.add_scalar('ReID/Rank1/iter', data[l][2], data[l][0])
writer.add_scalar('ReID/Rank5/iter', data[l][3], data[l][0])
writer.add_scalar('ReID/Rank10/iter', data[l][4], data[l][0])
writer.add_scalar('ReID/Rank20/iter', data[l][5], data[l][0])

writer.add_scalar('ReID/mAP/iter', data[l][1], data[l][0])
writer.add_scalar('ReID/Rank1/iter', data[l][2], data[l][0])
writer.add_scalar('ReID/Rank5/iter', data[l][3], data[l][0])
writer.add_scalar('ReID/Rank10/iter', data[l][4], data[l][0])
writer.add_scalar('ReID/Rank20/iter', data[l][5], data[l][0])

writer.add_scalar('ReID/mAP/iter', data[l][1], data[l][0])
writer.add_scalar('ReID/Rank1/iter', data[l][2], data[l][0])
writer.add_scalar('ReID/Rank5/iter', data[l][3], data[l][0])
writer.add_scalar('ReID/Rank10/iter', data[l][4], data[l][0])
writer.add_scalar('ReID/Rank20/iter', data[l][5], data[l][0])

writer.add_scalar('ReID/mAP/iter', data[l][1], data[l][0])
writer.add_scalar('ReID/Rank1/iter', data[l][2], data[l][0])
writer.add_scalar('ReID/Rank5/iter', data[l][3], data[l][0])
writer.add_scalar('ReID/Rank10/iter', data[l][4], data[l][0])
writer.add_scalar('ReID/Rank20/iter', data[l][5], data[l][0])


# for j in range(n-5, n):
#     writer.add_scalar('ReID/mAP/iter', data[j][1], data[j][0])
#     writer.add_scalar('ReID/Rank1/iter', data[j][2], data[j][0])
#     writer.add_scalar('ReID/Rank5/iter', data[j][3], data[j][0])
#     writer.add_scalar('ReID/Rank10/iter', data[j][4], data[j][0])
#     writer.add_scalar('ReID/Rank20/iter', data[j][5], data[j][0])



    # print(map[0].split(': ')[1].split('%')[0])



