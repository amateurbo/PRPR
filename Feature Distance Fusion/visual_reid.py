from torch.utils.tensorboard import SummaryWriter
import numpy as np
import re
import os

def main():
    path = 'log/log_caviar/'
    savepath = '/mnt/data/code/deep-person-reid-master/reid_caviar_test_log/'
    dir_list = os.listdir(path)
    for i in dir_list:
        viewlog(path+i, savepath+i)

def viewlog(train_dir, logdir):

    writer = SummaryWriter(logdir)
    train_log = os.listdir(train_dir)
    for line in train_log:
        log = re.findall(r"train.log", line)
        if log:
            train_log_dir = line
    f = open(train_dir + '/' + train_log_dir, 'r')
    results = f.readlines()
    reid = []
    for line in results:
        mAP = re.findall(r"mAP:", line)
        Rank1 = re.findall(r"Rank-1 ", line)
        Rank5 = re.findall(r"Rank-5", line)
        Rank10 = re.findall(r"Rank-10 ", line)
        Rank20 = re.findall(r"Rank-20", line)
        if mAP:
            reid.append(line)
        if Rank1:
            reid.append(line)
        if Rank5:
            reid.append(line)
        if Rank10:
            reid.append(line)
        if Rank20:
            reid.append(line)

    f2 = open(train_dir + '/reid.txt', 'w')
    for l in reid:
        f2.writelines(l)
    f2.close()
    # # 将多余的pop丢弃
    # for i in range(5):
    #     reid.pop()

    data = np.zeros((91, 6))
    for i in range(90, 0, -1):
        data[i][0] = i
        for j in range(5, 0, -1):
            data[i][j] = float(reid.pop().split(' ')[-1].split('%')[0] + ' ')

    #第二列倒序索引
    sdata2 =  (-data)[:,2].argsort()
    #第二列倒序
    data2 = data[sdata2]
    #可视化最大值
    writer.add_scalar('ReID/maxmAP/epoch', data2[0][1], data2[0][0])
    writer.add_scalar('ReID/maxRank1/epoch', data2[0][2], data2[0][0])
    writer.add_scalar('ReID/maxRank5/epoch', data2[0][3], data2[0][0])
    writer.add_scalar('ReID/maxRank10/epoch', data2[0][4], data2[0][0])
    writer.add_scalar('ReID/maxRank20/epoch', data2[0][5], data2[0][0])

    j = 0
    for i in data:
        writer.add_scalar('ReID/mAP/epoch', data[j][1], data[j][0])
        writer.add_scalar('ReID/Rank1/epoch', data[j][2], data[j][0])
        writer.add_scalar('ReID/Rank5/epoch', data[j][3], data[j][0])
        writer.add_scalar('ReID/Rank10/epoch', data[j][4], data[j][0])
        writer.add_scalar('ReID/Rank20/epoch', data[j][5], data[j][0])
        j += 1

    writer.add_scalar('ReID/mAP/epoch', data[90][1], data[90][0])
    writer.add_scalar('ReID/Rank1/epoch', data[90][2], data[90][0])
    writer.add_scalar('ReID/Rank5/epoch', data[90][3], data[90][0])
    writer.add_scalar('ReID/Rank10/epoch', data[90][4], data[90][0])
    writer.add_scalar('ReID/Rank20/epoch', data[90][5], data[90][0])

    writer.add_scalar('ReID/mAP/epoch', data[90][1], data[90][0])
    writer.add_scalar('ReID/Rank1/epoch', data[90][2], data[90][0])
    writer.add_scalar('ReID/Rank5/epoch', data[90][3], data[90][0])
    writer.add_scalar('ReID/Rank10/epoch', data[90][4], data[90][0])
    writer.add_scalar('ReID/Rank20/epoch', data[90][5], data[90][0])

    writer.add_scalar('ReID/mAP/epoch', data[90][1], data[90][0])
    writer.add_scalar('ReID/Rank1/epoch', data[90][2], data[90][0])
    writer.add_scalar('ReID/Rank5/epoch', data[90][3], data[90][0])
    writer.add_scalar('ReID/Rank10/epoch', data[90][4], data[90][0])
    writer.add_scalar('ReID/Rank20/epoch', data[90][5], data[90][0])

if __name__ == '__main__':
    main()