from torch.utils.tensorboard import SummaryWriter
import numpy as np
import re
import os


path = '/mnt/data/code/SwinIR-main/log_unresize_mlr_cam_s64w8_x2'
dir_list = os.listdir(path)
for i in range(len(dir_list)):
    train_dir = path + '/' + dir_list[i]
    writer = SummaryWriter('/mnt/data/code/SwinIR-main_2/reidlog_unresize_mlr_rcmarket_s64w8_x2/'+dir_list[i])
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

    # f2 = open('log_s64w8_0/osnet_x1_0_s64w8_0_E_x2_market1501_softmax_cosinelr/reid.txt', 'w')
    # for l in reid:
    #     f2.writelines(l)
    # f2.close()
    # 将多余的pop丢弃
    for i in range(5):
        reid.pop()

    data = np.zeros((51, 6))
    for i in range(90, 39, -1):
        data[i - 40][0] = i
        for j in range(5, 0, -1):
            data[i - 40][j] = float(reid.pop().split(' ')[-1].split('%')[0] + ' ')
    # print(data)
    j = 0
    for i in data:
        writer.add_scalar('ReID/mAP/epoch', data[j][1], data[j][0])
        writer.add_scalar('ReID/Rank1/epoch', data[j][2], data[j][0])
        writer.add_scalar('ReID/Rank5/epoch', data[j][3], data[j][0])
        writer.add_scalar('ReID/Rank10/epoch', data[j][4], data[j][0])
        writer.add_scalar('ReID/Rank20/epoch', data[j][5], data[j][0])
        j += 1
    #
    writer.add_scalar('ReID/mAP/epoch', data[50][1], data[50][0])
    writer.add_scalar('ReID/Rank1/epoch', data[50][2], data[50][0])
    writer.add_scalar('ReID/Rank5/epoch', data[50][3], data[50][0])
    writer.add_scalar('ReID/Rank10/epoch', data[50][4], data[50][0])
    writer.add_scalar('ReID/Rank20/epoch', data[50][5], data[50][0])

    writer.add_scalar('ReID/mAP/epoch', data[50][1], data[50][0])
    writer.add_scalar('ReID/Rank1/epoch', data[50][2], data[50][0])
    writer.add_scalar('ReID/Rank5/epoch', data[50][3], data[50][0])
    writer.add_scalar('ReID/Rank10/epoch', data[50][4], data[50][0])
    writer.add_scalar('ReID/Rank20/epoch', data[50][5], data[50][0])

    writer.add_scalar('ReID/mAP/epoch', data[50][1], data[50][0])
    writer.add_scalar('ReID/Rank1/epoch', data[50][2], data[50][0])
    writer.add_scalar('ReID/Rank5/epoch', data[50][3], data[50][0])
    writer.add_scalar('ReID/Rank10/epoch', data[50][4], data[50][0])
    writer.add_scalar('ReID/Rank20/epoch', data[50][5], data[50][0])


    #
    # f2.writelines(str(i) + 'epoch\n')
    # f2.writelines(reid.pop().split(' ')[-1].split('%')[0] + ' ')
    # f2.writelines(reid.pop().split(' ')[-1].split('%')[0] + ' ')
    # f2.writelines(reid.pop().split(' ')[-1].split('%')[0] + ' ')
    # f2.writelines(reid.pop().split(' ')[-1].split('%')[0] + ' ')
    # f2.writelines(reid.pop().split(' ')[-1].split('%')[0] + ' ')
    # f2.writelines(reid.pop().split(' ')[-1].split('%')[0] + ' ')



# f2.writelines(l.split(' ')[-1].split('%')[0])

# data = np.zeros((32, 4))
# for i in range(int((len(results))):
#     if(results[i])
