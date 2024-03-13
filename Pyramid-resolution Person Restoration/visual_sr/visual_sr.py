from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter('/mnt/data/code/SwinIR-main/srlog_x3/')

f = open('result_s64w8.txt', 'r')
results = f.readlines()
data = np.zeros((33, 14))
for i in range(int((len(results)+1)/14)):
    id = i * 14
    iter = results[id].split('_')[0]
    id += 2
    train_PSNR = results[id].split(' ')[3]
    train_SSIM_RGB = results[id].split(' ')[5].split('\\')[0]
    id += 1
    train_PSNR_Y = results[id].split(' ')[3]
    train_SSIM_Y = results[id].split(' ')[5].split('\\')[0]
    id += 2
    query_PSNR = results[id].split(' ')[3]
    query_SSIM_RGB = results[id].split(' ')[5].split('\\')[0]
    id += 1
    query_PSNR_Y = results[id].split(' ')[3]
    query_SSIM_Y= results[id].split(' ')[5].split('\\')[0]
    id += 2
    mAP = results[id].split(' ')[1].split('%')[0]
    id += 1
    Rank1 = results[id].split(' ')[3].split('%')[0]
    id += 1
    Rank5 = results[id].split(' ')[3].split('%')[0]
    id += 1
    Rank10 = results[id].split(' ')[2].split('%')[0]
    id += 1
    Rank20 = results[id].split(' ')[2].split('%')[0]
    data[i][0] = iter
    data[i][1] = train_PSNR
    data[i][2] = train_SSIM_RGB
    data[i][3] = train_PSNR_Y
    data[i][4] = train_SSIM_Y
    data[i][5] = query_PSNR
    data[i][6] = query_SSIM_RGB
    data[i][7] = query_PSNR_Y
    data[i][8] = query_SSIM_Y
    data[i][9] = mAP
    data[i][10] = Rank1
    data[i][11] = Rank5
    data[i][12] = Rank10
    data[i][13] = Rank20

f.close()
writer.close()
a = data
data = a[np.lexsort(a[:, ::-1].T)] #排序
for s in data:
    # writer.add_scalars('ReID/iter', {'mAP': s[9], 'Rank1': s[10], 'Rank5':s[10], 'Rank10':s[10], 'Rank20':s[10] }, s[0])
    writer.add_scalars('SR/PNSR/iter',
                       {'train_PSNR': s[1], 'train_PSNR_Y': s[3], 'query_PSNR': s[5], 'query_PSNR_Y': s[7]}, s[0])
    writer.add_scalars('SR/SSIM/iter',
                       {'train_SSIM_RGB': s[2], 'train_SSIM_Y': s[4], 'query_SSIM_RGB': s[6], 'query_SSIM_Y': s[8]}, s[0])
    writer.add_scalar('ReID/mAP/iter', s[9], s[0])
    writer.add_scalar('ReID/Rank1/iter', s[10], s[0])
    writer.add_scalar('ReID/Rank5/iter', s[11], s[0])
    writer.add_scalar('ReID/Rank10/iter', s[12], s[0])
    writer.add_scalar('ReID/Rank20/iter', s[13], s[0])





# for n_iter in range(100):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)