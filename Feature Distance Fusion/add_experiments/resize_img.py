import os
import os.path as osp

import cv2

def resize(
        datapath='',
        savepath='',
        interpolation=cv2.INTER_CUBIC,
        size=(128, 256),
):
    print("default size is {}".format(size))
    os.makedirs(savepath, exist_ok=True)
    imgnames = os.listdir(datapath)
    for imgname in imgnames:
        path = osp.join(datapath, imgname)
        img = cv2.imread(path)
        output = cv2.resize(img, size, interpolation=interpolation)
        cv2.imwrite(osp.join(savepath, imgname), output)

if __name__ == '__main__':
    # resize(datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/save/save0',
    #        savepath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/save_cubic/save0',
    #        interpolation=cv2.INTER_CUBIC)
    path = '/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/save'
    dirs = os.listdir(path)
    for dir in dirs:
        resize(datapath=osp.join(path, dir),
               savepath=osp.join(path.replace('save', 'save_cubic'), dir),
               interpolation=cv2.INTER_CUBIC)
    for dir in dirs:
        resize(datapath=osp.join(path, dir),
               savepath=osp.join(path.replace('save', 'save_linear'), dir),
               interpolation=cv2.INTER_LINEAR)
    for dir in dirs:
        resize(datapath=osp.join(path, dir),
               savepath=osp.join(path.replace('save', 'save_nearest'), dir),
               interpolation=cv2.INTER_NEAREST)