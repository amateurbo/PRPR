import argparse
import shutil

import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import os.path as osp
import torch
import requests

from models.network_swinir_new import SwinIR as net
from utils import util_calculate_psnr_ssim as util

# def checkpath(path):
#     if not osp.exists(path):
#         os.makedirs(path)

def getmodelscale(dirlist):
    if dirlist=='hr':
        return 1
    elif dirlist=='x2':
        return 2
    elif dirlist=='x3':
        return 3
    elif dirlist=='x4':
        return 4
    elif dirlist=='x8':
        return 8

def getmodelpath(modelpaths, dirlist):

    if dirlist=='x2':
        modelpath = osp.join(modelpaths, 'x2.pth')
    elif dirlist=='x3':
        modelpath = osp.join(modelpaths, 'x3.pth')
    elif dirlist=='x4':
        modelpath = osp.join(modelpaths, 'x4.pth')
    elif dirlist=='x8':
        modelpath = osp.join(modelpaths, 'x8.pth')

    assert osp.exists(modelpath)
    return modelpath


def trans(output):
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    return output

def inference(model, scale, img_lq, device, window_size=8):
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                          (2, 0, 1))  # HCW-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    # inference
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = model(img_lq)
        # output = output[..., :h_old * scale, :w_old * scale]
        output2 = output[0][..., :h_old * 2, :w_old * 2]
        output3 = output[1][..., :h_old * 3, :w_old * 3]
        output4 = output[2][..., :h_old * 4, :w_old * 4]

        output2 = trans(output2)
        output3 = trans(output3)
        output4 = trans(output4)
    return output2, output3, output4
    # # save image
    # output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    # if output.ndim == 3:
    #     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    # output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

    # return output

def inv_norm(input):
    output = input
    # if output.ndim == 3:
    #     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

    return output

def sr_down(
        subdirs,
        datapath='/mnt/data/datasets/MTA_reid/MTA_reid_new',
        savepath='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr',
        modelpath='/mnt/data/code/superresolution/dukemtmc/swinir_sr_dukemtmc_x4/models/3000_G.pth',
        format='.png'
):

    # dirlists = ['x8', 'x4', 'x3', 'x2', 'hr']
    # dirlists = ['x8']
    # subdirs = ['query', 'test', 'train']
    # subdirs = ['test', 'train']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = 8


    for subdir in subdirs:
        folder = osp.join(datapath, subdir)
        savedir_x2 = osp.join(savepath, 'x2', subdir)
        savedir_x3 = osp.join(savepath, 'x3', subdir)
        savedir_x4 = osp.join(savepath, 'x4', subdir)
        os.makedirs(savedir_x2, exist_ok=True)
        os.makedirs(savedir_x3, exist_ok=True)
        os.makedirs(savedir_x4, exist_ok=True)

        # modelpath = modelpath
        model = define_model(model_path=modelpath, scale=4)
        model = model.cuda()
        model.eval()


        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            # read image
            print(path)
            imgname, img= get_image_pair(folder, path)  # image to HWC-BGR, float32
            img_hr = inv_norm(img)


            img_x2hr, img_x3hr, img_x4hr = inference(model=model, scale=4, img_lq=img, device=device, window_size=window_size)
            cv2.imwrite(f'{savedir_x2}/{imgname}{format}', img_x2hr)
            cv2.imwrite(f'{savedir_x3}/{imgname}{format}', img_x3hr)
            cv2.imwrite(f'{savedir_x4}/{imgname}{format}', img_x4hr)




def get_image_pair(folder, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    img_lq = cv2.imread(f'{folder}/{imgname}{imgext}', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.
    return imgname, img_lq

def define_model(model_path, scale, training_patch_size=64):
    model = net(upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key_g = 'params'

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    return model


def reid(datapath,
         pypath='/mnt/data/code/deep-person-reid-master/scripts/MTA_reid_new.py',
         savedir='/mnt/data/code/reidlog/MTA_reid_new/',
         config_path='/mnt/data/code/deep-person-reid-master/configs/MTA_reid.yaml',
         ):
    # ReID
    #reid command
    dirlists = ['x2', 'x3', 'x4']
    for dirlist in dirlists:
        path = osp.join(datapath, dirlist)
        root = ' data.root ' + path
        reid_task = f'python {pypath} --config-file {config_path} '
        reid_log = ' data.save_dir '+ osp.join(savedir, dirlist)
        reid_command = reid_task + root + reid_log
        os.system(reid_command)

def checkpath(path):
    if not osp.exists(path):
        print('{} not exitsts!'.format(path))
        print('Makedirsing.....')
        os.makedirs(path)
        print('Makedirs {} success!\n\n'.format(path))




if __name__ == '__main__':
    # sd.sr_down(
    #     datapath='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new',
    #     savepath='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_l1',
    #     modelpath='/mnt/data/code/superresolution/market1501_l1/swinir_sr_market1501_x4/models/5000_G.pth',
    #     subdirs=['query', 'bounding_box_test', 'bounding_box_train'],
    #     format='.jpg'
    # )
    #
    # # reid the complete sub datasets respectively
    # # datapath: path of complete sub_datasets
    # # pypath: path of reid.py
    # # savedir: log path
    # # config_path: reid config path
    # sd.reid(
    #     datapath='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_l1',
    #     pypath='/mnt/data/code/deep-person-reid-master/scripts/mlr_market1501_new.py',
    #     savedir='/mnt/data/code/reidlog/mlr_market1501_l1/',
    #     config_path='/mnt/data/code/deep-person-reid-master/configs/mlr_market1501.yaml',
    # )


    ###----------------------------------------my mlr_dataset------------------------------------
    sr_down(
        datapath='/mnt/data/datasets/CUHK03/MLR_CUHK03',
        savepath='/mnt/data/datasets/CUHK03/mlr_cuhk03_l1',
        modelpath='/mnt/data/code/superresolution/cuhk03_l1/swinir_sr_cuhk03_x4/models/5000_G.pth',
        subdirs=['query', 'bounding_box_test', 'bounding_box_train'],
        # subdirs=['bounding_box_train'],
        format='.png'
    )

    # reid(
    #     datapath='/mnt/data/datasets/CUHK03/mlr_cuhk03_l1',
    #     pypath='/mnt/data/code/deep-person-reid-master/scripts/mlr_cuhk03_new.py',
    #     savedir='/mnt/data/code/reidlog/mlr_cuhk03_l1/',
    #     config_path='/mnt/data/code/deep-person-reid-master/configs/MLR_CUHK03.yaml',
    # )