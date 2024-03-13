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

from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util

# def checkpath(path):
#     if not osp.exists(path):
#         os.makedirs(path)


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
        output = output[..., :h_old * scale, :w_old * scale]

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

    return output

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
    dirlists = ['hr', 'lr1', 'lr2', 'lr3']
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


def sr_datasets(
        subdirs,
        datapath='/mnt/data/datasets/MTA_reid/MTA_reid_new',
        savepath='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr',
        modelpaths='/mnt/data/code/SwinIR-main/model_zoo/MTA_reid_new',
        format='.png'
):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = 8

    # 用x8超分
    modelpath = getmodelpath(modelpaths, 'x8')
    model_hr = define_model(model_path=modelpath, scale=8)
    model_hr.eval()
    model_hr = model_hr.to(device)
    # 用x4超分
    modelpath = getmodelpath(modelpaths, 'x4')
    model_lr1 = define_model(model_path=modelpath, scale=4)
    model_lr1.eval()
    model_lr1 = model_lr1.to(device)
    # 用x3超分
    modelpath = getmodelpath(modelpaths, 'x3')
    model_lr2 = define_model(model_path=modelpath, scale=3)
    model_lr2.eval()
    model_lr2 = model_lr2.to(device)
    # 用x2超分
    modelpath = getmodelpath(modelpaths, 'x2')
    model_lr3 = define_model(model_path=modelpath, scale=2)
    model_lr3.eval()
    model_lr3 = model_lr3.to(device)

    for subdir in subdirs:
        folder = osp.join(datapath, subdir)
        savedir_hr = osp.join(savepath, 'hr', subdir)       #用x8超分
        savedir_lr1 = osp.join(savepath, 'lr1', subdir)     #用x4超分
        savedir_lr2 = osp.join(savepath, 'lr2', subdir)     #用x3超分
        savedir_lr3 = osp.join(savepath, 'lr3', subdir)     #用x2超分
        os.makedirs(savedir_hr, exist_ok=True)
        os.makedirs(savedir_lr1, exist_ok=True)
        os.makedirs(savedir_lr2, exist_ok=True)
        os.makedirs(savedir_lr3, exist_ok=True)

        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            # read image
            print(path)
            imgname, img = get_image_pair(folder, path)  # image to HWC-BGR, float32
            h, w = img.shape[:-1]

            img_hr = inference(model=model_hr, scale=8, img_lq=img, device=device, window_size=window_size)
            img_lr1 = inference(model=model_lr1, scale=4, img_lq=img, device=device, window_size=window_size)
            img_lr2 = inference(model=model_lr2, scale=3, img_lq=img, device=device, window_size=window_size)
            img_lr3 = inference(model=model_lr3, scale=2, img_lq=img, device=device, window_size=window_size)

            cv2.imwrite(f'{savedir_hr}/{imgname}{format}', img_hr)
            cv2.imwrite(f'{savedir_lr1}/{imgname}{format}', img_lr1)
            cv2.imwrite(f'{savedir_lr2}/{imgname}{format}', img_lr2)
            cv2.imwrite(f'{savedir_lr3}/{imgname}{format}', img_lr3)


def sr_reid_mlr_market1501():
    sr_datasets(
        datapath='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501',
        savepath='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_SR',
        modelpaths='/mnt/data/code/SwinIR-main/model_zoo/mlr_mark1501_new',
        subdirs=['query', 'bounding_box_test', 'bounding_box_train'],
        format='.jpg'
    )

    # reid the complete sub datasets respectively
    # datapath: path of complete sub_datasets
    # pypath: path of reid.py
    # savedir: log path
    # config_path: reid config path
    reid(
        datapath='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_SR',
        pypath='/mnt/data/code/deep-person-reid-master/scripts/mlr_market1501_new.py',
        savedir='/mnt/data/code/reidlog/mlr_market1501_SR/',
        config_path='/mnt/data/code/deep-person-reid-master/configs/mlr_market1501.yaml',
    )

def downsample(
        datapath,
        savepath,
        subdirs,
        format,
):

    for subdir in subdirs:
        folder = osp.join(datapath, subdir)
        savedir_hr = osp.join(savepath, 'hr', subdir)
        savedir_lr1 = osp.join(savepath, 'lr1', subdir)
        savedir_lr2 = osp.join(savepath, 'lr2', subdir)
        savedir_lr3 = osp.join(savepath, 'lr3', subdir)
        os.makedirs(savedir_hr, exist_ok=True)
        os.makedirs(savedir_lr1, exist_ok=True)
        os.makedirs(savedir_lr2, exist_ok=True)
        os.makedirs(savedir_lr3, exist_ok=True)

        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            # read image
            print(path)
            (imgname, imgext) = os.path.splitext(os.path.basename(path))
            # imgname, img = get_image_pair(folder, path)  # image to HWC-BGR, float32
            img = cv2.imread(path)
            h, w = img.shape[:-1]

            img_hr = img
            img_lr1 = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_CUBIC)
            img_lr2 = cv2.resize(img, (w//3, h//3), interpolation=cv2.INTER_CUBIC)
            img_lr3 = cv2.resize(img, (w//4, h//4), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(f'{savedir_hr}/{imgname}{format}', img_hr)
            cv2.imwrite(f'{savedir_lr1}/{imgname}{format}', img_lr1)
            cv2.imwrite(f'{savedir_lr2}/{imgname}{format}', img_lr2)
            cv2.imwrite(f'{savedir_lr3}/{imgname}{format}', img_lr3)

def genedown():
    downsample(
        datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/hr',
        savepath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/hr',
        subdirs=['query', 'bounding_box_test', 'bounding_box_train'],
        format='.jpg'
    )

    downsample(
        datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr1',
        savepath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr1',
        subdirs=['query', 'bounding_box_test', 'bounding_box_train'],
        format='.jpg'
    )

    downsample(
        datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr2',
        savepath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr2',
        subdirs=['query', 'bounding_box_test', 'bounding_box_train'],
        format='.jpg'
    )

    downsample(
        datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr3',
        savepath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr3',
        subdirs=['query', 'bounding_box_test', 'bounding_box_train'],
        format='.jpg'
    )

if __name__ == '__main__':
    # genedown()


    reid(
        datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/hr',
        pypath='/mnt/data/code/deep-person-reid-master/scripts/mlr_dukemtmc_new.py',
        savedir='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/hr/',
        config_path='/mnt/data/code/deep-person-reid-master/configs/MLR_DukeMTMC.yaml',
    )
    reid(
        datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr1',
        pypath='/mnt/data/code/deep-person-reid-master/scripts/mlr_dukemtmc_new.py',
        savedir='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/lr1/',
        config_path='/mnt/data/code/deep-person-reid-master/configs/MLR_DukeMTMC.yaml',
    )
    reid(
        datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr2',
        pypath='/mnt/data/code/deep-person-reid-master/scripts/mlr_dukemtmc_new.py',
        savedir='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/lr2/',
        config_path='/mnt/data/code/deep-person-reid-master/configs/MLR_DukeMTMC.yaml',
    )
    reid(
        datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR/lr3',
        pypath='/mnt/data/code/deep-person-reid-master/scripts/mlr_dukemtmc_new.py',
        savedir='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/lr3/',
        config_path='/mnt/data/code/deep-person-reid-master/configs/MLR_DukeMTMC.yaml',
    )


    # reid(
    #     datapath='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_HAN',
    #     pypath='/mnt/data/code/deep-person-reid-master/scripts/mlr_market1501_new.py',
    #     savedir='/mnt/data/code/reidlog/mlr_market1501_HAN/',
    #     config_path='/mnt/data/code/deep-person-reid-master/configs/mlr_market1501.yaml',
    # )

    # sr_reid_mlr_market1501()
    # sr_datasets(
    #     datapath='/mnt/data/mlr_datasets/dukemtmc-reid/DukeMTMC-reID',
    #     savepath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR',
    #     modelpaths='/mnt/data/code/SwinIR-main/model_zoo/mlr_dukemtmc_new',
    #     subdirs=['query', 'bounding_box_test', 'bounding_box_train'],
    #     format='.jpg'
    # )
    #
    # # reid the complete sub datasets respectively
    # # datapath: path of complete sub_datasets
    # # pypath: path of reid.py
    # # savedir: log path
    # # config_path: reid config path
    # reid(
    #     datapath='/mnt/data/mlr_datasets/dukemtmc-reid/MLR_DukeMTMC-reID_SR',
    #     pypath='/mnt/data/code/deep-person-reid-master/scripts/mlr_dukemtmc_new.py',
    #     savedir='/mnt/data/code/reidlog/MLR_DukeMTMC-reID_SR/',
    #     config_path='/mnt/data/code/deep-person-reid-master/configs/MLR_DukeMTMC.yaml',
    # )
    # reid('/mnt/data/datasets/MTA_reid/MTA_reid_new_sr')

