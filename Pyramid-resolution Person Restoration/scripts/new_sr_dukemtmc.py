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
    def tensor2uint(img):
        img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        return np.uint8((img * 255.0).round())

    def imsave(img, img_path):
        img = np.squeeze(img)
        if img.ndim == 3:
            img = img[:, :, [2, 1, 0]]
        cv2.imwrite(img_path, img)


    dirlists = ['x8', 'x4', 'x3', 'x2', 'hr']
    # dirlists = ['x8']
    # subdirs = ['query', 'test', 'train']
    # subdirs = ['test', 'train']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = 8

    for dirlist in dirlists:

        for subdir in subdirs:
            folder = osp.join(datapath, dirlist, subdir)
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
            scale = getmodelscale(dirlist)
            # if scale == 1:

            for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
                # read image
                print(path)
                imgname, img= get_image_pair(folder, path)  # image to HWC-BGR, float32
                img_hr = inv_norm(img)
                img = np.expand_dims(img, 0)
                img = torch.tensor(img)
                img.permute(0,3,1,2)
                result = model(img)
                img_x2hr, img_x3hr, img_x4hr = inference(model=model, scale=4, img_lq=img, device=device, window_size=window_size)

                # img_x4hr = tensor2uint(img_x4hr)
                # img_x3hr = tensor2uint(img_x3hr)
                # img_x2hr = tensor2uint(img_x2hr)
                imsave(img_x4hr, f'{savedir_x4}/{imgname}{format}')
                imsave(img_x3hr, f'{savedir_x3}/{imgname}{format}')
                imsave(img_x2hr, f'{savedir_x2}/{imgname}{format}')
                # cv2.imwrite(f'{savedir_x2}/{imgname}{format}', img_x2hr)
                # cv2.imwrite(f'{savedir_x3}/{imgname}{format}', img_x3hr)
                # cv2.imwrite(f'{savedir_x4}/{imgname}{format}', img_x4hr)




def get_image_pair(folder, path):
    def imread_uint(path, n_channels=3):
        #  input: path
        # output: HxWx3(RGB or GGG), or HxWx1 (G)
        if n_channels == 1:
            img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
            img = np.expand_dims(img, axis=2)  # HxWx1
        elif n_channels == 3:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        return img

    def uint2single(img):
        return np.float32(img / 255.)
    img_lq = imread_uint(path)
    img_lq = uint2single(img_lq)
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    # img_lq = cv2.imread(f'{folder}/{imgname}{imgext}', cv2.IMREAD_COLOR).astype(
    #     np.float32) / 255.
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

    # # original mlr_dataset split to hr x2 x3 x4 x8 sub_dataset
    # # path: original mlr_dataset datapath
    # # savepath: sub_datasets savepath
    # splitdata(
    #     path='/mnt/data/mlr_datasets/dukemtmc-reid/DukeMTMC-reID',
    #     savepath='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_DukeMTMC-reID_new',
    # )

    # sub_datasets through super-resolution or downsampling to get complete dataset
    # datapath: path of sub_datasets
    # savepath: savepath of complete sub_datasets
    # modelpaths: path of super-resolution models
    # subdirs: subfolders of datasets
    # format: format of pictures in dataset
    sr_down(
        datapath='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_DukeMTMC-reID_new',
        savepath='/mnt/data/mlr_datasets/dukemtmc-reid/a_new_mlr_DukeMTMC-reID_new_sr',
        modelpath='/mnt/data/code/superresolution/market1501_l1/swinir_sr_market1501_x4/models/5000_G.pth',
        # subdirs=['query', 'bounding_box_test', 'bounding_box_train'],
        subdirs=['query'],
        format='.jpg'
    )

    # reid the complete sub datasets respectively
    # datapath: path of complete sub_datasets
    # pypath: path of reid.py
    # savedir: log path
    # config_path: reid config path
    # reid(
    #     datapath='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_DukeMTMC-reID_new_sr_myself_l2_nofreeze_r_10',
    #     pypath='/mnt/data/code/deep-person-reid-master/scripts/mlr_dukemtmc_new.py',
    #     savedir='/mnt/data/code/reidlog/mlr_dukemtmc_new_sr_myself_l2_nofreeze_r_10/',
    #     config_path='/mnt/data/code/deep-person-reid-master/configs/MLR_DukeMTMC.yaml',
    # )

