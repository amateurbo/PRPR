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
        modelpaths='/mnt/data/code/SwinIR-main/model_zoo/MTA_reid_new',
        format='.png'
):

    dirlists = ['hr', 'x2', 'x3', 'x4', 'x8']
    # dirlists = ['x8']
    # subdirs = ['query', 'test', 'train']
    # subdirs = ['test', 'train']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = 8

    for dirlist in dirlists:

        for subdir in subdirs:
            folder = osp.join(datapath, dirlist, subdir)
            savedir_hr = osp.join(savepath, 'hr', subdir)
            savedir_x2 = osp.join(savepath, 'x2', subdir)
            savedir_x3 = osp.join(savepath, 'x3', subdir)
            savedir_x4 = osp.join(savepath, 'x4', subdir)
            savedir_x8 = osp.join(savepath, 'x8', subdir)
            os.makedirs(savedir_hr, exist_ok=True)
            os.makedirs(savedir_x2, exist_ok=True)
            os.makedirs(savedir_x3, exist_ok=True)
            os.makedirs(savedir_x4, exist_ok=True)
            os.makedirs(savedir_x8, exist_ok=True)

            scale = getmodelscale(dirlist)
            if scale == 1:
                for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
                    # read image
                    print(path)
                    imgname, img= get_image_pair(folder, path)  # image to HWC-BGR, float32
                    img_hr = inv_norm(img)
                    w, h = img_hr.shape[:-1]
                    img_x2 = cv2.resize(img_hr, (int(h/2), int(w/2)), interpolation=cv2.INTER_CUBIC)
                    img_x3 = cv2.resize(img_hr, (int(h/3), int(w/3)), interpolation=cv2.INTER_CUBIC)
                    img_x4 = cv2.resize(img_hr, (int(h/4), int(w/4)), interpolation=cv2.INTER_CUBIC)
                    img_x8 = cv2.resize(img_hr, (int(h/8), int(w/8)), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(f'{savedir_hr}/{imgname}{format}', img_hr)
                    cv2.imwrite(f'{savedir_x2}/{imgname}{format}', img_x2)
                    cv2.imwrite(f'{savedir_x3}/{imgname}{format}', img_x3)
                    cv2.imwrite(f'{savedir_x4}/{imgname}{format}', img_x4)
                    cv2.imwrite(f'{savedir_x8}/{imgname}{format}', img_x8)

            elif scale == 2:
                modelpath = getmodelpath(modelpaths, dirlist)

                model_2 = define_model(model_path=modelpath, scale=2)
                model_2.eval()
                model_2 = model_2.to(device)

                save_dir = osp.join(savepath, dirlist, subdir)
                os.makedirs(save_dir, exist_ok=True)
                for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
                    # read image
                    print(path)
                    imgname, img= get_image_pair(folder, path)  # image to HWC-BGR, float32

                    h, w = img.shape[:-1]
                    if (h < 4) or (w < 4):
                        h, w = img.shape[:-1]
                        img = cv2.resize(img, (h * 2, w * 2), interpolation=cv2.INTER_CUBIC)

                    img_hr = inference(model=model_2, scale=2, img_lq=img, device=device, window_size=window_size)
                    # img_x2 = inv_norm(img)
                    # w, h = img_x2.shape[:-1]
                    # img_x3 = cv2.resize(img_x2, (int(h/1.5), int(w/1.5)), interpolation=cv2.INTER_CUBIC)
                    # img_x4 = cv2.resize(img_x2, (int(h/2), int(w/2)), interpolation=cv2.INTER_CUBIC)
                    # img_x8 = cv2.resize(img_x2, (int(h/4), int(w/4)), interpolation=cv2.INTER_CUBIC)
                    w, h = img_hr.shape[:-1]
                    img_x2 = cv2.resize(img_hr, (int(h/2), int(w/2)), interpolation=cv2.INTER_CUBIC)
                    img_x3 = cv2.resize(img_hr, (int(h/3), int(w/3)), interpolation=cv2.INTER_CUBIC)
                    img_x4 = cv2.resize(img_hr, (int(h/4), int(w/4)), interpolation=cv2.INTER_CUBIC)
                    img_x8 = cv2.resize(img_hr, (int(h/8), int(w/8)), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(f'{savedir_hr}/{imgname}{format}', img_hr)
                    cv2.imwrite(f'{savedir_x2}/{imgname}{format}', img_x2)
                    cv2.imwrite(f'{savedir_x3}/{imgname}{format}', img_x3)
                    cv2.imwrite(f'{savedir_x4}/{imgname}{format}', img_x4)
                    cv2.imwrite(f'{savedir_x8}/{imgname}{format}', img_x8)

            elif scale == 3:
                modelpath = getmodelpath(modelpaths, 'x2')
                model_2 = define_model(model_path=modelpath, scale=2)
                model_2.eval()
                model_2 = model_2.to(device)

                modelpath = getmodelpath(modelpaths, 'x3')
                model_3 = define_model(model_path=modelpath, scale=3)
                model_3.eval()
                model_3 = model_3.to(device)

                save_dir = osp.join(savepath, dirlist, subdir)
                os.makedirs(save_dir, exist_ok=True)
                for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
                    # read image
                    print(path)
                    imgname, img= get_image_pair(folder, path)  # image to HWC-BGR, float32

                    w, h = img.shape[:-1]
                    if (h < 4) or (w < 4):
                        h, w = img.shape[:-1]
                        img = cv2.resize(img, (h * 2, w * 2), interpolation=cv2.INTER_CUBIC)

                    img_hr = inference(model=model_3, scale=3, img_lq=img, device=device, window_size=window_size)
                    # img_srx2 = inference(model=model_2, scale=2, img_lq=img, device=device, window_size=window_size)
                    # img_x2 = cv2.resize(img_srx2, (int(img_srx2.shape[1]/1.3), int(img_srx2.shape[0]/1.3)), interpolation=cv2.INTER_CUBIC)
                    # w, h = img.shape[:-1]
                    # img_x3 = inv_norm(img)
                    # img_x4 = cv2.resize(img_x3, (int(h/1.3), int(w/1.3)), interpolation=cv2.INTER_CUBIC)
                    # img_x8 = cv2.resize(img_x3, (int(h/2.6), int(w/2.6)), interpolation=cv2.INTER_CUBIC)
                    w, h = img_hr.shape[:-1]
                    img_x2 = cv2.resize(img_hr, (int(h/2), int(w/2)), interpolation=cv2.INTER_CUBIC)
                    img_x3 = cv2.resize(img_hr, (int(h/3), int(w/3)), interpolation=cv2.INTER_CUBIC)
                    img_x4 = cv2.resize(img_hr, (int(h/4), int(w/4)), interpolation=cv2.INTER_CUBIC)
                    img_x8 = cv2.resize(img_hr, (int(h/8), int(w/8)), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(f'{savedir_hr}/{imgname}{format}', img_hr)
                    cv2.imwrite(f'{savedir_x2}/{imgname}{format}', img_x2)
                    cv2.imwrite(f'{savedir_x3}/{imgname}{format}', img_x3)
                    cv2.imwrite(f'{savedir_x4}/{imgname}{format}', img_x4)
                    cv2.imwrite(f'{savedir_x8}/{imgname}{format}', img_x8)

            elif scale == 4:
                modelpath = getmodelpath(modelpaths, 'x2')
                model_2 = define_model(model_path=modelpath, scale=2)
                model_2.eval()
                model_2 = model_2.to(device)

                modelpath = getmodelpath(modelpaths, 'x4')
                model_4 = define_model(model_path=modelpath, scale=4)
                model_4.eval()
                model_4 = model_4.to(device)

                save_dir = osp.join(savepath, dirlist, subdir)
                os.makedirs(save_dir, exist_ok=True)
                for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
                    # read image
                    print(path)
                    imgname, img = get_image_pair(folder, path)  # image to HWC-BGR, float32

                    h, w = img.shape[:-1]
                    if (h < 4) or (w < 4):
                        h, w = img.shape[:-1]
                        img = cv2.resize(img, (h * 2, w * 2), interpolation=cv2.INTER_CUBIC)

                    img_hr = inference(model=model_4, scale=4, img_lq=img, device=device, window_size=window_size)
                    # img_x2 = inference(model=model_2, scale=2, img_lq=img, device=device, window_size=window_size)
                    # img_x3 = cv2.resize(img_x2, (int(img_x2.shape[1] / 1.5), int(img_x2.shape[0] / 1.5)), interpolation=cv2.INTER_CUBIC)
                    # img_x4 = inv_norm(img)
                    # w, h = img_x4.shape[:-1]
                    # img_x8 = cv2.resize(img_x4, (int(h / 2), int(w / 2)), interpolation=cv2.INTER_CUBIC)
                    w, h = img_hr.shape[:-1]
                    img_x2 = cv2.resize(img_hr, (int(h/2), int(w/2)), interpolation=cv2.INTER_CUBIC)
                    img_x3 = cv2.resize(img_hr, (int(h/3), int(w/3)), interpolation=cv2.INTER_CUBIC)
                    img_x4 = cv2.resize(img_hr, (int(h/4), int(w/4)), interpolation=cv2.INTER_CUBIC)
                    img_x8 = cv2.resize(img_hr, (int(h/8), int(w/8)), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(f'{savedir_hr}/{imgname}{format}', img_hr)
                    cv2.imwrite(f'{savedir_x2}/{imgname}{format}', img_x2)
                    cv2.imwrite(f'{savedir_x3}/{imgname}{format}', img_x3)
                    cv2.imwrite(f'{savedir_x4}/{imgname}{format}', img_x4)
                    cv2.imwrite(f'{savedir_x8}/{imgname}{format}', img_x8)

            elif scale == 8:
                modelpath = getmodelpath(modelpaths, 'x2')
                model_2 = define_model(model_path=modelpath, scale=2)
                model_2.eval()
                model_2 = model_2.to(device)

                modelpath = getmodelpath(modelpaths, 'x4')
                model_4 = define_model(model_path=modelpath, scale=4)
                model_4.eval()
                model_4 = model_4.to(device)

                modelpath = getmodelpath(modelpaths, 'x8')
                model_8 = define_model(model_path=modelpath, scale=8)
                model_8.eval()
                model_8 = model_8.to(device)

                save_dir = osp.join(savepath, dirlist, subdir)
                os.makedirs(save_dir, exist_ok=True)
                for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
                    # read image
                    print(path)
                    imgname, img = get_image_pair(folder, path)  # image to HWC-BGR, float32

                    h, w = img.shape[:-1]
                    if (h < 4) or (w < 4):
                        h, w = img.shape[:-1]
                        img = cv2.resize(img, (h * 2, w * 2), interpolation=cv2.INTER_CUBIC)

                    img_hr = inference(model=model_8, scale=8, img_lq=img, device=device, window_size=window_size)
                    # img_x2 = inference(model=model_4, scale=4, img_lq=img, device=device, window_size=window_size)
                    # img_x3 = cv2.resize(img_x2, (int(img_x2.shape[1] / 1.5), int(img_x2.shape[0] / 1.5)), interpolation=cv2.INTER_CUBIC)
                    # img_x4 = inference(model=model_2, scale=2, img_lq=img, device=device, window_size=window_size)
                    # img_x8 = inv_norm(img)
                    w, h = img_hr.shape[:-1]
                    img_x2 = cv2.resize(img_hr, (int(h/2), int(w/2)), interpolation=cv2.INTER_CUBIC)
                    img_x3 = cv2.resize(img_hr, (int(h/3), int(w/3)), interpolation=cv2.INTER_CUBIC)
                    img_x4 = cv2.resize(img_hr, (int(h/4), int(w/4)), interpolation=cv2.INTER_CUBIC)
                    img_x8 = cv2.resize(img_hr, (int(h/8), int(w/8)), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(f'{savedir_hr}/{imgname}{format}', img_hr)
                    cv2.imwrite(f'{savedir_x2}/{imgname}{format}', img_x2)
                    cv2.imwrite(f'{savedir_x3}/{imgname}{format}', img_x3)
                    cv2.imwrite(f'{savedir_x4}/{imgname}{format}', img_x4)
                    cv2.imwrite(f'{savedir_x8}/{imgname}{format}', img_x8)


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
    dirlists = ['hr', 'x2', 'x3', 'x4', 'x8']
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

def splitdata(path, savepath, x4=512, x3=2048, x2=3640, hr=8192):

    # subdirs = ['query', 'test', 'train']
    subdirs = ['query', 'bounding_box_test', 'bounding_box_train']

    for subdir in subdirs:
        subpath = osp.join(path, subdir)
        savesubpath_hr = osp.join(savepath, 'hr', subdir)
        checkpath(savesubpath_hr)
        savesubpath_x2 = osp.join(savepath, 'x2', subdir)
        checkpath(savesubpath_x2)
        savesubpath_x3 = osp.join(savepath, 'x3', subdir)
        checkpath(savesubpath_x3)
        savesubpath_x4 = osp.join(savepath, 'x4', subdir)
        checkpath(savesubpath_x4)
        savesubpath_x8 = osp.join(savepath, 'x8', subdir)
        checkpath(savesubpath_x8)


        imgnames = os.listdir(subpath)
        for imgname in imgnames:
            img = cv2.imread(osp.join(subpath, imgname))
            height, width = img.shape[:-1]
            pixels = height * width

            if pixels >= hr:
                saveimgpath = osp.join(savesubpath_hr, imgname)
            elif pixels >= x2:
                saveimgpath = osp.join(savesubpath_x2, imgname)
            elif pixels >= x3:
                saveimgpath = osp.join(savesubpath_x3, imgname)
            elif pixels >= x4:
                saveimgpath = osp.join(savesubpath_x4, imgname)
            else:
                saveimgpath = osp.join(savesubpath_x8, imgname)

            cv2.imwrite(saveimgpath, img)



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
        savepath='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_DukeMTMC-reID_new_sr_down_2',
        modelpaths='/mnt/data/code/SwinIR-main/model_zoo/mlr_dukemtmc_new',
        subdirs=['query', 'bounding_box_test', 'bounding_box_train'],
        format='.jpg'
    )

    # reid the complete sub datasets respectively
    # datapath: path of complete sub_datasets
    # pypath: path of reid.py
    # savedir: log path
    # config_path: reid config path
    reid(
        datapath='/mnt/data/mlr_datasets/dukemtmc-reid/mlr_DukeMTMC-reID_new_sr_down_2',
        pypath='/mnt/data/code/deep-person-reid-master/scripts/mlr_dukemtmc_new.py',
        savedir='/mnt/data/code/reidlog/mlr_dukemtmc_new_2/',
        config_path='/mnt/data/code/deep-person-reid-master/configs/MLR_DukeMTMC.yaml',
    )
    # reid('/mnt/data/datasets/MTA_reid/MTA_reid_new_sr')