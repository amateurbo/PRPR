import argparse
import shutil
import tqdm
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import os.path as osp
import torch
import requests
import time
import concurrent.futures
import numba

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
        #pad input image to be a multiple of window_size
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

def write_tofile(filename, content):
    cv2.imwrite(filename, content)


def sr_down(
        subdirs,
        datapath='/mnt/data/datasets/MTA_reid/MTA_reid_new',
        savepath='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr',
        modelpath='/mnt/data/code/superresolution/dukemtmc/swinir_sr_dukemtmc_x4/models/3000_G.pth',
        format='.png'
):

    # dirlists = ['x8', 'x4', 'x3', 'x2', 'hr']
    # dirlists = ['x8']
    subdirs = ['query', 'test', 'train']
    subdirs = ['test', 'train']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = 8


    for subdir in subdirs:
        print("restore the images of {}".format(subdir))
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

        start_time = time.time()
        pre_time = time.time()

        # concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            #     executor.submit(inf, idx, pre_time, folder, path, model, device)
                # inf(idx, pre_time, folder, path, model, device)

            # read image
            # if (idx + 1) % 100 == 0:
            #     print('restore the {}th image successfully!'.format(idx + 1))
            #     # print(f"当前100总耗时:{time.time() - start_time:.4f}秒")
            if (idx + 1) % 1000 == 0:
                print(f"当前1000总耗时:{time.time() - pre_time:.4f}秒")
                pre_time = time.time()

            # print(path)
            imgname, img= get_image_pair(folder, path)  # image to HWC-BGR, float32

            # h, w = img.shape[:-1]
            # if (h < 4) or (w < 4):
            #     h, w = img.shape[:-1]
            #     img = cv2.resize(img, (h * 2, w * 2), interpolation=cv2.INTER_CUBIC)

            img_x2hr, img_x3hr, img_x4hr = inference(model=model, scale=4, img_lq=img, device=device, window_size=window_size)


        # executor.submit(write_tofile, f'{savedir_x2}/{imgname}{format}', img_x2hr)
        # executor.submit(write_tofile, f'{savedir_x3}/{imgname}{format}', img_x3hr)
        # executor.submit(write_tofile, f'{savedir_x4}/{imgname}{format}', img_x4hr)
        # cv2.imwrite(f'{savedir_x2}/{imgname}{format}', img_x2hr)
        # cv2.imwrite(f'{savedir_x3}/{imgname}{format}', img_x3hr)
        # cv2.imwrite(f'{savedir_x4}/{imgname}{format}', img_x4hr)
        print(idx + 1)

        end_time = time.time()
        print(f"程序总耗时:{end_time - start_time:.4f}秒")

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
    model.load_state_dict(pretrained_model)
    # model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    return model

def resize_img_cubic(path, savepath, subdir, format):
    savedir_x2 = osp.join(savepath, 'x2', subdir)
    savedir_x3 = osp.join(savepath, 'x3', subdir)
    savedir_x4 = osp.join(savepath, 'x4', subdir)
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    img_lq = cv2.imread(f'{folder}/{imgname}{imgext}')
    h, w = img_lq.shape[:2]
    img_x2hr = cv2.resize(img_lq, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    img_x3hr = cv2.resize(img_lq, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    img_x4hr = cv2.resize(img_lq, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f'{savedir_x2}/{imgname}{format}', img_x2hr)
    cv2.imwrite(f'{savedir_x3}/{imgname}{format}', img_x3hr)
    cv2.imwrite(f'{savedir_x4}/{imgname}{format}', img_x4hr)

def sr_down_CUBIC(
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
        print("restore the images of {}".format(subdir))
        folder = osp.join(datapath, subdir)
        savedir_x2 = osp.join(savepath, 'x2', subdir)
        savedir_x3 = osp.join(savepath, 'x3', subdir)
        savedir_x4 = osp.join(savepath, 'x4', subdir)
        os.makedirs(savedir_x2, exist_ok=True)
        os.makedirs(savedir_x3, exist_ok=True)
        os.makedirs(savedir_x4, exist_ok=True)



        for path in tqdm.tqdm(sorted(glob.glob(os.path.join(folder, '*')))):



            savedir_x2 = osp.join(savepath, 'x2', subdir)
            savedir_x3 = osp.join(savepath, 'x3', subdir)
            savedir_x4 = osp.join(savepath, 'x4', subdir)
            (imgname, imgext) = os.path.splitext(os.path.basename(path))
            img_lq = cv2.imread(f'{folder}/{imgname}{imgext}')
            h, w = img_lq.shape[:2]
            img_x2hr = cv2.resize(img_lq, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            img_x3hr = cv2.resize(img_lq, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
            img_x4hr = cv2.resize(img_lq, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f'{savedir_x2}/{imgname}{format}', img_x2hr)
            cv2.imwrite(f'{savedir_x3}/{imgname}{format}', img_x3hr)
            cv2.imwrite(f'{savedir_x4}/{imgname}{format}', img_x4hr)


        # end_time = time.time()
        # print(f"程序总耗时:{end_time - start_time:.4f}秒")



def reid(datapath,
         pypath='/mnt/data/code/deep-person-reid-master/scripts/MTA_reid_new.py',
         savedir='/mnt/data/code/reidlog/MTA_reid_new/',
         config_path='/mnt/data/code/deep-person-reid-master/configs/MTA_reid.yaml',
         ):
    # ReID
    #reid command
    # dirlists = ['x2', 'x3', 'x4']
    dirlists = ['x3']
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


def sr_down(
        subdirs,
        datapath='/mnt/data/datasets/MTA_reid/MTA_reid_new',
        savepath='/mnt/data/datasets/MTA_reid/MTA_reid_new_sr',
        modelpath='/mnt/data/code/superresolution/dukemtmc/swinir_sr_dukemtmc_x4/models/3000_G.pth',
        format='.png'
):

    # dirlists = ['x8', 'x4', 'x3', 'x2', 'hr']
    # dirlists = ['x8']
    subdirs = ['query', 'test', 'train']
    subdirs = ['test', 'train']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window_size = 8


    for subdir in subdirs:
        print("restore the images of {}".format(subdir))
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

        start_time = time.time()
        pre_time = time.time()

        # concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
            #     executor.submit(inf, idx, pre_time, folder, path, model, device)
                # inf(idx, pre_time, folder, path, model, device)

            # read image
            # if (idx + 1) % 100 == 0:
            #     print('restore the {}th image successfully!'.format(idx + 1))
            #     # print(f"当前100总耗时:{time.time() - start_time:.4f}秒")
            if (idx + 1) % 1000 == 0:
                print(f"当前1000总耗时:{time.time() - pre_time:.4f}秒")
                pre_time = time.time()

            # print(path)
            imgname, img= get_image_pair(folder, path)  # image to HWC-BGR, float32

            # h, w = img.shape[:-1]
            # if (h < 4) or (w < 4):
            #     h, w = img.shape[:-1]
            #     img = cv2.resize(img, (h * 2, w * 2), interpolation=cv2.INTER_CUBIC)

            img_x2hr, img_x3hr, img_x4hr = inference(model=model, scale=4, img_lq=img, device=device, window_size=window_size)


        # executor.submit(write_tofile, f'{savedir_x2}/{imgname}{format}', img_x2hr)
        # executor.submit(write_tofile, f'{savedir_x3}/{imgname}{format}', img_x3hr)
        # executor.submit(write_tofile, f'{savedir_x4}/{imgname}{format}', img_x4hr)
        # cv2.imwrite(f'{savedir_x2}/{imgname}{format}', img_x2hr)
        # cv2.imwrite(f'{savedir_x3}/{imgname}{format}', img_x3hr)
        # cv2.imwrite(f'{savedir_x4}/{imgname}{format}', img_x4hr)
        print(idx + 1)

        end_time = time.time()


if __name__ == '__main__':

    ###----------------------------------------my mlr_dataset------------------------------------
    # sr_down(
    #     datapath='/mnt/data/datasets/MTA_reid',
    #     savepath='/mnt/data/datasets/MTA_reid/MTA_reid_l1_new',
    #     modelpath='/mnt/data/code/superresolution/div2k_l1/swinir_sr_mtareid_x4/models/1000_G.pth',
    #     subdirs=['atest'],
    #     format='.png',
    # )
    sr_down_CUBIC(
        datapath='/mnt/data/datasets/MTA_reid',
        savepath='/mnt/data/datasets/MTA_reid/CUBIC_Result',
        modelpath='/mnt/data/code/superresolution/mtareid_l1/swinir_sr_mtareid_x4/models/5000_G.pth',
        subdirs= ['query', 'test', 'train'],
        format=".png"
    )

    # reid(
    #     datapath='/mnt/data/datasets/MTA_reid/CUBIC_Result',
    #     pypath='/mnt/data/code/deep-person-reid-master/scripts/MTA_reid_new.py',
    #     savedir='/mnt/data/code/reidlog/MTA_reid_CUBIC/',
    #     config_path='/mnt/data/code/deep-person-reid-master/configs/MTA_reid.yaml',
    # )

    #
    # reid(
    #     datapath='/mnt/data/datasets/MTA_reid/MTA_reid_HAN',
    #     pypath='/mnt/data/code/deep-person-reid-master/scripts/MTA_reid_new.py',
    #     savedir='/mnt/data/code/reidlog/MTA_reid_HAN/',
    #     config_path='/mnt/data/code/deep-person-reid-master/configs/MTA_reid.yaml',
    # )