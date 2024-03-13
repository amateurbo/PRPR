import os
import os.path as osp

models_path = '/home/gpu/code/SwinIR-main/model_zoo/MTA_reid' #model path
query = ' --folder_lq testsets/MTA_reid/query/ --folder_gt testsets/MTA_reid/query/' #query 低分图像路径 高分图像路径
train = ' --folder_lq testsets/MTA_reid/train/ --folder_gt testsets/MTA_reid/train/' #train 低分图像路径 高分图像路径
test = ' --folder_lq testsets/MTA_reid/test/ --folder_gt testsets/MTA_reid/test/' #train 低分图像路径 高分图像路径


dataname = 'MTA_reid'

model_scales = ['x2', 'x3', 'x4', 'x8']

for scale in model_scales:
    numscale = scale.split('x')[1]
    sr_task = 'python main_nogt_swinir_MTA_reid.py --task classical_sr --scale ' + numscale + ' --training_patch_size 64 --window_size 8 --model_path '
    dir = osp.join(models_path, scale)
    modelname = os.listdir(dir)[0]
    model_path = osp.join(models_path, scale, modelname)
    sr_query = sr_task + model_path + query
    sr_train = sr_task + model_path + train
    sr_test = sr_task + model_path + test
    reid_log = ' data.save_dir /mnt/data/code/reidlog/MTA_reid_new/'+ scale +'/osnet_x1_0_test_'+ scale +'_MTA_reid_softmax_cosinelr'

    #超分
    os.system(sr_query)
    # os.system('cp -r /mnt/data/datasets/DukeMTMC-reID/bounding_box_train/  results/swinir_classical_sr_'+ scale +'_testsets/MTA_reid/')
    # os.system('ln -s /mnt/data/datasets/DukeMTMC-reID/MTA_reid/bounding_box_test/  results/swinir_classical_sr_'+ scale +'_testsets/MTA_reid/')
    os.system(sr_train)
    os.system(sr_test)

    #更改文件夹名称
    dirdatasets = '../results_MTA_reid_new/swinir_classical_sr_' + dataname + '_' + scale + '_' + modelname.split('.')[0]
    rename_dirdatasets = 'mv ../results_MTA_reid_new/swinir_classical_sr_'+ scale +'_testsets ' + dirdatasets
    os.system(rename_dirdatasets)

    # ReID
    #reid command
    dirdatasets = '../results_MTA_reid_new/swinir_classical_sr_' + dataname + '_' + scale + '_' + modelname.split('.')[0]
    root = ' data.root /mnt/data/code/SwinIR-main/' + dirdatasets
    reid_task = 'python /mnt/data/code/deep-person-reid-master/scripts/MTA_reid.py ' \
                '--config-file /mnt/data/code/deep-person-reid-master/configs/MTA_reid.yaml '
    reid_log = reid_log.replace('test', scale + '_' + modelname.split('.')[0])
    reid =reid_task + root + reid_log
    os.system(reid)