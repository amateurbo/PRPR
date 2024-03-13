import subprocess
import os
import os.path as osp

models_path = '/home/gpu/code/SwinIR-main/model_zoo/CUHK03/x2' #model path
query = ' --folder_lq testsets/MLR_CUHK03/query/ --folder_gt testsets/MLR_CUHK03/query/' #query 低分图像路径 高分图像路径
train = ' --folder_lq testsets/MLR_CUHK03/bounding_box_train/ --folder_gt testsets/MLR_CUHK03/bounding_box_train/' #train 低分图像路径 高分图像路径
test = ' --folder_lq testsets/MLR_CUHK03/bounding_box_test/ --folder_gt testsets/MLR_CUHK03/bounding_box_test/' #train 低分图像路径 高分图像路径
sr_task = 'python main_nogt_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --window_size 8 --model_path '
set = 'x2'
dataname = 'MLR_CUHK03_'

dir_list = os.listdir(models_path)
log = []

#遍历模型 超分
for i in  range(len(dir_list)):
    model_path = osp.join(models_path, dir_list[i])
    sr_query = sr_task + model_path + query
    sr_train = sr_task + model_path + train
    reid_log = ' data.save_dir /mnt/data/code/reidlog/CUHK03/x2/osnet_x1_0_test_x2_CUHK03_softmax_cosinelr'

    #超分
    os.system(sr_query)
    os.system('cp -r /mnt/data/datasets/CUHK03/bounding_box_train/  results/swinir_classical_sr_x2_testsets/MLR_CUHK03/')
    os.system('ln -s /mnt/data/datasets/CUHK03/MLR_CUHK03/bounding_box_test/  results/swinir_classical_sr_x2_testsets/MLR_CUHK03/')
    os.system(sr_train)

    #更改文件夹名称
    dirdatasets = 'results/swinir_classical_sr_' + set + '_' + dir_list[i].split('.')[0]
    rename_dirdatasets = 'mv results/swinir_classical_sr_x2_testsets ' + dirdatasets
    os.system(rename_dirdatasets)

    # ReID
    #reid command
    dirdatasets = 'results/swinir_classical_sr_'+ dataname  + set + '_' + dir_list[i].split('.')[0]
    root = ' data.root /mnt/data/code/SwinIR-main/' + dirdatasets
    reid_task = 'python /mnt/data/code/deep-person-reid-master/scripts/MLR_CUHK03.py ' \
                '--config-file /mnt/data/code/deep-person-reid-master/configs/MLR_CUHK03.yaml '
    reid_log = reid_log.replace('test', set + '_' + dir_list[i].split('.')[0])
    reid =reid_task + root + reid_log
    os.system(reid)







