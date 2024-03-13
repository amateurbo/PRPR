import subprocess
import os

models_path = '/home/gpu/code/SwinIR-main/model_zoo/MTA_reid_x8/' #model path
query = ' --folder_lq testsets/MTA_reid/query/ --folder_gt testsets/MTA_reid/query/' #query 低分图像路径 高分图像路径
train = ' --folder_lq testsets/MTA_reid/train/ --folder_gt testsets/MTA_reid/train/' #train 低分图像路径 高分图像路径
test = ' --folder_lq testsets/MTA_reid/test/ --folder_gt testsets/MTA_reid/test/' #train 低分图像路径 高分图像路径
sr_task = 'python main_nogt_swinir_MTA_reid.py --task classical_sr --scale 8 --training_patch_size 64 --window_size 8 --model_path '
set = 'x8'

dir_list = os.listdir(models_path)
log = []

#遍历模型 超分
for i in  range(len(dir_list)):
    model_path =models_path + dir_list[i]
    sr_query = sr_task + model_path + query
    sr_train = sr_task + model_path + train
    sr_test = sr_task + model_path + test
    reid_log = ' data.save_dir /mnt/data/code/reidlog/MTA_reid/x8/osnet_x1_0_test_x8_MTA_reid_softmax_cosinelr'

    # #超分
    # os.system(sr_query)
    # os.system(sr_train)
    # #将其他数据复制过来
    # path = 'results/swinir_classical_sr_x8_testsets/MTA_reid'
    # os.system('ln -s /mnt/data/datasets/MTA_reid/test/ ' + path)
    # # os.system(sr_test)
    #
    # #更改文件夹名称
    # dirdatasets = 'results/swinir_classical_sr_' + set + '_' + dir_list[i].split('.')[0]
    # rename_dirdatasets = 'mv results/swinir_classical_sr_x8_testsets ' + dirdatasets
    # os.system(rename_dirdatasets)

    # ReID
    #reid command
    dirdatasets = 'results/swinir_classical_sr_' + set + '_' + dir_list[i].split('.')[0]
    root = ' data.root /mnt/data/code/SwinIR-main/' + dirdatasets
    reid_task = 'python /mnt/data/code/deep-person-reid-master/scripts/MTA_reid.py ' \
                '--config-file /mnt/data/code/deep-person-reid-master/configs/MTA_reid.yaml '
    reid_log = reid_log.replace('test', set + '_' + dir_list[i].split('.')[0])
    reid =reid_task + root + reid_log
    os.system(reid)







