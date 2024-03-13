import subprocess
import os

models_path = '/home/gpu/code/SwinIR-main/model_zoo/rcam_duke_s64w8_x2/' #model path
query = ' --folder_lq testsets/mlr_cam_dukemtmc/DukeMTMC-reID/query/ --folder_gt testsets/mlr_cam_dukemtmc/DukeMTMC-reID/hrquery/' #query 低分图像路径 高分图像路径
train = ' --folder_lq testsets/mlr_cam_dukemtmc/DukeMTMC-reID/bounding_box_train/ --folder_gt testsets/mlr_cam_dukemtmc/DukeMTMC-reID/hrtrain/' #train 低分图像路径 高分图像路径
sr_task = 'python main_test_swinir.py --task classical_sr --scale 2 --training_patch_size 64 --window_size 8 --model_path '
set = 'mlr_rcduke_s64w8_x2' #random choice camera dukemtmc

dir_list = os.listdir(models_path)
log = []

#遍历模型 超分
for i in  range(len(dir_list)):
    model_path =models_path + dir_list[i]
    sr_query = sr_task + model_path + query
    sr_train = sr_task + model_path + train
    reid_log = ' data.save_dir log_mlr_rcduke_s64w8_x2/osnet_x1_0_test_x2_dukemtmc_softmax_cosinelr'


    #超分
    #超分query
    s1, s2, s3 = None, None, None
    p = subprocess.Popen(sr_query, shell=True, stdout=subprocess.PIPE, bufsize=1)
    for line in iter(p.stdout.readline, b''):
        s1 = s2
        s2 = s3
        s3 = str(line)
        print(line)
    p.stdout.close()
    p.wait()
    # 记录超分结果
    result = open('results/result.txt', 'a')
    result.writelines(dir_list[i] + ' \n' + s1 + '\n' + s2 + '\n' + s3 + '\n')
    result.close()

    #将其他数据复制过来
    os.system('ln -s /mnt/data/reiddata/DukeMTMC-reID/dukemtmc-reid/DukeMTMC-reID/bounding_box_test/ results/swinir_classical_sr_x2_testsets/mlr_cam_dukemtmc/DukeMTMC-reID/')
    os.system('cp -r /mnt/data/reiddata/DukeMTMC-reID/dukemtmc-reid/DukeMTMC-reID/bounding_box_train results/swinir_classical_sr_x2_testsets/mlr_cam_dukemtmc/DukeMTMC-reID/bounding_box_train')

    # 超分train
    s1, s2, s3 = None, None, None
    p = subprocess.Popen(sr_train, shell=True, stdout=subprocess.PIPE, bufsize=1)
    for line in iter(p.stdout.readline, b''):
        s1 = s2
        s2 = s3
        s3 = str(line)
        print(line)
    p.stdout.close()
    p.wait()
    # 记录超分结果
    result = open('results/result.txt', 'a')
    result.writelines(s1 + '\n' + s2 + '\n' + s3 + '\n')
    result.close()


    #更改文件夹名称

    os.system('mv results/swinir_classical_sr_x2_testsets/mlr_cam_dukemtmc/ results/swinir_classical_sr_x2_testsets/dukemtmc-reid/')
    dirdatasets = 'results/swinir_classical_sr_x2_' + set + '_' + dir_list[i].split('.')[0]
    rename_dirdatasets = 'mv results/swinir_classical_sr_x2_testsets ' + dirdatasets
    os.system(rename_dirdatasets)


    # ReID


    #reid command
    dirdatasets = 'results/swinir_classical_sr_x2_' + set + '_' + dir_list[i].split('.')[0]
    root = ' --root /mnt/data/code/SwinIR-main/' + dirdatasets

    reid_task = 'python /mnt/data/code/deep-person-reid-master/scripts/main.py --config-file /mnt/data/code/deep-person-reid-master/configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml -s dukemtmcreid -t dukemtmcreid --transforms random_flip random_erase '
    reid_log = reid_log.replace('test', set + '_' + dir_list[i].split('.')[0])
    reid =reid_task + root + reid_log
    s1, s2, s3, s4, s5, s6, s7, s8, s9 = None, None, None, None, None, None, None, None, None
    # os.chdir('/mnt/data/code/deep-person-reid-master')
    p = subprocess.Popen(reid, shell=True, stdout=subprocess.PIPE, bufsize=1)
    for line in iter(p.stdout.readline, b''):
        s1 = s2; s2 = s3; s3 = s4; s4 = s5; s5 = s6; s6 = s7; s7 = s8; s8 = s9
        s9 = str(line)
        print(line)
    p.stdout.close()
    p.wait()
    # 记录ReID结果
    result = open('results/result.txt', 'a')
    result.writelines(dir_list[i] + s1 + ' \n' + s2 + ' \n' + s4 + ' \n' + s5 + ' \n' + s6 + ' \n' + s7 + '\n\n')
    result.close()





