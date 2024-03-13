import os
for i in range(5):
    print('adf')
    # if(i == 1):
    #     os.system(
    #         'python /mnt/data/code/deep-person-reid-master/scripts/mlr_market1501.py --config-file /mnt/data/code/deep-person-reid-master/configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml  data.save_dir /mnt/data/code/reidlog/mlr_market1501_hr_dsx1.2/hr_dsx1.2  data.root /mnt/data/finish/market1501/mlr_market1501_hr_dsx1.2/hr_dsx1.2'
    # #     )
    # if(i == 2):
    #     os.system(
    #         'python /mnt/data/code/deep-person-reid-master/scripts/mlr_market1501.py --config-file /mnt/data/code/deep-person-reid-master/configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml  data.save_dir /mnt/data/code/reidlog/mlr_market1501_hr_dsx2/x2sr  data.root /mnt/data/finish/market1501/mlr_market1501_hr_dsx2/x2sr'
    #     )
    # if(i == 3):
    #     os.system(
    #         'python /mnt/data/code/deep-person-reid-master/scripts/mlr_market1501.py --config-file /mnt/data/code/deep-person-reid-master/configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml  data.save_dir /mnt/data/code/reidlog/mlr_market1501_hr_dsx1.2/x3sr  data.root /mnt/data/finish/market1501/mlr_market1501_hr_dsx1.2/x3sr'
    #     )
    # if(i == 4):
    #     os.system(
    #         'python /mnt/data/code/deep-person-reid-master/scripts/mlr_market1501.py --config-file /mnt/data/code/deep-person-reid-master/configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml  data.save_dir /mnt/data/code/reidlog/mlr_market1501_hr_dsx1.2/x4sr  data.root /mnt/data/finish/market1501/mlr_market1501_hr_dsx1.2/x4sr'
    #     )

    # if (i == 1):
    #     os.system(
    #         'python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root /mnt/data/datasets/market1501x1_5 data.save_dir /mnt/data/code/reidlog/market1501/market1501x1_5')
    # if (i == 2):
    #     os.system(
    #         'python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root /mnt/data/datasets/market1501x2 data.save_dir /mnt/data/code/reidlog/market1501/market1501x2')
    # if (i == 3):
    #     os.system(
    #         'python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root /mnt/data/datasets/market1501x3 data.save_dir /mnt/data/code/reidlog/market1501/market1501x3')
    # if (i == 4):
    #     os.system(
    #         'python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root /mnt/data/datasets/market1501x4 data.save_dir /mnt/data/code/reidlog/market1501/market1501x4')

    # if(i==1):
    #     os.system(
    #         'python scripts/main.py --config-file configs/dukemtmc.yaml --transforms random_flip random_erase --root /mnt/data/code/SwinIR-main_2/results/swinir_classical_sr_x2_msmt_sr_2800E/ data.save_dir /mnt/data/code/deep-person-reid-master/log/dukemtmcreid_384*192_sr_2800E')
    # if(i==2):
    #     os.system(
    #         'python scripts/main.py --config-file configs/dukemtmc.yaml --transforms random_flip random_erase --root /mnt/data/code/SwinIR-main_2/results/swinir_classical_sr_x2_msmt_sr_5300E/ data.save_dir /mnt/data/code/deep-person-reid-master/log/dukemtmcreid_384*192_sr_5300E')
    # if(i==3):
    #     os.system(
    #         'python scripts/main.py --config-file configs/dukemtmc.yaml --transforms random_flip random_erase --root /mnt/data/code/SwinIR-main_2/results/swinir_classical_sr_x2_msmt_sr_700E/ data.save_dir /mnt/data/code/deep-person-reid-master/log/dukemtmcreid_384*192_sr_700E')

#python scripts/CAVIAR.py --config-file configs/caviar.yaml data.root /mnt/data/mlr_datasets/ data.save_dir /mnt/data/code/reidlog/caviar/60_fix5_caviar_5

#python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root /mnt/data/datasets/market1501x3 data.save_dir /mnt/data/code/reidlog/market1501/market1501x3