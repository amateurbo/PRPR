import sr_down_mlr_dukemtmc_2 as sd

if __name__ == '__main__':

    # # original mlr_dataset split to hr x2 x3 x4 x8 sub_dataset
    # # path: original mlr_dataset datapath
    # # savepath: sub_datasets savepath
    # sd.splitdata(
    #     path='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/market1501',
    #     savepath='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new',
    # )
    #
    # sub_datasets through super-resolution or downsampling to get complete dataset
    # datapath: path of sub_datasets
    # savepath: savepath of complete sub_datasets
    # modelpaths: path of super-resolution models
    # subdirs: subfolders of datasets
    # format: format of pictures in dataset
    sd.sr_down(
        datapath='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new',
        savepath='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_sr_down_2_new',
        modelpaths='/mnt/data/code/SwinIR-main/model_zoo/mlr_market1501_new',
        subdirs=['query', 'bounding_box_test', 'bounding_box_train'],
        format='.jpg'
    )

    # reid the complete sub datasets respectively
    # datapath: path of complete sub_datasets
    # pypath: path of reid.py
    # savedir: log path
    # config_path: reid config path
    sd.reid(
        datapath='/mnt/data/code/Generate_Mlrdataset/MLR_Market1501/split0/mlr_market1501_new_sr_down_2_new',
        pypath='/mnt/data/code/deep-person-reid-master/scripts/mlr_market1501_new.py',
        savedir='/mnt/data/code/reidlog/mlr_market1501_new_2_new/',
        config_path='/mnt/data/code/deep-person-reid-master/configs/mlr_market1501.yaml',
    )