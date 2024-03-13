import  torchreid
model = 'mlr_market1501-90'
weight_path = 'log_data_market/osnet_x1_0_c1_mlr_market1501_softmax_cosinelr/2021-11-23-12-14-00model/model.pth.tar-84'


torchreid.utils.load_pretrained_weights(model, weight_path)
mlr = 'log_data_market/ '
# python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root data_market/sr_600E_noresize_mlrc1_market1501/  data.save_dir log_data_market/osnet_x1_0_sr_600E_noresize_mlrc1_market1501_softmax_cosinelr
#
