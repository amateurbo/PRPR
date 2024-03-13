import os
import subprocess

p = subprocess.Popen('python scripts/main.py --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml --transforms random_flip random_erase --root data/market1501/  data.save_dir log_mlr/osnet_x1_0_market1501_default_softmax_cosinelr', shell=True, stdout=subprocess.PIPE, bufsize=1)
for line in iter(p.stdout.readline, b''):
    print(line)
p.stdout.close()
p.wait()