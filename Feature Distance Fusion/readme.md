### Installation

The installation can refer to [Torchreid](https://github.com/KaiyangZhou/deep-person-reid).

```python
conda create --name torchreid python=3.7 

conda activate torchreid

pip -r requirements.txt

python setup.py develop

```

### Pretrained models
[osnet_x1_0](https://drive.google.com/file/d/1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY/view?usp=sharing)
    Path: ~/.cache/torch/checkpoints

### Feature Distance Fusion Example

```python
import scripts.compute_weight_distance as cwd
from scripts.compute_weight_distance import computeCMC
from scripts.mlr_dukemtmc_new import MLR_DukeMTMC

from torchreid import metrics
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

save_dir = '/data/code/reidlog/mlr_dukemtmc_l1/test'
log_name = 'test.log'
log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
sys.stdout = Logger(osp.join(save_dir, log_name))
weight = True

# Root is the path to the dataset
# Model_load_weights is the weight of the training
# Dataset is the name of the dataset

distmatori, q_pids, g_pids, q_camids, g_camids, qsize1_, qsize2_, qsize_, gsize_, qf, gf = cwd.main(
    root='/data/datasets/DukeMTMC-reID/MLR_DukeMTMC',
    model_load_weights='/data/code/reidlog/MLR_DukeMTMC/x0/2022-03-13-14-46-24model/model.pth.tar-219',
    dataset='MLR_DukeMTMC'
)

distmat_hr4, _, _, _, _, _, _, _, _, qfhr, gfhr = cwd.main(
    root='/data/mlr_datasets/dukemtmc-reid/mlr_dukemtmc_l1/x4',
    model_load_weights='/data/code/reidlog/mlr_dukemtmc_l1/x4/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC',
)

distmat_hr3, _, _, _, _, _, _, _, _, qflr1, gflr1 = cwd.main(
    root='/data/mlr_datasets/dukemtmc-reid/mlr_dukemtmc_l1/x3',
    model_load_weights='/data/code/reidlog/mlr_dukemtmc_l1/x3/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC',
)

distmat_hr2, _, _, _, _, _, _, _, _, qflr2, gflr2 = cwd.main(
    root='/data/mlr_datasets/dukemtmc-reid/mlr_dukemtmc_l1/x2',
    model_load_weights='/data/code/reidlog/mlr_dukemtmc_l1/x2/model/model-best.pth.tar',
    dataset='MLR_DukeMTMC',
)

print('dist ori*2+hr+lr1+lr2')
computeCMC(distmatori*2 + distmat_hr4 + distmat_hr3 + distmat_hr2, q_pids, g_pids, q_camids, g_camids)
print('\n\n')

