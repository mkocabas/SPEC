# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import yaml
import time
import shutil
import operator
from loguru import logger
from functools import reduce
from yacs.config import CfgNode as CN
from flatten_dict import flatten, unflatten

# from pare.utils.cluster import execute_task_on_cluster
from pare.core.config import get_grid_search_configs

DATASET_FOLDERS = {
    # 'pano': '/is/cluster/work/mkocabas/datasets/panorama_dataset/preprocessed_data',
    'pano': '/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/data/dataset_folders/20210215-preprocessed_pano_dataset',
    'pano_scalenet': '/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/data/dataset_folders/20210219-preprocessed_pano_dataset'
}
##### CONFIGS #####
hparams = CN()

# General settings
hparams.LOG_DIR = 'logs/camcalib'
hparams.METHOD = 'camcalib'
hparams.EXP_NAME = 'default'
hparams.RUN_TEST = False
hparams.PROJECT_NAME = 'camcalib'
hparams.SEED_VALUE = -1

hparams.SYSTEM = CN()
hparams.SYSTEM.GPU = ''
hparams.SYSTEM.CLUSTER_NODE = 0.0

# Dataset hparams
hparams.DATASET = CN()
hparams.DATASET.TRAIN_DS = 'pano' # 'pano_agora'
hparams.DATASET.VAL_DS = 'pano' # 'pano_agora'
hparams.DATASET.MIN_RES = 600
hparams.DATASET.MAX_RES = 1000
hparams.DATASET.BATCH_SIZE = 64
hparams.DATASET.NUM_WORKERS = 8
hparams.DATASET.PIN_MEMORY = True
hparams.DATASET.SHUFFLE_TRAIN = True
hparams.DATASET.IMG_RES = 224

# optimizer config
hparams.OPTIMIZER = CN()
hparams.OPTIMIZER.TYPE = 'adam'
hparams.OPTIMIZER.LR = 0.001 # 0.00003
hparams.OPTIMIZER.WD = 0.0

# Training process hparams
hparams.TRAINING = CN()
hparams.TRAINING.RESUME = None
hparams.TRAINING.PRETRAINED = None
hparams.TRAINING.MAX_EPOCHS = 100
hparams.TRAINING.LOG_SAVE_INTERVAL = 50
hparams.TRAINING.LOG_FREQ_TB_IMAGES = 500
hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH = 1
hparams.TRAINING.RELOAD_DATALOADERS_EVERY_EPOCH = True
hparams.TRAINING.SAVE_IMAGES = False

# Neural network hyperparams
hparams.MODEL = CN()
hparams.MODEL.BACKBONE = 'resnet34'
hparams.MODEL.NUM_FC_LAYERS = 1
hparams.MODEL.NUM_FC_CHANNELS = 1024
hparams.MODEL.LOSS_VFOV_WEIGHT = 1.0
hparams.MODEL.LOSS_PITCH_WEIGHT = 1.0
hparams.MODEL.LOSS_ROLL_WEIGHT = 1.0
hparams.MODEL.LOSS_TYPE = 'ce' # 'softargmax_l2'


def get_hparams_defaults():
    """Get a yacs hparamsNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return hparams.clone()


def update_hparams(hparams_file):
    hparams = get_hparams_defaults()
    hparams.merge_from_file(hparams_file)
    return hparams.clone()


def update_hparams_from_dict(cfg_dict):
    hparams = get_hparams_defaults()
    cfg = hparams.load_cfg(str(cfg_dict))
    hparams.merge_from_other_cfg(cfg)
    return hparams.clone()


def run_grid_search_experiments(
        cfg_id,
        cfg_file,
        use_cluster,
        bid,
        memory,
        script='camcalib_train.py',
        cmd_opts=(),
        gpu_min_mem=10000,
        gpu_arch=('tesla', 'quadro', 'rtx'),
        num_gpus=1,
):
    cfg = yaml.load(open(cfg_file))
    # parse config file to get a list of configs and related hyperparameters
    different_configs, hyperparams = get_grid_search_configs(
        cfg,
        excluded_keys=[],
    )
    logger.info(f'Grid search hparams: \n {hyperparams}')

    different_configs = [update_hparams_from_dict(c) for c in different_configs]
    logger.info(f'======> Number of experiment configurations is {len(different_configs)}')

    config_to_run = CN(different_configs[cfg_id])
    config_to_run.merge_from_list(cmd_opts)

    # ==== create logdir using hyperparam settings
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{logtime}_{config_to_run.EXP_NAME}'

    def get_from_dict(dict, keys):
        return reduce(operator.getitem, keys, dict)

    for hp in hyperparams:
        v = get_from_dict(different_configs[cfg_id], hp.split('/'))
        logdir += f'_{hp.replace("/", ".").replace("_", "").lower()}-{v}'

    logdir = os.path.join(config_to_run.LOG_DIR, config_to_run.EXP_NAME, logdir + '_train')

    os.makedirs(logdir, exist_ok=True)
    shutil.copy(src=cfg_file, dst=os.path.join(logdir, 'config.yaml'))

    config_to_run.LOG_DIR = logdir

    def save_dict_to_yaml(obj, filename, mode='w'):
        with open(filename, mode) as f:
            yaml.dump(obj, f, default_flow_style=False)

    # save config
    save_dict_to_yaml(
        unflatten(flatten(config_to_run)),
        os.path.join(config_to_run.LOG_DIR, 'config_to_run.yaml')
    )

    return config_to_run