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
import sys
import torch
import argparse
from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
sys.path.append('..')

from camcalib.trainer import CameraRegressorModule
from pare.utils.train_utils import load_pretrained_model
from camcalib.config import get_hparams_defaults, run_grid_search_experiments


def train(hparams, fast_dev_run=False):
    log_dir = hparams.LOG_DIR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_gpus = torch.cuda.device_count()
    hparams.DATASET.BATCH_SIZE *= num_gpus
    # hparams.DATASET.NUM_WORKERS *= num_gpus

    logger.add(
        os.path.join(log_dir, 'train.log'),
        level='INFO',
        colorize=False,
    )

    logger.info(torch.cuda.get_device_properties(device))
    hparams.SYSTEM.GPU = torch.cuda.get_device_properties(device).name

    logger.info(f'Hyperparameters: \n {hparams}')

    # initialize tensorboard logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name='tb_logs',
        log_graph=False,
    )

    ckpt_callback = ModelCheckpoint(
        monitor='val/loss',
        verbose=True,
        save_top_k=30,
        mode='min',
        # save_on_train_epoch_end=True,
    )

    pl_module = CameraRegressorModule(hparams)

    if hparams.TRAINING.PRETRAINED is not None:
        logger.warning(f'Loading pretrained model from {hparams.TRAINING.PRETRAINED}')
        ckpt = torch.load(hparams.TRAINING.PRETRAINED)['state_dict']
        load_pretrained_model(pl_module, ckpt, overwrite_shape_mismatch=True)

    trainer = pl.Trainer(
        gpus=-1,
        accelerator='dp',
        logger=[tb_logger], #, comet_logger],
        max_epochs=hparams.TRAINING.MAX_EPOCHS,
        callbacks=[ckpt_callback],
        log_every_n_steps=50,
        terminate_on_nan=True,
        default_root_dir=log_dir,
        progress_bar_refresh_rate=50,
        check_val_every_n_epoch=hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH,
        # checkpoint_callback=ckpt_callback,
        reload_dataloaders_every_epoch=hparams.TRAINING.RELOAD_DATALOADERS_EVERY_EPOCH,
        resume_from_checkpoint=hparams.TRAINING.RESUME,
        num_sanity_val_steps=0,
        fast_dev_run=args.fdr,
    )

    if hparams.RUN_TEST:
        logger.info(f'Running evaluation on {hparams.DATASET.VAL_DS}')
        trainer.test(pl_module)
        exit()

    logger.info('*** Started training ***')
    trainer.fit(pl_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--opts', default=[], nargs='*', help='additional options to update config')
    parser.add_argument('--cfg_id', type=int, default=0, help='cfg id to run when multiple experiments are spawned')
    parser.add_argument('--cluster', default=False, action='store_true', help='creates submission files for cluster')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training from where it left off')
    parser.add_argument('--bid', type=int, default=5, help='amount of bid for cluster')
    parser.add_argument('--memory', type=int, default=64000, help='memory amount for cluster')
    parser.add_argument('--num_cpus', type=int, default=4, help='num cpus for cluster')
    parser.add_argument('--num_gpus', type=int, default=1, help='num cpus for cluster')
    parser.add_argument('--gpu_min_mem', type=int, default=10000, help='minimum amount of GPU memory')
    parser.add_argument('--gpu_arch', default=['tesla', 'quadro', 'rtx'],
                        nargs='*', help='additional options to update config')
    parser.add_argument('--fdr', action='store_true', help='fast dev run')
    # parser.add_argument()

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    hparams = run_grid_search_experiments(
        cfg_id=args.cfg_id,
        cfg_file=args.cfg,
        bid=args.bid,
        use_cluster=args.cluster,
        memory=args.memory,
        script='camcalib_train.py',
        cmd_opts=args.opts,
        gpu_min_mem=args.gpu_min_mem,
        gpu_arch=args.gpu_arch,
        num_gpus=args.num_gpus,
    )

    train(hparams, fast_dev_run=args.fdr)