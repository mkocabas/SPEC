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
sys.path.append('')

from pare.utils.train_utils import load_pretrained_model, resume_training, set_seed, \
    add_init_smpl_params_to_dict

from spec.trainer import SPECTrainer
from spec.config import run_grid_search_experiments
from spec.utils.compute_error import compute_error

torch.multiprocessing.set_sharing_strategy('file_system')


def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(hparams.SEED_VALUE)

    logger.info(torch.cuda.get_device_properties(device))
    hparams.SYSTEM.GPU = torch.cuda.get_device_properties(device).name

    logger.info(f'Hyperparameters: \n {hparams}')

    hparams.DATASET.NUM_WORKERS = 0 # set this to be compatible with other machines
    model = SPECTrainer(hparams=hparams).to(device)

    # TRAINING.PRETRAINED_LIT points to the checkpoint files trained using this repo
    # This has a separate cfg value since in some cases we use checkpoint files from different repos
    if hparams.TRAINING.PRETRAINED_LIT is not None:
        logger.warning(f'Loading pretrained model from {hparams.TRAINING.PRETRAINED_LIT}')
        ckpt = torch.load(hparams.TRAINING.PRETRAINED_LIT)['state_dict']
        load_pretrained_model(model, ckpt, overwrite_shape_mismatch=True)

    if hparams.TRAINING.RESUME is not None:
        resume_ckpt = torch.load(hparams.TRAINING.RESUME)
        if not 'model.head.init_pose' in resume_ckpt['state_dict'].keys():
            logger.info('Adding init SMPL parameters to the resume checkpoint...')
            resume_ckpt = torch.load(hparams.TRAINING.RESUME)
            resume_ckpt['state_dict'] = add_init_smpl_params_to_dict(resume_ckpt['state_dict'])
            torch.save(resume_ckpt, hparams.TRAINING.RESUME)

    amp_params = {}
    if hparams.TRAINING.USE_AMP:
        logger.info(f'Using automatic mixed precision: ampl_level 02, precision 16...')
        amp_params = {
            'amp_level': 'O2',
            # 'amp_backend': 'apex',
            'precision': 16,
        }

    trainer = pl.Trainer(
        gpus=1,
        resume_from_checkpoint=hparams.TRAINING.RESUME,
        **amp_params,
    )

    logger.info('*** Started testing ***')
    trainer.test(model=model)

    for dataset in hparams.DATASET.VAL_DS.split('_'):
        compute_error(os.path.join(hparams.LOG_DIR, f'evaluation_results_{dataset}.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--opts', default=[], nargs='*', help='additional options to update config')
    parser.add_argument('--cfg_id', type=int, default=0, help='cfg id to run when multiple experiments are spawned')
    parser.add_argument('--cluster', default=False, action='store_true', help='creates submission files for cluster')
    parser.add_argument('--resume', default=False, action='store_true', help='resume training from where it left off')
    parser.add_argument('--resume_wo_optimizer', default=False, action='store_true',
                        help='resume training from where it left off but do not use optimizer')
    parser.add_argument('--bid', type=int, default=5, help='amount of bid for cluster')
    parser.add_argument('--memory', type=int, default=64000, help='memory amount for cluster')
    parser.add_argument('--num_cpus', type=int, default=8, help='num cpus for cluster')
    parser.add_argument('--gpu_min_mem', type=int, default=10000, help='minimum amount of GPU memory')
    parser.add_argument('--gpu_arch', default=['tesla', 'quadro', 'rtx'],
                        nargs='*', help='additional options to update config')
    parser.add_argument('--disable_comet', action='store_true')
    parser.add_argument('--fdr', action='store_true')

    args = parser.parse_args()

    logger.info(f'Input arguments: \n {args}')

    if args.resume:
        resume_training(args)

    hparams = run_grid_search_experiments(
        cfg_id=args.cfg_id,
        cfg_file=args.cfg,
        bid=args.bid,
        use_cluster=args.cluster,
        memory=args.memory,
        script='spec_eval.py',
        cmd_opts=args.opts,
        gpu_min_mem=args.gpu_min_mem,
        gpu_arch=args.gpu_arch,
    )

    hparams.RUN_TEST = True

    main(hparams)