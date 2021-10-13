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
import json
import torch
import joblib
import numpy as np
from loguru import logger
import pytorch_lightning as pl
from smplx import SMPL as SMPL_native
from torch.utils.data import DataLoader

from .dataset import CamDataset, MixedCamDataset
from pare.utils.train_utils import set_seed
from pare.utils.eval_utils import reconstruction_error, compute_error_verts
from pare.utils.geometry import batch_euler2matrix
from pare.utils.image_utils import denormalize_images
from pare.utils.image_utils import read_img

from . import config
from . import constants
from pare.models import SMPL
from .utils.renderer_cam import render_image_group


class SPECTrainer(pl.LightningModule):

    def __init__(self, hparams):
        super(SPECTrainer, self).__init__()

        self.hparams.update(hparams)

        if self.hparams.METHOD == 'hmr_cam':
            from .models import HMR
            from .losses import HMRCamLoss
            self.model = HMR(
                backbone=self.hparams.HMR.BACKBONE,
                img_res=self.hparams.DATASET.IMG_RES,
                pretrained=self.hparams.TRAINING.PRETRAINED,
                use_cam_feats=self.hparams.HMR.USE_CAM_FEATS,
                use_cam=True,
            )
            self.loss_fn = HMRCamLoss(
                shape_loss_weight=self.hparams.HMR.SHAPE_LOSS_WEIGHT,
                keypoint_loss_weight=self.hparams.HMR.KEYPOINT_LOSS_WEIGHT,
                pose_loss_weight=self.hparams.HMR.POSE_LOSS_WEIGHT,
                beta_loss_weight=self.hparams.HMR.BETA_LOSS_WEIGHT,
                openpose_train_weight=self.hparams.HMR.OPENPOSE_TRAIN_WEIGHT,
                gt_train_weight=self.hparams.HMR.GT_TRAIN_WEIGHT,
                loss_weight=self.hparams.HMR.LOSS_WEIGHT,
                smpl_part_loss_weight=self.hparams.HMR.SMPL_PART_LOSS_WEIGHT,
            )
        else:
            logger.error(f'{self.hparams.METHOD} is undefined!')
            exit()

        self.smpl = SMPL(
            config.SMPL_MODEL_DIR,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            create_transl=False
        )
        self.add_module('smpl', self.smpl)

        # smpl_native regresses joint regressor with 24 smpl kinematic tree joints
        # It is used during training of PARE part branch to obtain 2d gt/predicted keypoints
        # in original SMPL coordinates
        self.smpl_native = SMPL_native(
            config.SMPL_MODEL_DIR,
            # batch_size=self.hparams.DATASET.BATCH_SIZE,
            create_transl=False
        )
        self.add_module('smpl_native', self.smpl_native)

        # Initialize the training datasets only in training mode
        if not hparams.RUN_TEST:
            self.train_ds = self.train_dataset()

        self.val_ds = self.val_dataset()

        # self.example_input_array = torch.rand(1, 3, self.hparams.DATASET.IMG_RES, self.hparams.DATASET.IMG_RES)

        self.register_buffer(
            'J_regressor',
            torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        )

        if len(self.val_ds) > 0:
            self.val_accuracy_results = {ds.dataset:[] for ds in self.val_ds}
        else:
            self.val_accuracy_results = []

        # Initialiatize variables required for evaluation
        self.init_evaluation_variables()

    def init_evaluation_variables(self):
        # stores mean mpjpe/pa-mpjpe values for all validation dataset samples
        self.val_mpjpe = [] # np.zeros(len(self.val_ds))
        self.val_pampjpe = [] # np.zeros(len(self.val_ds))
        self.val_mpjpe_24 = []
        self.val_pampjpe_24 = []
        self.val_v2v = []

        # This dict is used to store metrics and metadata for a more detailed analysis
        # per-joint, per-sequence, occluded-sequences etc.
        self.evaluation_results = {
            'imgname': [],
            'dataset_name': [],
            'mpjpe': [], # np.zeros((len(self.val_ds), 14)),
            'pampjpe': [], # np.zeros((len(self.val_ds), 14)),
            'mpjpe_24': [],
            'pampjpe_24': [],
        }

        # use this to save the errors for each image
        if self.hparams.TESTING.SAVE_IMAGES:
            self.val_images_errors = []

        if self.hparams.TESTING.SAVE_RESULTS:
            self.evaluation_results['pose'] = []
            self.evaluation_results['shape'] = []
            self.evaluation_results['cam'] = []
            self.evaluation_results['vertices'] = []

    def forward(self, x, cam_rotmat, cam_intrinsics, bbox_scale, bbox_center, img_w, img_h):
        return self.model(x, cam_rotmat, cam_intrinsics, bbox_scale, bbox_center, img_w, img_h)

    def training_step(self, batch, batch_nb):
        # Get data from the batch
        images = batch['img']  # input image
        gt_pose = batch['pose']  # SMPL pose parameters
        gt_betas = batch['betas']  # SMPL beta parameters

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(
            betas=gt_betas,
            body_pose=gt_pose[:, 3:],
            global_orient=gt_pose[:, :3]
        )
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        cam_rotmat = batch['cam_rotmat']
        cam_intrinsics = batch['cam_int']
        bbox_scale = batch['scale']
        bbox_center = batch['center']
        img_h = batch['orig_shape'][:,0]
        img_w = batch['orig_shape'][:,1]

        pred = self(images, cam_rotmat, cam_intrinsics, bbox_scale, bbox_center, img_w, img_h)

        batch['vertices'] = gt_vertices

        loss, loss_dict = self.loss_fn(pred=pred, gt=batch)
        tensorboard_logs = loss_dict

        self.log_dict(tensorboard_logs)

        if batch_nb % self.hparams.TRAINING.LOG_FREQ_TB_IMAGES == 0:
            self.train_summaries(input_batch=batch, output=pred)

        return {'loss': loss, 'log': tensorboard_logs}

    def train_summaries(self, input_batch, output):
        images = input_batch['img']
        images = denormalize_images(images)

        pred_vertices = output['smpl_vertices'].detach()

        pred_cam_t = output['pred_cam_t'].detach()

        save_dir = os.path.join(self.hparams.LOG_DIR, 'training_images')
        os.makedirs(save_dir, exist_ok=True)

        cam_pitch = -input_batch['cam_pitch']

        cam_roll = torch.zeros_like(cam_pitch)
        if 'cam_roll' in input_batch.keys():
            cam_roll = input_batch['cam_roll']

        render_rotmat = batch_euler2matrix(
            torch.stack([cam_pitch, torch.zeros_like(cam_pitch), cam_roll], dim=-1)
        )

        max_save_img = 1

        for i in range(images.shape[0]):
            imgname = input_batch['imgname'][i]
            focal_length = (input_batch['focal_length'][i, 0], input_batch['focal_length'][i, 1])

            cy, cx = input_batch['orig_shape'][i] // 2

            save_filename = None
            if self.hparams.TRAINING.SAVE_IMAGES:
                save_filename = os.path.join(save_dir, f'{self.global_step:08d}_{i:02d}_{os.path.basename(imgname)}')

            rendered_img = render_image_group(
                image=read_img(imgname),
                camera_translation=pred_cam_t[i],
                vertices=pred_vertices[i],
                camera_rotation=render_rotmat[i],
                focal_length=focal_length,
                camera_center=(cx, cy),
                save_filename=save_filename,
                keypoints_2d=input_batch['keypoints_orig'][i].cpu().numpy(),
            )

            # DEBUG
            # import matplotlib.pyplot as plt
            # plt.imshow(rendered_img)
            # plt.show()

            if i >= (max_save_img - 1):
                break

    def validation_step(self, batch, batch_nb, dataloader_nb, vis=False, save=True, mesh_save_dir=None):
        images = batch['img']
        imgnames = batch['imgname']
        dataset_names = batch['dataset_name']

        cam_rotmat = batch['cam_rotmat'] if self.hparams.TESTING.USE_GT_CAM else batch['pred_cam_rotmat']
        cam_intrinsics = batch['cam_int'] if self.hparams.TESTING.USE_GT_CAM else batch['pred_cam_int']
        bbox_scale = batch['scale']
        bbox_center = batch['center']
        img_h = batch['orig_shape'][:, 0]
        img_w = batch['orig_shape'][:, 1]

        curr_batch_size = images.shape[0]

        with torch.no_grad():
            pred = self(images, cam_rotmat, cam_intrinsics, bbox_scale, bbox_center, img_w, img_h)
            pred_vertices = pred['smpl_vertices']

            try:
                pred_joints_24 = self.smpl_native(
                    shape=pred['pred_shape'],
                    body_pose=pred['pred_pose'][:, 1:].contiguous(),
                    global_orient=pred['pred_pose'][:, 0].unsqueeze(1).contiguous(),
                    pose2rot=False,
                ).joints[:, :24]
            except Exception as e:
                print(e)
                import IPython; IPython.embed(); exit()

        joint_mapper_h36m = constants.H36M_TO_J17 if dataset_names[0] == 'mpi-inf-3dhp' \
            else constants.H36M_TO_J14

        if dataset_names[0] in ['mpii', 'coco']:
            # Only for qualitative result experiments
            if self.hparams.TESTING.SAVE_IMAGES:
                self.validation_summaries(batch, pred, batch_nb, dataloader_nb)
                error, r_error = torch.zeros(1), torch.zeros(1)
                error_per_joint, r_error_per_joint = torch.zeros(14), torch.zeros(14)
            else:
                logger.error('Set `TESTING.SAVE_IMAGES` to `True` when using ITW datasets the evaluation dataset')
                exit()
        else:
            J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)

            gt_keypoints_3d = batch['pose_3d'].cuda()
            gt_joints_24 = batch['joints_24'].cuda()
            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            pred_pelvis_j_24 = pred_joints_24[:, [0], :].clone()
            pred_joints_24 = pred_joints_24 - pred_pelvis_j_24

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            idx_start = batch_nb * self.hparams.DATASET.BATCH_SIZE
            idx_stop = batch_nb * self.hparams.DATASET.BATCH_SIZE + curr_batch_size

            # Reconstuction_error
            r_error, r_error_per_joint = reconstruction_error(
                pred_keypoints_3d.cpu().numpy(),
                gt_keypoints_3d.cpu().numpy(),
                reduction=None,
            )
            # import IPython; IPython.embed(); exit()
            error_j_24 = torch.sqrt(((pred_joints_24 - gt_joints_24) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            r_error_j_24, r_error_per_joint_j_24 = reconstruction_error(
                pred_joints_24.cpu().numpy(),
                gt_joints_24.cpu().numpy(),
                reduction=None,
            )

            # Per-vertex error
            if 'vertices' in batch.keys():
                gt_vertices = batch['vertices'].cuda()

                # logger.debug(f'GT vertices shape: {gt_vertices.shape}')
                # logger.debug(f'PR vertices shape: {pred_vertices.shape}')
                # logger.debug(f'ARRAY: {gt_vertices}')

                v2v = compute_error_verts(
                    pred_verts=pred_vertices.cpu().numpy(),
                    target_verts=gt_vertices.cpu().numpy(),
                )
                self.val_v2v += v2v.tolist()
            else:
                self.val_v2v += np.zeros_like(error).tolist()

            ####### DEBUG 3D JOINT PREDICTIONS and GT ###########
            # from ..utils.vis_utils import show_3d_pose
            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(12, 7))
            # plt.title(f'error {error[0].item()*1000:.2f}, r_err {r_error[0].item()*1000:.2f}')
            # ax = fig.add_subplot('121', projection='3d', aspect='auto')
            # show_3d_pose(kp_3d=pred_joints_24[0].cpu(), ax=ax, dataset='smpl')
            #
            # ax = fig.add_subplot('122', projection='3d', aspect='auto')
            # show_3d_pose(kp_3d=gt_joints_24[0].cpu(), ax=ax, dataset='smpl')
            # plt.show()
            #####################################################

            self.val_mpjpe += error.tolist()
            self.val_pampjpe += r_error.tolist()
            self.val_mpjpe_24 += error_j_24.tolist()
            self.val_pampjpe_24 += r_error_j_24.tolist()

            error_per_joint = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).cpu().numpy()
            error_per_joint_24 = torch.sqrt(((pred_joints_24 - gt_joints_24) ** 2).sum(dim=-1)).cpu().numpy()

            self.evaluation_results['mpjpe'] += error_per_joint[:,:14].tolist()
            self.evaluation_results['pampjpe'] += r_error_per_joint[:,:14].tolist()
            self.evaluation_results['mpjpe_24'] += error_per_joint_24.tolist()
            self.evaluation_results['pampjpe_24'] += r_error_per_joint_j_24.tolist()
            self.evaluation_results['imgname'] += imgnames
            self.evaluation_results['dataset_name'] += dataset_names

            if self.hparams.TESTING.SAVE_RESULTS:
                tolist = lambda x: [i for i in x.cpu().numpy()]
                self.evaluation_results['pose'] += tolist(pred['pred_pose'])
                self.evaluation_results['shape'] += tolist(pred['pred_shape'])
                self.evaluation_results['cam'] += tolist(pred['pred_cam'])
                self.evaluation_results['vertices'] += tolist(pred_vertices)

            if self.hparams.TESTING.SAVE_IMAGES and batch_nb % self.hparams.TESTING.SAVE_FREQ == 0:
                # this saves the rendered images
                self.validation_summaries(batch, pred, batch_nb, dataloader_nb)

        return {
            'mpjpe': error.mean(),
            'pampjpe': r_error.mean(),
            'per_mpjpe': error_per_joint,
            'per_pampjpe': r_error_per_joint
        }

    def validation_summaries(self, input_batch, output, batch_idx, dataloader_nb):
        # images = input_batch['img']
        images = input_batch['disp_img']
        images = denormalize_images(images)

        pred_vertices = output['smpl_vertices'].detach()
        pred_cam_t = output['pred_cam_t'].detach()
        pred_kp_2d = output['pred_kp2d'].detach() if 'pred_kp2d' in output.keys() else None

        mesh_filename = None

        cam_pitch = -input_batch['pred_cam_pitch']
        cam_roll = input_batch['pred_cam_roll']

        render_rotmat = batch_euler2matrix(
            torch.stack([cam_pitch, torch.zeros_like(cam_pitch), cam_roll], dim=-1)
        )

        save_dir = os.path.join(self.hparams.LOG_DIR, 'output_images')
        os.makedirs(save_dir, exist_ok=True)

        max_save_img = 1

        for i in range(images.shape[0]):

            imgname = input_batch['imgname'][i]
            focal_length = (input_batch['pred_cam_focal_length'][i], input_batch['pred_cam_focal_length'][i])

            cy, cx = input_batch['orig_shape'][i] // 2

            cam_params = torch.stack([
                input_batch['pred_cam_vfov'][i], input_batch['pred_cam_pitch'][i],
                input_batch['pred_cam_roll'][i], input_batch['pred_cam_focal_length'][i]]
            ).cpu().numpy()

            save_filename = None
            if self.hparams.TRAINING.SAVE_IMAGES:
                # save_filename = os.path.join(save_dir, f'result_{self.current_epoch:04d}_'
                #                                        f'{dataloader_nb:02d}_{batch_idx:05d}_{i:02d}.jpg')

                save_filename = os.path.join(save_dir, f'{self.current_epoch:04d}_{dataloader_nb:02d}_'
                                                       f'{batch_idx:05d}_{i:02d}_{os.path.basename(imgname)}')

            render_image_group(
                image=read_img(imgname),
                camera_translation=pred_cam_t[i],
                vertices=pred_vertices[i],
                camera_rotation=render_rotmat[i],
                focal_length=focal_length,
                camera_center=(cx, cy),
                save_filename=save_filename,
                mesh_filename=mesh_filename,
                keypoints_2d=input_batch['keypoints_orig'][i].cpu().numpy(),
                cam_params=cam_params,
            )

            if i >= (max_save_img - 1):
                break

    def validation_epoch_end(self, outputs):
        if 'coco' in self.val_ds or 'mpii' in self.val_ds:
            logger.info('...THE END...')
            exit()

        self.val_mpjpe = np.array(self.val_mpjpe)
        self.val_pampjpe = np.array(self.val_pampjpe)
        self.val_mpjpe_24 = np.array(self.val_mpjpe_24)
        self.val_pampjpe_24 = np.array(self.val_pampjpe_24)
        self.val_v2v = np.array(self.val_v2v)

        for k,v in self.evaluation_results.items():
            self.evaluation_results[k] = np.array(v)

        if len(self.val_ds) == 1:
            avg_mpjpe, avg_pampjpe = 1000 * self.val_mpjpe.mean(), 1000 * self.val_pampjpe.mean()
            avg_mpjpe_24, avg_pampjpe_24 = 1000 * self.val_mpjpe_24.mean(), 1000 * self.val_pampjpe_24.mean()
            avg_v2v = 1000 * self.val_v2v.mean()

            logger.info(f'***** Epoch {self.current_epoch} *****')
            logger.info('MPJPE: ' + str(avg_mpjpe))
            logger.info('PA-MPJPE: ' + str(avg_pampjpe))
            logger.info('MPJPE (24j): ' + str(avg_mpjpe_24))
            logger.info('PA-MPJPE  (24j): ' + str(avg_pampjpe_24))
            logger.info('V2V (mm): ' + str(avg_v2v))

            acc = {
                'val_mpjpe': avg_mpjpe.item(),
                'val_pampjpe': avg_pampjpe.item(),
                'val_mpjpe_24': avg_mpjpe_24.item(),
                'val_pampjpe_24': avg_pampjpe_24.item(),
                'val_v2v': avg_v2v.item(),
            }

            self.val_save_best_results(acc)

            # save the mpjpe and pa-mpjpe results per image
            if self.hparams.TESTING.SAVE_IMAGES and len(self.val_images_errors) > 0:
                save_path = os.path.join(self.hparams.LOG_DIR, 'val_images_error.npy')
                logger.info(f'Saving the errors of images {save_path}')
                np.save(save_path, np.asarray(self.val_images_errors))

            # save the detailed experiment results for post-analysis script
            # use these with scripts/analyze_per_joint_per_seq.py
            joblib.dump(
                self.evaluation_results,
                os.path.join(self.hparams.LOG_DIR, f'evaluation_results_{self.hparams.DATASET.VAL_DS}.pkl')
            )

            avg_mpjpe, avg_pampjpe = torch.tensor(avg_mpjpe), torch.tensor(avg_pampjpe)
            tensorboard_logs = {
                'val/val_mpjpe': avg_mpjpe,
                'val/val_pampjpe': avg_pampjpe,
            }
            val_log = {
                'val_loss': avg_pampjpe,
                'val_mpjpe': avg_mpjpe,
                'val_pampjpe': avg_pampjpe,
                'log': tensorboard_logs
            }
        else:
            logger.info(f'***** Epoch {self.current_epoch} *****')
            val_log = {}
            val_log['log'] = {}

            for ds_idx, ds in enumerate(self.val_ds):
                ds_name = ds.dataset
                idxs = self.evaluation_results['dataset_name'] == ds_name

                mpjpe = 1000 * self.val_mpjpe[idxs].mean()
                pampjpe = 1000 * self.val_pampjpe[idxs].mean()
                mpjpe_24 = 1000 * self.val_mpjpe_24[idxs].mean()
                pampjpe_24 = 1000 * self.val_pampjpe_24[idxs].mean()
                v2v = 1000 * self.val_v2v[idxs].mean()

                logger.info(f'{ds_name} MPJPE: ' + str(mpjpe))
                logger.info(f'{ds_name} PA-MPJPE: ' + str(pampjpe))
                logger.info(f'{ds_name} MPJPE (24j): ' + str(mpjpe_24))
                logger.info(f'{ds_name} PA-MPJPE (24j): ' + str(pampjpe_24))
                logger.info(f'{ds_name} V2V: ' + str(v2v))

                acc = {
                    'val_mpjpe': mpjpe.item(),
                    'val_pampjpe': pampjpe.item(),
                    'val_mpjpe_24': mpjpe_24.item(),
                    'val_pampjpe_24': pampjpe_24.item(),
                    'val_v2v': v2v.item(),
                }

                val_log[f'val_mpjpe_{ds_name}'] = mpjpe
                val_log[f'val_pampjpe_{ds_name}'] = pampjpe
                val_log[f'val_mpjpe_24_{ds_name}'] = mpjpe_24
                val_log[f'val_pampjpe_24_{ds_name}'] = pampjpe_24

                val_log['log'][f'val/val_mpjpe_{ds_name}'] = mpjpe
                val_log['log'][f'val/val_pampjpe_{ds_name}'] = pampjpe
                val_log['log'][f'val_mpjpe_24_{ds_name}'] = mpjpe_24
                val_log['log'][f'val_pampjpe_24_{ds_name}'] = pampjpe_24

                self.val_save_best_results(acc, ds_name)

                # save the mpjpe and pa-mpjpe results per image
                if self.hparams.TESTING.SAVE_IMAGES and len(self.val_images_errors) > 0:
                    save_path = os.path.join(self.hparams.LOG_DIR, 'val_images_error.npy')
                    logger.info(f'Saving the errors of images {save_path}')
                    np.save(save_path, np.asarray(self.val_images_errors))

                eval_res = {k: v[idxs] for k,v in self.evaluation_results.items()}
                joblib.dump(
                    eval_res,
                    os.path.join(self.hparams.LOG_DIR, f'evaluation_results_{ds_name}.pkl')
                )

                # always set the first dataset as the main one
                if ds_idx == 0:
                    avg_mpjpe, avg_pampjpe = mpjpe, pampjpe
                    val_log['val_loss'] = avg_pampjpe
                    val_log['val_mpjpe'] = avg_mpjpe
                    val_log['val_pampjpe'] = avg_pampjpe

                    val_log['log'][f'val/val_mpjpe'] = avg_mpjpe
                    val_log['log'][f'val/val_pampjpe'] = avg_pampjpe

        for k, v in val_log.items():
            if k == 'log':
                pass
            else:
                self.log(k, v)

        # reset evaluation variables
        self.init_evaluation_variables()
        return val_log

    def test_step(self, batch, batch_nb, dataloader_nb):
        return self.validation_step(batch, batch_nb, dataloader_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.OPTIMIZER.LR,
            weight_decay=self.hparams.OPTIMIZER.WD
        )

    def train_dataset(self):

        if self.hparams.DATASET.TEACHER_FORCE_SCHEDULE != '':
            tf_sched = self.hparams.DATASET.TEACHER_FORCE_SCHEDULE.split(' ')
            tf_dict = {x.split('+')[0]: x.split('+')[1] for x in tf_sched}
            logger.info('TEACHER_FORCE: ', tf_dict)
            if str(self.current_epoch) in tf_dict.keys():
                self.hparams.DATASET.TEACHER_FORCE = float(tf_dict[str(self.current_epoch)])
                logger.debug(f'Updated teacher force to: {self.hparams.DATASET.TEACHER_FORCE}')

        if self.hparams.DATASET.TRAIN_DS == 'all':
            train_ds = MixedCamDataset(
                options=self.hparams.DATASET,
                ignore_3d=self.hparams.DATASET.IGNORE_3D,
                is_train=True
            )
        elif self.hparams.DATASET.TRAIN_DS == 'stage':
            # stage dataset is used to
            stage_datasets = self.hparams.DATASET.STAGE_DATASETS.split(' ')
            stage_dict = {x.split('+')[0]: x.split('+')[1] for x in stage_datasets}
            assert self.hparams.DATASET.STAGE_DATASETS.startswith('0'), 'Stage datasets should start from epoch 0'

            if str(self.current_epoch) in stage_dict.keys():
                self.hparams.DATASET.DATASETS_AND_RATIOS = stage_dict[str(self.current_epoch)]

            train_ds = MixedCamDataset(
                options=self.hparams.DATASET,
                ignore_3d=self.hparams.DATASET.IGNORE_3D,
                is_train=True
            )
        else:
            train_ds = CamDataset(
                options=self.hparams.DATASET,
                dataset=self.hparams.DATASET.TRAIN_DS,
                ignore_3d=self.hparams.DATASET.IGNORE_3D,
                is_train=True,
            )

        return train_ds

    def train_dataloader(self):
        set_seed(self.hparams.SEED_VALUE)

        self.train_ds = self.train_dataset()

        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
        )

    def val_dataset(self):
        datasets = self.hparams.DATASET.VAL_DS.split('_')
        logger.info(f'Validation datasets are: {datasets}')
        val_datasets = []
        for dataset_name in datasets:
            val_datasets.append(
                CamDataset(
                    options=self.hparams.DATASET,
                    dataset=dataset_name,
                    is_train=False,
                )
            )

        return val_datasets

    def val_dataloader(self):
        dataloaders = []
        for val_ds in self.val_ds:
            dataloaders.append(
                DataLoader(
                    dataset=val_ds,
                    batch_size=self.hparams.DATASET.BATCH_SIZE,
                    shuffle=False,
                    num_workers=self.hparams.DATASET.NUM_WORKERS,
                )
            )
        return dataloaders

    def test_dataloader(self):
        return self.val_dataloader()

    def val_save_best_results(self, acc, ds_name=None):
        # log the running training metrics
        if ds_name:
            fname = f'val_accuracy_results_{ds_name}.json'
            json_file = os.path.join(self.hparams.LOG_DIR, fname)
            self.val_accuracy_results[ds_name].append([self.global_step, acc, self.current_epoch])
            with open(json_file, 'w') as f:
                json.dump(self.val_accuracy_results[ds_name], f, indent=4)
        else:
            fname = 'val_accuracy_results.json'
            json_file = os.path.join(self.hparams.LOG_DIR, fname)
            self.val_accuracy_results.append([self.global_step, acc, self.current_epoch])
            with open(json_file, 'w') as f:
                json.dump(self.val_accuracy_results, f, indent=4)
