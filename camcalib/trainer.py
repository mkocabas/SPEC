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
import cv2
import torch
import numpy as np
from loguru import logger
from imageio import imsave
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .loss import CameraRegressorLoss
from .vis_utils import show_horizon_line
from .model import CameraRegressorNetwork
from .pano_dataset import CameraRegressorDataset, collator
from .pano_agora_dataset import PanoAgoraDataset
from .cam_utils import convert_preds_to_angles

from pare.utils.image_utils import denormalize_images


class CameraRegressorModule(pl.LightningModule):

    def __init__(self, hparams):
        super(CameraRegressorModule, self).__init__()

        self.model = CameraRegressorNetwork(
            backbone=hparams.MODEL.BACKBONE,
            num_fc_layers=hparams.MODEL.NUM_FC_LAYERS,
            num_fc_channels=hparams.MODEL.NUM_FC_CHANNELS,
        )

        self.loss_fn = CameraRegressorLoss(
            vfov_loss_weight=hparams.MODEL.LOSS_VFOV_WEIGHT,
            pitch_loss_weight=hparams.MODEL.LOSS_PITCH_WEIGHT,
            roll_loss_weight=hparams.MODEL.LOSS_ROLL_WEIGHT,
            loss_type=hparams.MODEL.LOSS_TYPE,
        )
        self.hparams.update(hparams)

        self.example_input_array = torch.rand(1, 3, self.hparams.DATASET.IMG_RES, self.hparams.DATASET.IMG_RES)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        images = batch['img']
        gt_vfov = batch['vfov_bin']
        gt_pitch = batch['pitch_bin']
        gt_roll = batch['roll_bin']

        pred_vfov, pred_pitch, pred_roll = self(images)

        loss, loss_dict = self.loss_fn(pred_vfov, pred_pitch, pred_roll, gt_vfov, gt_pitch, gt_roll)

        # import IPython; IPython.embed(); exit()

        if batch_nb % 1000 == 0:
            self.save_images(batch, preds=(pred_vfov, pred_pitch, pred_roll), step='train')

        new_loss_dict = {}
        for k, v in loss_dict.items():
            new_loss_dict[f'train/{k}'] = v
        loss_dict = new_loss_dict

        # self.log_dict(loss_dict)

        return {'loss': loss, 'log': loss_dict}

    def validation_step(self, batch, batch_nb):
        images = batch['img']
        gt_vfov = batch['vfov_bin']
        gt_pitch = batch['pitch_bin']
        gt_roll = batch['roll_bin']

        pred_vfov, pred_pitch, pred_roll = self(images)

        loss, loss_dict = self.loss_fn(pred_vfov, pred_pitch, pred_roll, gt_vfov, gt_pitch, gt_roll)

        if batch_nb % 4 == 0:
            self.save_images(batch, preds=(pred_vfov, pred_pitch, pred_roll), step='val', batch_nb=batch_nb)

        new_loss_dict = {}
        for k,v in loss_dict.items():
            new_loss_dict[f'val/{k}'] = v
        loss_dict = new_loss_dict

        self.log_dict(loss_dict)

        pred_vfov, pred_pitch, pred_roll = convert_preds_to_angles(
            pred_vfov, pred_pitch, pred_roll, loss_type=self.hparams.MODEL.LOSS_TYPE,
        )

        if isinstance(pred_roll, np.ndarray):
            pred_roll = torch.from_numpy(pred_roll)

        vfov_acc = torch.abs(pred_vfov.to(batch['vfov'].device) - batch['vfov']).mean().rad2deg()
        pitch_acc = torch.abs(pred_pitch.to(batch['pitch'].device) - batch['pitch']).mean().rad2deg()
        roll_acc = torch.abs(pred_roll.to(batch['roll'].device) - batch['roll']).mean().rad2deg()

        return {'loss': loss, 'log': loss_dict, 'vfov_acc': vfov_acc,
                'pitch_acc': pitch_acc, 'roll_acc': roll_acc}

    def save_images(self, batch, preds, step='val', batch_nb=0):
        batch_img = batch['img']
        batch_img = denormalize_images(batch_img) * 255
        batch_img = np.transpose(batch_img.cpu().numpy(), (0, 2, 3, 1))

        gt_vfov = batch['vfov']
        gt_pitch = batch['pitch']
        gt_roll = batch['roll']

        idx = 0
        extract = lambda x: x[idx].detach().cpu().numpy()
        img = batch_img[idx].copy()

        orig_img_size = batch['img_sizes'][idx]

        img = img[:orig_img_size[0], :orig_img_size[1]]

        if self.hparams.MODEL.LOSS_TYPE in ('kl', 'ce'):
            pred_vfov, pred_pitch, pred_roll = map(extract, preds)
            pred_vfov, pred_pitch, pred_roll = convert_preds_to_angles(
                pred_vfov, pred_pitch, pred_roll, loss_type=self.hparams.MODEL.LOSS_TYPE,
                return_type='np',
            )
        else:
            preds = convert_preds_to_angles(
                *preds, loss_type=self.hparams.MODEL.LOSS_TYPE, return_type='torch'
            )
            pred_vfov = extract(preds[0])
            pred_pitch = extract(preds[1])
            pred_roll = extract(preds[2]) # [idx]

        gt_vfov, gt_pitch, gt_roll = map(extract, (gt_vfov, gt_pitch, gt_roll))

        # pred_vfov = bins2vfov(pred_vfov)
        # pred_pitch = bins2pitch(pred_pitch)
        # pred_roll = bins2roll(pred_roll)

        pred_f_pix = orig_img_size[0] / 2. / np.tan(pred_vfov / 2.)
        gt_f_pix = orig_img_size[0] / 2. / np.tan(gt_vfov / 2.)

        img, _ = show_horizon_line(img.copy(), gt_vfov, gt_pitch, gt_roll, focal_length=gt_f_pix,
                                   debug=True, color=(0, 0, 255), width=3, GT=True)
        img, _ = show_horizon_line(img.copy(), pred_vfov, pred_pitch, pred_roll, focal_length=pred_f_pix,
                                   debug=True, color=(255, 0, 0), width=3, GT=False)

        save_dir = os.path.join(self.hparams.LOG_DIR, f'{step}_output_images')
        os.makedirs(save_dir, exist_ok=True)

        imsave(
            os.path.join(save_dir, f'result_{self.global_step:07d}_{batch_nb:05d}.jpg'),
            img
        )

    def get_angular_distance(self, pred_vfov, pred_pitch, pred_roll, gt_vfov, gt_pitch, gt_roll):
        # np.abs()

        raise NotImplementedError

    def validation_epoch_end(self, outputs):
        # import IPython; IPython.embed(); exit()

        val_losses = torch.tensor([x['loss'].mean() for x in outputs])
        vfov_acc = torch.tensor([x['vfov_acc'].mean() for x in outputs])
        pitch_acc = torch.tensor([x['pitch_acc'].mean() for x in outputs])
        roll_acc = torch.tensor([x['roll_acc'].mean() for x in outputs])

        val_log = {
            'val_loss': val_losses.mean()
        }

        logger.info(f'[EPOCH {self.current_epoch}] Val loss reached {val_losses.mean().item()}')
        logger.info(f'[EPOCH {self.current_epoch}] vfov acc: {vfov_acc.mean().item()}')
        logger.info(f'[EPOCH {self.current_epoch}] pitch acc: {pitch_acc.mean().item()}')
        logger.info(f'[EPOCH {self.current_epoch}] roll acc: {roll_acc.mean().item()}')

        return val_log

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.OPTIMIZER.LR,
            weight_decay=self.hparams.OPTIMIZER.WD
        )

    def train_dataloader(self):
        if self.hparams.DATASET.TRAIN_DS == 'pano_agora':
            train_dataset = PanoAgoraDataset(
                is_train=True,
                loss_type=self.hparams.MODEL.LOSS_TYPE,
                min_size=self.hparams.DATASET.MIN_RES,
                max_size=self.hparams.DATASET.MAX_RES,
            )
        else:
            train_dataset = CameraRegressorDataset(
                dataset=self.hparams.DATASET.TRAIN_DS,
                is_train=True,
                loss_type=self.hparams.MODEL.LOSS_TYPE,
                min_size=self.hparams.DATASET.MIN_RES,
                max_size=self.hparams.DATASET.MAX_RES,
            )

        logger.info(f'Train dataset len: {len(train_dataset)}')

        return DataLoader(
            dataset=train_dataset,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=self.hparams.DATASET.SHUFFLE_TRAIN,
            collate_fn=collator,
        )

    def val_dataloader(self):
        if self.hparams.DATASET.TRAIN_DS == 'pano_agora':
            val_dataset = PanoAgoraDataset(
                is_train=False,
                loss_type=self.hparams.MODEL.LOSS_TYPE,
                min_size=self.hparams.DATASET.MIN_RES,
                max_size=self.hparams.DATASET.MAX_RES,
            )
        else:
            val_dataset = CameraRegressorDataset(
                dataset=self.hparams.DATASET.VAL_DS,
                is_train=False,
                loss_type=self.hparams.MODEL.LOSS_TYPE,
                min_size=self.hparams.DATASET.MIN_RES,
                max_size=self.hparams.DATASET.MAX_RES,
            )

        logger.info(f'Val dataset len: {len(val_dataset)}')

        return DataLoader(
            dataset=val_dataset,
            batch_size=self.hparams.DATASET.BATCH_SIZE,
            num_workers=self.hparams.DATASET.NUM_WORKERS,
            pin_memory=self.hparams.DATASET.PIN_MEMORY,
            shuffle=False,
            collate_fn=collator,
        )

    def test_dataloader(self):
        return self.val_dataloader()




