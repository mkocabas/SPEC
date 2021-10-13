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

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from pare.models.layers.softargmax import softargmax1d


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, pred, target):
        target = F.one_hot(target, num_classes=pred.shape[-1]).float()
        return F.kl_div(F.log_softmax(pred, dim=1), target, reduction='batchmean')


class SoftargmaxClsLoss(nn.Module):
    def __init__(self, criterion='l2'):
        '''
        criterion (str) 'l2' 'biased_l2'
        '''
        super(SoftargmaxClsLoss, self).__init__()
        self.criterion = criterion

    def forward(self, pred, target):
        '''
        pred (N, 256)
        target (N,)
        '''
        pred = pred.unsqueeze(1) # (N, 1, 256)
        pred_argmax, _ = softargmax1d(pred, normalize_keypoints=True) # (N, 1, 1)
        pred_argmax = pred_argmax.reshape(-1)

        # logger.debug(f'pred: {pred_argmax}\ntarget: {target}')
        if self.criterion == 'l2':
            loss = (target - pred_argmax) ** 2 # F.mse_loss(target, pred_argmax)
        elif self.criterion == 'biased_l2':
            l2 = (target - pred_argmax) ** 2
            loss = torch.where(torch.gt(pred_argmax, target), l2, l2/(l2+1))
        else:
            raise ValueError(f'{self.criterion} is not defined!')

        return loss.mean()


class CameraRegressorLoss(nn.Module):
    def __init__(
            self,
            vfov_loss_weight=1.0,
            pitch_loss_weight=1.0,
            roll_loss_weight=1.0,
            loss_type='kl',
    ):
        '''
        loss_type (str):
            'kl', 'ce', 'l2', 'biased_l2', 'softargmax_l2', 'softargmax_biased_l2'
        '''
        super(CameraRegressorLoss, self).__init__()
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == 'kl':
            self.criterion = KLDivergence()
        elif loss_type == 'softargmax_l2':
            self.criterion = SoftargmaxClsLoss(criterion='l2')
        elif loss_type == 'softargmax_biased_l2':
            self.criterion = SoftargmaxClsLoss(criterion='l2')
            self.criterion_biased = SoftargmaxClsLoss(criterion='biased_l2')
        else:
            raise ValueError(f'{loss_type} is not defined..')

        self.loss_type = loss_type
        self.vfov_loss_weight = vfov_loss_weight
        self.pitch_loss_weight = pitch_loss_weight
        self.roll_loss_weight = roll_loss_weight

    def forward(
            self,
            pred_vfov,
            pred_pitch,
            pred_roll,
            gt_vfov,
            gt_pitch,
            gt_roll,
    ):
        pred_vfov = pred_vfov.squeeze(-1)
        pred_pitch = pred_pitch.squeeze(-1)
        pred_roll = pred_roll.squeeze(-1)

        # logger.debug(f'pred_v: {pred_vfov.shape}, gt_f: {gt_vfov.shape}')
        # logger.debug(f'pred_p: {pred_pitch.shape}, gt_c: {gt_pitch.shape}')
        # logger.debug(f'pred_r: {pred_roll.shape}, gt_c: {gt_roll.shape}')

        if self.loss_type == 'softargmax_biased_l2':
            vfov_loss = self.vfov_loss_weight * self.criterion_biased(pred_vfov, gt_vfov)
        else:
            vfov_loss = self.vfov_loss_weight * self.criterion(pred_vfov, gt_vfov)
        pitch_loss = self.pitch_loss_weight * self.criterion(pred_pitch, gt_pitch)
        roll_loss = self.roll_loss_weight * self.criterion(pred_roll, gt_roll)

        loss = vfov_loss + pitch_loss + roll_loss

        loss_dict = {
            'loss': loss,
            'vfov_loss': vfov_loss,
            'pitch_loss': pitch_loss,
            'roll_loss': roll_loss,
        }

        return loss, loss_dict

if __name__ == '__main__':
    sacls = SoftargmaxClsLoss()

    pred = torch.rand(1, 256)
    target = torch.rand(1)
    target[:] = 1

    print(softargmax1d(pred.unsqueeze(1))[0])

    loss = sacls(pred, target)

    print(loss)
