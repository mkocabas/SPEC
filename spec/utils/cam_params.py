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
import torch
import joblib

from pare.utils.geometry import batch_euler2matrix


def read_cam_params(output_path, img_fname, orig_shape):
    # These are predicted camera parameters
    # cam_param_folder = CAM_PARAM_FOLDERS[dataset_name][cam_param_type]

    pred_cam_params = joblib.load(os.path.join(output_path, 'camcalib', os.path.basename(img_fname) + '.pkl'))

    cam_pitch = pred_cam_params['pitch'].item()
    cam_roll = pred_cam_params['roll'].item()

    cam_vfov = pred_cam_params['vfov'].item()

    cam_focal_length = pred_cam_params['f_pix']

    cam_rotmat = batch_euler2matrix(torch.tensor([[cam_pitch, 0., cam_roll]]).float())[0]

    pred_cam_int = torch.zeros(3, 3)

    cx, cy = orig_shape[1] / 2, orig_shape[0] / 2

    pred_cam_int[0, 0] = cam_focal_length
    pred_cam_int[1, 1] = cam_focal_length

    pred_cam_int[:-1, -1] = torch.tensor([cx, cy])

    cam_int = pred_cam_int.float()

    return cam_rotmat, cam_int, cam_vfov, cam_pitch, cam_roll, cam_focal_length