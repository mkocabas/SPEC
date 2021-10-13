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
import joblib
import numpy as np
from tqdm import tqdm
from loguru import logger
from smplx import SMPL as SMPLorig

from pare.models import SMPL
from pare.core.constants import H36M_TO_J14
from pare.utils.eval_utils import compute_error_verts, reconstruction_error

from ..config import DATASET_FILES, SMPL_MODEL_DIR


def eval_j_24(pred_joints, gt_joints):
    pred_pelvis = pred_joints[:, [0], :].clone()
    pred_joints = pred_joints - pred_pelvis

    gt_pelvis = gt_joints[:, [0], :].clone()
    gt_joints = gt_joints - gt_pelvis

    pampjpe, _ = reconstruction_error(
        pred_joints.cpu().numpy(),
        gt_joints.cpu().numpy(),
        reduction=None,
    )

    pampjpe *= 1000

    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * 1000
    return mpjpe, pampjpe


def eval_single(pred_vertices, gt_vertices, J_regressor_batch):
    pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
    pred_pelvis = pred_joints[:, [0], :].clone()
    pred_joints = pred_joints[:, H36M_TO_J14, :]
    pred_joints = pred_joints - pred_pelvis

    gt_joints = torch.matmul(J_regressor_batch, gt_vertices)
    gt_pelvis = gt_joints[:, [0], :].clone()
    gt_joints = gt_joints[:, H36M_TO_J14, :]
    gt_joints = gt_joints - gt_pelvis

    # v2v = compute_error_verts(
    #     pred_verts=pred_vertices.cpu().numpy(),
    #     target_verts=gt_vertices.cpu().numpy(),
    # ) * 1000

    v2v = compute_error_verts(
        pred_verts=(pred_vertices - pred_pelvis).cpu().numpy(),
        target_verts=(gt_vertices - gt_pelvis).cpu().numpy(),
        # target_verts=(gt_vertices).cpu().numpy(),
    ) * 1000

    # pred_joints = pred_model_out.joints[:, 25:]
    # gt_joints = gt_model_out.joints[:, 25:]

    pampjpe, _ = reconstruction_error(
        pred_joints.cpu().numpy(),
        gt_joints.cpu().numpy(),
        reduction=None,
    )

    pampjpe *= 1000

    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * 1000
    return mpjpe, pampjpe, v2v


def compute_error(results_file):
    device = 'cuda'

    dataset_name = os.path.basename(results_file).replace('evaluation_results_', '').replace('.pkl', '')
    dataset_file = DATASET_FILES[0][dataset_name]

    results = joblib.load(results_file)

    logger.add(
        results_file.replace('.pkl', '_analysis.log'),
        level='INFO',
        colorize=False,
    )

    data = np.load(dataset_file)

    pose_key = 'pose_0yaw_inverseyz' if dataset_name.startswith('3dpw') else 'pose'

    pred_cam_rotmat = joblib.load(f'data/camcalib/{dataset_name}_cam_rotmat.pkl')
    pred_vertices = torch.from_numpy(results['vertices']).float()

    del results

    J_regressor = torch.from_numpy(np.load('data/J_regressor_h36m.npy')).float().to(device)
    J_regressor_batch = J_regressor[None, :].expand(1, -1, -1)

    body_model = SMPL(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False,
        create_global_orient=True,
    ).to(device)

    body_model_orig = SMPLorig(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False,
        create_global_orient=True,
    ).to(device)

    wvertex2vertex_error = np.zeros(len(data['imgname']))
    pampjpe_error = np.zeros(len(data['imgname']))
    wmpjpe_error = np.zeros(len(data['imgname']))

    mpjpe_error = np.zeros(len(data['imgname']))
    vertex2vertex_error = np.zeros(len(data['imgname']))

    pampjpe_24_error = np.zeros(len(data['imgname']))
    wmpjpe_24_error = np.zeros(len(data['imgname']))
    mpjpe_24_error = np.zeros(len(data['imgname']))

    batch_idxs = np.array_split(np.arange(len(data['imgname'])), 100)

    for idx in tqdm(batch_idxs):

        pred_cam_rotmat_ = pred_cam_rotmat[idx].float().to(device)

        gt_pose = torch.from_numpy(data[pose_key][idx]).float()
        gt_betas = torch.from_numpy(data['shape'][idx]).float()
        gt_model_out = body_model(
            global_orient=gt_pose[:, :3].to(device=device).float(),
            body_pose=gt_pose[:, 3:].to(device=device).float(),
            betas=gt_betas.to(device=device),
        )

        gt_vertices = gt_model_out.vertices

        gt_joints = body_model_orig(
            global_orient=gt_pose[:, :3].to(device=device).float(),
            body_pose=gt_pose[:, 3:].to(device=device).float(),
            betas=gt_betas.to(device=device),
        ).joints[:, :24]

        if dataset_name == 'spec-syn':
            gt_cam_rotmat = torch.from_numpy(data['cam_rotmat'][idx]).float().to(device)
            gt_cam_vertices = torch.bmm(gt_cam_rotmat, gt_vertices.transpose(2, 1)).transpose(2, 1)
            gt_cam_joints = torch.bmm(gt_cam_rotmat, gt_joints.transpose(2, 1)).transpose(2, 1)
            pred_cam_rotmat_ = gt_cam_rotmat
        else:
            # gt_cam_rotmat = torch.from_numpy(data['cam_rotmat'][idx]).float().to(device)
            gt_pose_cam = torch.from_numpy(data['pose_cam'][idx]).float()
            gt_model_out_cam = body_model(
                global_orient=gt_pose_cam[:, :3].to(device=device).float(),
                body_pose=gt_pose_cam[:, 3:].to(device=device).float(),
                betas=gt_betas.to(device=device),
            )
            gt_cam_vertices = gt_model_out_cam.vertices

            gt_cam_joints = body_model_orig(
                global_orient=gt_pose_cam[:, :3].to(device=device).float(),
                body_pose=gt_pose_cam[:, 3:].to(device=device).float(),
                betas=gt_betas.to(device=device),
            ).joints[:, :24]

        pred_verts = pred_vertices[idx].to(device=device).float()
        pred_joints = torch.einsum('bik,ji->bjk', [pred_verts, body_model_orig.J_regressor])

        pred_vertices_gt_cam = torch.bmm(pred_cam_rotmat_, pred_verts.transpose(2, 1)).transpose(2, 1)
        pred_cam_joints = torch.einsum('bik,ji->bjk', [pred_vertices_gt_cam, body_model_orig.J_regressor])

        wmpjpe, pampjpe, wv2v = eval_single(pred_verts, gt_vertices, J_regressor_batch)
        mpjpe, _, v2v = eval_single(pred_vertices_gt_cam, gt_cam_vertices, J_regressor_batch)

        wmpjpe_24, pampjpe_24 = eval_j_24(pred_joints, gt_joints)
        mpjpe_24, _ = eval_j_24(pred_cam_joints, gt_cam_joints)

        wvertex2vertex_error[idx] = wv2v
        vertex2vertex_error[idx] = v2v

        wmpjpe_error[idx] = wmpjpe
        mpjpe_error[idx] = mpjpe

        pampjpe_error[idx] = pampjpe

        pampjpe_24_error[idx] = pampjpe_24
        wmpjpe_24_error[idx] = wmpjpe_24
        mpjpe_24_error[idx] = mpjpe_24

    logger.info(f'***** RESULTS ON {dataset_name.upper()} *****')
    if dataset_name == '3dpw-test-cam':
        # standard protocol for 3dpw is 14 joint evaluation
        logger.info(f'W-MPJPE: {wmpjpe_error.mean():.3f}')
        logger.info(f'C-MPJPE: {wmpjpe_error.mean():.3f}')
        logger.info(f'MPJPE: {mpjpe_error.mean():.3f}')
        logger.info(f'PA-MPJPE: {pampjpe_error.mean():.3f}')
    else:
        # we use 24 SMPL joints for SPEC-SYN and SPEC-MTP evaluation
        logger.info(f'W-MPJPE-24: {wmpjpe_24_error.mean():.3f}')
        logger.info(f'C-MPJPE-24: {wmpjpe_24_error.mean():.3f}')
        logger.info(f'MPJPE-24: {mpjpe_24_error.mean():.3f}')
        logger.info(f'PA-MPJPE-24: {pampjpe_24_error.mean():.3f}')

    logger.info(f'W-V2V: {wvertex2vertex_error.mean():.3f}')
    logger.info(f'C-V2V: {wvertex2vertex_error.mean():.3f}')
    logger.info(f'V2V: {vertex2vertex_error.mean():.3f}')