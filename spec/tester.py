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
import joblib
import numpy as np
from loguru import logger
from yolov3.yolo import YOLOv3
from multi_person_tracker import MPT
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from pare.utils.geometry import batch_euler2matrix
from pare.utils.train_utils import load_pretrained_model
from pare.utils.vibe_image_utils import get_single_image_crop_demo

from .models import HMR
from .config import update_hparams
from .utils.cam_params import read_cam_params
from .utils.renderer_cam import render_image_group

MIN_NUM_FRAMES = 0


class SPECTester:
    def __init__(self, args):
        self.args = args
        self.model_cfg = update_hparams(args.cfg)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self._build_model()
        self._load_pretrained_model()
        self.model.eval()

    def _build_model(self):
        # ========= Define SPEC model ========= #
        model_cfg = self.model_cfg

        model = HMR(
            backbone=model_cfg.HMR.BACKBONE,
            img_res=model_cfg.DATASET.IMG_RES,
            pretrained=model_cfg.TRAINING.PRETRAINED,
            use_cam_feats=model_cfg.HMR.USE_CAM_FEATS,
            use_cam=True,
        ).to(self.device)

        return model

    def _load_pretrained_model(self):
        # ========= Load pretrained weights ========= #
        if self.args.ckpt == 'spin':
            logger.warning('CKPT file is not provided, using SPIN weights')
        else:
            logger.info(f'Loading pretrained model from {self.args.ckpt}')
            ckpt = torch.load(self.args.ckpt)['state_dict']
            load_pretrained_model(self.model, ckpt, overwrite_shape_mismatch=True, remove_lightning=True)
            logger.info(f'Loaded pretrained weights from \"{self.args.ckpt}\"')

    def run_detector(self, image_folder):
        # run multi object tracker
        mot = MPT(
            device=self.device,
            batch_size=self.args.tracker_batch_size,
            display=self.args.display,
            detector_type=self.args.detector,
            output_format='dict',
            yolo_img_size=self.args.yolo_img_size,
        )
        bboxes = mot.detect(image_folder)
        return bboxes

    def run_camcalib(self, image_folder, output_folder):
        cmd = f'python scripts/camcalib_demo.py --img_folder {image_folder} --out_folder {output_folder}/camcalib --no_save'
        os.system(cmd)

    @torch.no_grad()
    def run_on_image_folder(self, image_folder, detections, output_path, output_img_folder, bbox_scale=1.0):
        image_file_names = [
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
        ]
        image_file_names = sorted(image_file_names)

        for img_idx, img_fname in enumerate(image_file_names):
            dets = detections[img_idx]

            if len(dets) < 1:
                continue

            img = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)

            orig_height, orig_width = img.shape[:2]

            inp_images = torch.zeros(len(dets), 3, self.model_cfg.DATASET.IMG_RES,
                                     self.model_cfg.DATASET.IMG_RES, device=self.device, dtype=torch.float)

            batch_size = inp_images.shape[0]

            bbox_scale = []
            bbox_center = []
            for det_idx, det in enumerate(dets):
                bbox = det
                norm_img, raw_img, kp_2d = get_single_image_crop_demo(
                    img,
                    bbox,
                    kp_2d=None,
                    scale=1.0,
                    crop_size=self.model_cfg.DATASET.IMG_RES
                )
                inp_images[det_idx] = norm_img.float().to(self.device)

                bbox_scale.append(bbox[2] / 200.)
                bbox_center.append([bbox[0], bbox[1]])

            bbox_center = torch.tensor(bbox_center)
            bbox_scale = torch.tensor(bbox_scale)
            img_h = torch.tensor(orig_height).repeat(batch_size)
            img_w = torch.tensor(orig_width).repeat(batch_size)

            cam_rotmat, cam_intrinsics, cam_vfov, cam_pitch, cam_roll, cam_focal_length = \
                read_cam_params(output_path, img_fname, (orig_height, orig_width))

            # import IPython; IPython.embed(); exit()

            cam_rotmat = cam_rotmat.unsqueeze(0).repeat(batch_size, 1, 1)
            cam_intrinsics = cam_intrinsics.unsqueeze(0).repeat(batch_size, 1, 1)

            output = self.model(
                inp_images,
                cam_rotmat=cam_rotmat.float().to(self.device),
                cam_intrinsics=cam_intrinsics.float().to(self.device),
                bbox_scale=bbox_scale.float().to(self.device),
                bbox_center=bbox_center.float().to(self.device),
                img_w=img_w.float().to(self.device),
                img_h=img_h.float().to(self.device),
            )

            for k,v in output.items():
                output[k] = v.cpu().numpy()

            del inp_images

            if not self.args.no_save:
                save_f = os.path.join(
                    output_path, 'spec_results',
                    os.path.basename(img_fname).replace(img_fname.split('.')[-1], 'pkl')
                )
                joblib.dump(output, save_f)

            if not self.args.no_render:
                pred_vertices = torch.from_numpy(output['smpl_vertices'])
                pred_cam_t = torch.from_numpy(output['pred_cam_t'])

                render_rotmat = batch_euler2matrix(
                    torch.tensor([[-cam_pitch, 0., cam_roll]])
                )[0] # pyrender opengl convention

                focal_length = (cam_focal_length, cam_focal_length)

                cy, cx = orig_height // 2, orig_width // 2

                # import IPython; IPython.embed(); exit()

                cam_params = np.array([cam_vfov, cam_pitch, cam_roll, cam_focal_length])

                for i in range(batch_size):
                    mesh_filename = None
                    if self.args.save_obj:
                        mesh_folder = os.path.join(output_path, 'meshes', os.path.basename(img_fname).split('.')[0])
                        os.makedirs(mesh_folder, exist_ok=True)
                        mesh_filename = os.path.join(mesh_folder, f'{i:06d}.obj')

                    fname, img_ext = os.path.splitext(img_fname)
                    save_filename = os.path.join(output_img_folder, f'{os.path.basename(fname)}_{i:06d}{img_ext}')

                    render_image_group(
                        image=img,
                        camera_translation=pred_cam_t[i],
                        vertices=pred_vertices[i],
                        camera_rotation=render_rotmat,
                        focal_length=focal_length,
                        camera_center=(cx, cy),
                        save_filename=save_filename,
                        mesh_filename=mesh_filename,
                        cam_params=cam_params,
                    )

                    if self.args.display:
                        cv2.imshow('SPEC results', img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        if self.args.display:
            cv2.destroyAllWindows()