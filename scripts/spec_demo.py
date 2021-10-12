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
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
import cv2
import time
import joblib
import argparse
from loguru import logger

sys.path.append('')

from spec.tester import SPECTester

CFG = 'data/spec/checkpoints/spec_config.yaml'
CKPT = 'data/spec/checkpoints/spec_checkpoint.ckpt'
MIN_NUM_FRAMES = 0


def main(args):

    demo_mode = args.mode

    if demo_mode == 'video':
        raise NotImplementedError
    elif demo_mode == 'webcam':
        raise NotImplementedError
    elif demo_mode == 'folder':
        args.tracker_batch_size = 1
        input_image_folder = args.image_folder
        output_path = os.path.join(args.output_folder, input_image_folder.rstrip('/').split('/')[-1] + '_' + args.exp)
        os.makedirs(output_path, exist_ok=True)

        output_img_folder = os.path.join(output_path, 'spec_results')
        os.makedirs(output_img_folder, exist_ok=True)

        num_frames = len(os.listdir(input_image_folder))
    else:
        raise ValueError(f'{demo_mode} is not a valid demo mode.')

    logger.add(
        os.path.join(output_path, 'demo.log'),
        level='INFO',
        colorize=False,
    )
    logger.info(f'Demo options: \n {args}')

    tester = SPECTester(args)

    total_time = time.time()

    if args.mode == 'video':
        raise NotImplementedError
    elif args.mode == 'folder':
        logger.info(f'Number of input frames {num_frames}')

        total_time = time.time()
        # CamCalib
        tester.run_camcalib(input_image_folder, output_path)
        # Person detector
        detections = tester.run_detector(input_image_folder)
        spec_time = time.time()
        tester.run_on_image_folder(input_image_folder, detections, output_path, output_img_folder)
        end = time.time()

        fps = num_frames / (end - spec_time)

        del tester.model

        logger.info(f'SPEC FPS: {fps:.2f}')
        total_time = time.time() - total_time
        logger.info(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
        logger.info(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    logger.info('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str,
                        help='config file that defines model hyperparams', default=CFG)

    parser.add_argument('--ckpt', type=str,
                        help='checkpoint path', default=CKPT)

    parser.add_argument('--exp', type=str, default='',
                        help='short description of the experiment')

    parser.add_argument('--mode', default='folder', choices=['video', 'folder', 'webcam'],
                        help='Demo type')

    parser.add_argument('--vid_file', type=str,
                        help='input video path or youtube link')

    parser.add_argument('--image_folder', type=str,
                        help='input image folder')

    parser.add_argument('--output_folder', type=str, default='logs/demo/demo_results',
                        help='output folder to write results')

    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                        help='path to directory STAF pose tracking method installed.')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size of SPEC')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')

    parser.add_argument('--min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. '
                             'Decreasing the minimum cutoff frequency decreases slow speed jitter')

    parser.add_argument('--beta', type=float, default=1.0,
                        help='one euro filter beta. '
                             'Increasing the speed coefficient(beta) decreases speed lag.')

    parser.add_argument('--no_render', action='store_true',
                        help='disable final rendering of output video.')

    parser.add_argument('--no_save', action='store_true',
                        help='disable final save of output results.')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--draw_keypoints', action='store_true',
                        help='draw 2d keypoints on rendered image.')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    args = parser.parse_args()

    main(args)