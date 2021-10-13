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
import json

sys.path.append('')

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

import time
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from envmap import EnvironmentMap, rotation_matrix
import skimage.io as io
# from pare.utils.image_utils import read_img


PANO_DATASET_PATH = '/is/cluster/work/mkocabas/datasets/panorama_dataset/'


def project_image(e, pitch, yaw, roll, fov):
    rot = rotation_matrix(azimuth=np.radians(yaw), elevation=np.radians(pitch), roll=np.radians(roll))
    image = e.project(fov, rot, ar=1 / 1, resolution=(640, 640)) * 255
    return image


def crop_panoramic_image():
    # img_1 = '/ps/project/perspective_camera/render/unreal/images/20210127-perspective-citypark-panoramic-01-04/Frame_00206_FinalColor.png'
    img_1 = '/ps/project/perspective_camera/panorama_dataset/pixexid_panoramic_images/td0g3wf-ferris-wheel-miami-bayside.jpeg'

    img_1 = '/ps/project/perspective_camera/render/unreal/images/20210127-perspective-hqresidentialhouse/Frame_00013_FinalColor.png'

    e = EnvironmentMap(img_1, "latlong")

    yaw = np.concatenate([np.linspace(0, 120, 40), np.flip(np.linspace(0, 120, 40))])
    roll = np.concatenate([np.linspace(0, 15, 5), np.flip(np.linspace(0, 15, 5)),
                           np.linspace(0, -15, 5), np.flip(np.linspace(0, -15, 5))])
    pitch = np.concatenate([np.linspace(0, 30, 5), np.flip(np.linspace(0, 30, 5)),
                            np.linspace(0, -30, 5), np.flip(np.linspace(0, -30, 5))])

    fov_arr = np.concatenate([np.linspace(15,120,30), np.flip(np.linspace(15,120,30))])
    save_folder = '/home/mkocabas/Videos/panorama_video'
    idx = 0
    r = 0
    fov = 60  # degrees

    norm_img = lambda x: (x - x.min()) / np.ptp(x)

    for y in yaw:

        image = project_image(e, pitch=0, yaw=y, roll=0, fov=fov)
        image = norm_img(image) * 255

        io.imsave(f'{save_folder}/{idx:06d}.png', image.astype(np.uint8))
        idx += 1
        print(idx, end='\r')

    for p in pitch:
        image = project_image(e, pitch=p, yaw=0, roll=0, fov=fov)
        image = norm_img(image) * 255

        io.imsave(f'{save_folder}/{idx:06d}.png', image.astype(np.uint8))
        idx += 1
        print(idx, end='\r')

    for r in roll:
        image = project_image(e, pitch=0, yaw=0, roll=r, fov=fov)
        image = norm_img(image) * 255

        io.imsave(f'{save_folder}/{idx:06d}.png', image.astype(np.uint8))
        idx += 1
        print(idx, end='\r')

    for f in fov_arr:
        image = project_image(e, pitch=0, yaw=0, roll=0, fov=f)
        image = norm_img(image) * 255

        io.imsave(f'{save_folder}/{idx:06d}.png', image.astype(np.uint8))
        idx += 1
        print(idx, end='\r')

    # print(p, y, r, '->', idx, yaw.shape[0]*roll.shape[0]*pitch.shape[0], '\r')
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()


def test_aspect_ratio():
    aspect_ratio = [1/1, 5/4, 4/3, 3/2, 16/9]

    image_resolutions = [(640, 640), (750, 600), (800, 600), (900, 600), (992, 558), (558, 992)]
    freq = [0.1, 0.1, 0.5, 0.1, 0.1]

    img = '/ps/project/perspective_camera/render/unreal/images/20210126-perspective-citypark-panoramic-01/Frame_00001_FinalColor.png' # '/home/mkocabas/Pictures/3d50eke-360-panorama-pier-miami-bayside.jpeg'
    e = EnvironmentMap(img, "latlong")

    imgs = []
    for res in image_resolutions:
        fov = 45
        yaw = pitch = roll = 0.
        rot = rotation_matrix(azimuth=np.radians(yaw), elevation=np.radians(pitch), roll=np.radians(roll))

        ar = res[0] / res[1]
        print(res)

        image = e.project(fov, rot, ar=ar, resolution=res)
        imgs.append(image)

        print('image shape', image.shape)
        # image = np.concatenate(imgs, axis=1)

        plt.imshow(image)
        plt.show()


def count_valid_images(idx):
    # dataset_folder = '/is/cluster/work/mkocabas/datasets/panorama_dataset/flickr_panoramic_images'
    dataset_folder = f'{PANO_DATASET_PATH}/flickr_panoramic_images'

    valid_imgs = []
    non_valid_imgs = []

    unique_img_fnames = joblib.load(f'{PANO_DATASET_PATH}/flickr_panoramic_images/unique_images.npy')
    img_fnames = np.array_split(unique_img_fnames, 486)[idx].tolist()

    for img_fname in tqdm(img_fnames):
        # print(img_fname)
        try:
            e = EnvironmentMap(img_fname, "latlong")
            # h, w, _ = read_img(img_fname).shape
        except Exception as e:
            print(e)
            non_valid_imgs.append(img_fname)
            continue

        valid_imgs.append(img_fname)

    print(len(valid_imgs), '/', len(img_fnames))
    joblib.dump(
        valid_imgs,
        os.path.join(dataset_folder, f'split_valid_images/valid_imgs_pano_{idx}.pkl'),
    )

    joblib.dump(
        non_valid_imgs,
        os.path.join(dataset_folder, f'split_valid_images/non_valid_pano_imgs_{idx}.pkl'),
    )


def count_valid_pixexid():
    dataset_folder = f'{PANO_DATASET_PATH}/pixexid_panoramic_images'

    valid_imgs = []
    non_valid_imgs = []

    img_fnames = [os.path.join(dataset_folder, x) for x in os.listdir(dataset_folder)
                  if x.endswith('jpg') or x.endswith('jpeg')]

    for img_fname in tqdm(img_fnames):
        # print(img_fname)
        try:
            e = EnvironmentMap(img_fname, "latlong")
            # h, w, _ = read_img(img_fname).shape
        except Exception as e:
            print(e)
            non_valid_imgs.append(img_fname)
            continue

        valid_imgs.append(img_fname)

    print(len(valid_imgs), '/', len(img_fnames))
    joblib.dump(
        valid_imgs,
        os.path.join(dataset_folder, f'valid_imgs_pano.pkl'),
    )

    joblib.dump(
        non_valid_imgs,
        os.path.join(dataset_folder, f'non_valid_pano_imgs.pkl'),
    )


def count_unique_images():
    dataset_folder = f'{PANO_DATASET_PATH}/flickr_panoramic_images'

    unique_img_fnames = []
    all_img_ids = []

    for sub_img_folder in sorted([os.path.join(dataset_folder, x) for x in os.listdir(dataset_folder)]):

        if not os.path.isdir(sub_img_folder):
            continue

        # print(sub_img_folder)

        img_fnames = [x for x in os.listdir(sub_img_folder)
                      if x.endswith('.jpg') and not x in all_img_ids]

        all_img_ids += img_fnames

        abs_img_fnames = [os.path.join(sub_img_folder, x) for x in img_fnames]

        print(sub_img_folder)
        print(len(img_fnames))

        unique_img_fnames += abs_img_fnames

    joblib.dump(unique_img_fnames, os.path.join(dataset_folder, 'unique_images.npy'))
    import IPython; IPython.embed(); exit()


def sample_cam_params(num_cam, save_path):
    pitch = np.random.normal(loc=0.046, scale=0.3, size=num_cam)
    roll = np.random.normal(loc=0, scale=0.05, size=num_cam)
    vfov = np.random.normal(loc=67.5, scale=20, size=num_cam)

    for idx, vf in enumerate(vfov):
        if 120 > vf > 15:
            pass
        else:
            vf = np.random.normal(loc=67.5, scale=20)

            while not(120 > vf > 15):
                vf = np.random.normal(loc=67.5, scale=20)

            vfov[idx] = vf

    # vfov = np.clip(vfov, 15, 120)

    cam_params = {
        'pitch': pitch,
        'roll': roll,
        'vfov': vfov,
    }
    print('Saving cam params...')
    joblib.dump(cam_params, save_path)
    return cam_params


def list_synthetic_pano_images():
    images_path = '/ps/project/perspective_camera/render/unreal/images/'

    image_fnames = []

    subdirs = [
        '20210122-perspective-citypark-panoramic',
        '20210125-perspective-citypark-panoramic',
        '20210126-perspective-citypark-panoramic-01',
        '20210127-perspective-citypark-panoramic-01-04',
        '20210127-perspective-hqresidentialhouse',
        '20210128-perspective-downtownwest',
        '20210128-perspective-moderncitydowntown',
        '20210204-perspective-bodies',
        '20210208-perspective-supermarket',
    ]

    for s in subdirs:
        for im in os.listdir(images_path + s):
            if im.endswith('.png'):
                image_fnames.append(images_path + s + '/' + im)

    joblib.dump(image_fnames, '/is/cluster/work/mkocabas/datasets/panorama_dataset/synthetic_pano_images.pkl')

    import IPython; IPython.embed(); exit()


def preprocess_calib_data(idx=None):
    num_crops_per_image = 12

    num_splits = 600

    preprocessed_dataset_path = f'/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/data/dataset_folders/20210215-preprocessed_pano_dataset'

    os.makedirs(preprocessed_dataset_path, exist_ok=True)
    os.makedirs(f'{preprocessed_dataset_path}/images', exist_ok=True)
    os.makedirs(f'{preprocessed_dataset_path}/annotations', exist_ok=True)
    os.makedirs(f'{preprocessed_dataset_path}/errors', exist_ok=True)

    pano_img_files = joblib.load(f'{PANO_DATASET_PATH}/valid_imgs_pano.pkl')
    pano_img_files += joblib.load(f'{PANO_DATASET_PATH}/synthetic_pano_images.pkl')

    file_index = np.arange(len(pano_img_files))

    # sampled_cam_params = sample_cam_params(
    #     len(pano_img_files) * num_crops_per_image,
    #     save_path=preprocessed_dataset_path + '/sampled_cam_params.pkl',
    # )

    sampled_cam_params = joblib.load(f'{preprocessed_dataset_path}/sampled_cam_params.pkl')

    vfov = sampled_cam_params['vfov']
    pitch = sampled_cam_params['pitch']
    roll = sampled_cam_params['roll']

    # import IPython; IPython.embed(); exit()

    if idx is not None:
        pano_img_files = np.array_split(pano_img_files, num_splits)[idx]
        file_index = np.array_split(file_index, num_splits)[idx]

    print('Starting pano data preprocessing')
    print(f'Number of pano images: {len(pano_img_files)}')

    resolutions = [(640, 640), (750, 600), (800, 600), (900, 600), (992, 558), (558, 992)]
    res_freq = [0.1, 0.1, 0.5, 0.1, 0.1, 0.1]

    for f_idx, pano_img_f in tqdm(zip(file_index, pano_img_files)):

        try:
            env = EnvironmentMap(pano_img_f, "latlong")
        except AssertionError as e:
            print(e)
            continue

        for j in range(num_crops_per_image):
            i = f_idx * num_crops_per_image + j
            vf, p, r = vfov[i], pitch[i], roll[i]

            yaw = np.random.uniform(360)
            rot = rotation_matrix(azimuth=np.radians(yaw), elevation=p, roll=r)

            res = resolutions[np.random.multinomial(1, res_freq, size=1).argmax()]
            ar = res[0] / res[1]

            try:
                image = env.project(vf, rot, ar=ar, resolution=res)
            except Exception as exc:
                print(exc)
                with open(f'{preprocessed_dataset_path}/errors/{pano_img_f.split("/")[-1].replace("jpg", "txt")}', 'a') as f:
                    f.write(f'{exc}\n')

                continue

            # save image
            img_f_name = os.path.join(preprocessed_dataset_path, 'images', f'{f_idx:06d}_{j:03d}.png')
            io.imsave(img_f_name, (image * 255.).astype(np.uint8))

            # save params
            annot = {
                'orig_pano_imgname': pano_img_f,
                'imgname': img_f_name,
                'vfov': vf,
                'pitch': p,
                'roll': r,
                'img_res': image.shape,
            }
            annot_f_name = img_f_name.replace('images', 'annotations').replace('.png', '.json')

            with open(annot_f_name, 'w') as f:
                json.dump(annot, f)

            # import IPython; IPython.embed(); exit()

    print('DONE!')


def split_train_val_set():
    dataset_path = '/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/data/dataset_folders/20210215-preprocessed_pano_dataset'

    all_image_files = sorted(os.listdir(os.path.join(dataset_path, 'images')))

    val_pano_images = np.random.choice(np.arange(34831), size=500)

    val_images = []
    train_images = []

    for img_f in all_image_files:
        if int(img_f.split('/')[-1].split('_')[0]) in val_pano_images:
            val_images.append(img_f)
        else:
            train_images.append(img_f)

    joblib.dump(train_images, os.path.join(dataset_path, 'train_images.pkl'))
    joblib.dump(val_images, os.path.join(dataset_path, 'val_images.pkl'))


def test_panorama_dataset():
    dataset_path = f'{PANO_DATASET_PATH}preprocessed_data'
    image_dir = os.path.join(dataset_path, 'images')
    annot_dir = os.path.join(dataset_path, 'annotations')

    image_fnames = joblib.load(os.path.join(dataset_path, 'image_filenames.pkl'))

    for i in range(100):
        idx = np.random.randint(400000)
        img = io.imread(os.path.join(image_dir, image_fnames[idx]))
        ann = json.load(open(os.path.join(annot_dir, image_fnames[idx])))

        plt.title(f'vfov: {ann["vfov"]:.1f}, '
                  f'pitch: {np.degrees(ann["pitch"]):.1f}, '
                  f'roll: {np.degrees(ann["roll"]):.1f}')
        plt.imshow(img)
        plt.show()

    import IPython; IPython.embed(); exit()


if __name__ == '__main__':
    # count_unique_images()
    # count_valid_images(int(sys.argv[1]))
    # crop_panoramic_image()
    # count_valid_pixexid()
    # test_aspect_ratio()
    # sample_cam_params(num_cam=34831 * 12, save_path='/ps/project/perspective_camera/panorama_dataset/20210215-preprocessed_pano_dataset/sampled_cam_params.pkl')
    split_train_val_set()
    # preprocess_calib_data(int(sys.argv[1]))
    # test_panorama_dataset()
    # list_synthetic_pano_images()