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
import json
import torch
import joblib
import random
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm

from camcalib.config import DATASET_FOLDERS
from camcalib.cam_utils import pitch_bins, roll_bins, vfov_bins, vfov2soft_idx, pitch2soft_idx, roll2soft_idx
from camcalib.cam_utils import pitch_bins_centers, roll_bins_centers, vfov_bins_centers, soft_idx_to_angle

from pare.utils.image_utils import read_img
from torch.utils.data import Dataset, DataLoader
from pare.utils.image_utils import denormalize_images


def get_eval_transform(min_size=600, max_size=1000):
    return transforms.Compose([
        transforms.Resize(min_size),# Resize(min_size, max_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def preprocess_data(split='train'):
    dataset = {
        'imgname': [],
        'pitch': [],
        'roll': [],
        'vfov': [],
    }

    ###### AGORA CAM
    from pare.core.config import DATASET_FOLDERS, DATASET_FILES
    is_train = 1 if split == 'train' else 0

    agora_dataset_file = DATASET_FILES[is_train]['agora-cam-v2']
    agora_data = np.load(agora_dataset_file)

    imgnames = agora_data['imgname']
    pitches = agora_data['cam_pitch']
    rolls = agora_data['cam_roll']
    fs = agora_data['focal_length']

    for idx in tqdm(range(len(agora_data['imgname'])), desc='AGORA-CAM', ascii=False):
        imgname = os.path.join(DATASET_FOLDERS['agora-cam-v2'], imgnames[idx])
        pitch = pitches[idx]
        roll = rolls[idx]
        # hfov = agora_data['cam_hfov'][idx]
        f = fs[idx]

        vfov = 2 * np.arctan(1080. / 2. / f)[0]

        dataset['imgname'].append(imgname)
        dataset['pitch'].append(pitch)
        dataset['roll'].append(roll)
        dataset['vfov'].append(vfov)

    # import ipdb; ipdb.set_trace()
    ###### PANO360
    dataset_folder = '/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/data/dataset_folders/20210219-preprocessed_pano_dataset'
    image_filenames = joblib.load(os.path.join(dataset_folder, f'{split}_images.pkl'))

    for imgname in tqdm(image_filenames, desc='PANO'):

        dataset['imgname'].append(imgname)
        data = json.load(open(imgname.replace('.jpg', '.json')))
        pitch = data['pitch']  # in radians
        roll = data['roll']  # in radians
        vfov = data['vfov']  # in radians

        dataset['pitch'].append(pitch)
        dataset['roll'].append(roll)
        dataset['vfov'].append(vfov)

    np.savez(f'data/dataset_extras/pano_agora_dataset_{split}.npz', **dataset)


class PanoAgoraDataset(Dataset):
    def __init__(
            self,
            is_train=True,
            min_size=600,
            max_size=1000,
            loss_type='kl',
            num_images=-1
    ):
        self.loss_type = loss_type

        split = 'train' if is_train else 'val'
        self.data = np.load(f'data/dataset_extras/pano_agora_dataset_{split}.npz')

        self.image_filenames = self.data['imgname']
        self.pitch = self.data['pitch']
        self.roll = self.data['roll']
        self.vfov = self.data['vfov']

        if is_train:
            color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            )

            self.data_transform = transforms.Compose([
                color_jitter,
                Resize(min_size, max_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.data_transform = transforms.Compose([
                Resize(min_size, max_size),
                # transforms.Resize(min_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if num_images > 0:
            self.image_filenames = np.random.choice(self.image_filenames, num_images)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        item = {}

        imgname = self.image_filenames[index] # os.path.join(self.dataset_folder, 'images', self.image_filenames[index])

        pil_img = Image.open(imgname).convert('RGB')
        orig_img_shape = pil_img.size
        norm_img = self.data_transform(pil_img)

        item['img'] = norm_img
        item['imgname'] = imgname
        item['orig_shape'] = orig_img_shape

        pitch = self.pitch[index]  # in radians
        roll = self.roll[index]  # in radians
        vfov = self.vfov[index]  # in radians

        item['vfov'] = torch.tensor(vfov)
        item['pitch'] = torch.tensor(pitch)
        item['roll'] = torch.tensor(roll)

        # print(f'fov: {np.degrees(vfov):.1f}, pitch: {np.degrees(pitch):.1f}, roll: {np.degrees(roll):.1f}')

        if self.loss_type in ('kl', 'ce'):
            item['vfov_bin'] = torch.tensor(np.digitize(vfov, vfov_bins)).long()
            item['pitch_bin'] = torch.tensor(np.digitize(pitch, pitch_bins)).long()
            item['roll_bin'] = torch.tensor(np.digitize(roll, roll_bins)).long()
        elif self.loss_type in ('softargmax_l2', 'softargmax_biased_l2'):
            item['vfov_bin'] = torch.tensor(vfov2soft_idx(vfov)).float()
            item['pitch_bin'] = torch.tensor(pitch2soft_idx(pitch)).float()
            item['roll_bin'] = torch.tensor(roll2soft_idx(roll)).float() # 2 * (torch.tensor(np.digitize(roll, roll_bins)).float() / roll_bins.shape[0]) - 1

        return item


class ImageFolder(Dataset):
    def __init__(
            self,
            image_list,
            min_size=600,
            max_size=1000,
    ):
        self.image_filenames = image_list

        self.data_transform = transforms.Compose([
            transforms.Resize(min_size),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        item = {}

        imgname = os.path.join(self.image_filenames[index])

        pil_img = Image.open(imgname).convert('RGB')
        orig_img_shape = pil_img.size
        norm_img = self.data_transform(pil_img)

        item['img'] = norm_img
        item['imgname'] = imgname

        item['orig_shape'] = orig_img_shape

        return item


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


def collator(batch):
    new_batch = {}
    images = to_image_list([x['img'] for x in batch])
    new_batch['img'] = images.tensors
    new_batch['img_sizes'] = images.image_sizes

    for k, v in batch[0].items():
        if k is 'img':
            continue

        if isinstance(v, torch.Tensor):
            new_batch[k] = torch.stack([x[k] for x in batch])
        else:
            new_batch[k] = [x[k] for x in batch]

    return new_batch


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]

        return ImageList(batched_imgs, image_sizes)
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


def test_dataset():
    import matplotlib.pyplot as plt
    from vis_utils import show_horizon_line

    ds = PanoAgoraDataset(is_train=True, loss_type='softargmax_l2')

    print('Dataset length: ', len(ds))
    dl = DataLoader(dataset=ds, batch_size=1, shuffle=True, collate_fn=collator)

    for idx, batch in enumerate(dl):
        img = batch['img']
        orig_img_size = batch['img_sizes'][0]
        # print(img.min(), img.max())
        # breakpoint()
        img = denormalize_images(img) * 255
        img = np.transpose(img.numpy(), (0, 2, 3, 1))

        pitch_bin = batch['pitch_bin'][0].numpy()
        roll_bin = batch['roll_bin'][0].numpy()
        vfov_bin = batch['vfov_bin'][0].numpy()

        print('Bin label', vfov_bin, pitch_bin, roll_bin)

        if ds.loss_type in ('kl', 'ce'):
            pitch = np.degrees(pitch_bins_centers[pitch_bin])
            roll = np.degrees(roll_bins_centers[roll_bin])
            vfov = np.degrees(vfov_bins_centers[vfov_bin])

        elif ds.loss_type in ('softargmax_l2', 'softargmax_biased_l2'):

            vfov = soft_idx_to_angle(vfov_bin, min=np.min(vfov_bins), max=np.max(vfov_bins))
            pitch = soft_idx_to_angle(pitch_bin, min=np.min(pitch_bins), max=np.max(pitch_bins))
            roll = soft_idx_to_angle(roll_bin, min=-0.6, max=0.6) # roll_bins_centers[((roll_bin + 1) / 2 * 256).round().astype(int)]

            vfov, pitch, roll = np.degrees(vfov), np.degrees(pitch), np.degrees(roll)

        else:
            raise ValueError

        img, _ = show_horizon_line(img[0].copy(), np.radians(vfov), np.radians(pitch), np.radians(roll),
                                   debug=True, color=(0, 255, 0), width=5, GT=True)

        print('Real', np.degrees(batch['vfov'].item()), np.degrees(batch['pitch'].item()), np.degrees(batch['roll'].item()),)
        print('Quant', vfov, pitch, roll)
        # print(batch['focal_length'])
        plt.title(f'fov: {vfov:.1f}, pitch: {pitch:.1f}, roll: {roll:.1f}')
        plt.imshow(img[:orig_img_size[0], :orig_img_size[1]])
        plt.show()


if __name__ == '__main__':
    test_dataset()