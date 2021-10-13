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
import joblib
import requests
from tqdm import tqdm
from flickrapi import FlickrAPI

# flickr groups
# Ricoh theta       : https://www.flickr.com/groups/2846464@N25/
# Equirectangular   : https://www.flickr.com/groups/44671723@N00/
# Ricoh theta 2     : https://www.flickr.com/groups/2440649@N22/
# Ricoh theta z1    : https://www.flickr.com/groups/14634250@N21
# http://photopin.com/free-photos/equirectangular


API_KEY = 'XXXXXX' # change this with your own api key
API_SECRET = 'XXXXXX' # change this with your own api secret


FLICKR_GROUPS = {
    'ricoh_theta': ('2846464@N25', 40),
    'equirectangular': ('44671723@N00', 260),
    'ricoh_theta_2': ('2440649@N22', 50),
    'ricoh_theta_3': ('14634250@N21', 10),
    '360': ('91922148@N00', 300),
    'flickr_vr': ('2934608@N23', 50),
    'pano_lover': ('2737732@N23', 20),
    'people': ('89483931@N00', 2793),
}


class Flickr:
    """ This class is to collect images from a Flickr group """

    def __init__(self, api_key, api_secret, download_with_exif_only=False, download_original=False):
        self.api = FlickrAPI(api_key, api_secret, format='parsed-json')
        self.download_with_exif_only = download_with_exif_only
        self.download_original = download_original

    def get_images_from_ids(self, image_ids):
        images = []

        for photo_id in tqdm(image_ids):
            sizes = self.api.photos.getSizes(photo_id=photo_id)

            # get the largest size available
            url = sizes['sizes']['size'][-1]['source']

            if self.download_original:
                if not sizes['sizes']['size'][-1]['label'] == 'Original':
                    continue
            try:
                tags = self.api.photos_getExif(photo_id=photo_id)
            except:
                tags = None
                if self.download_with_exif_only:
                    continue

            images.append((url, str(photo_id), tags))

        print(len(images), 'will be dowloaded...')
        return images

    def get_images_urls_metadata(self, group_id=None, tag=None, num_images=50000, extras='url_o', page_id=None):
        """
        This function returns the images urls and images metadata from the selected group
        """
        # extras = 'url_l'
        images = []
        per_page = 500

        if group_id is not None:
            photos = self.api.groups.pools.getPhotos(group_id=group_id, extras=extras)
        elif tag is not None:
            photos = self.api.photos.search(tags=tag, extras=extras, per_page=per_page)
        else:
            raise ValueError

        print('Start Collecting...')

        print(f'Total number of pages {photos["photos"]["pages"]}')
        print(f'Total number of images {photos["photos"]["total"]}')

        if isinstance(page_id, int):
            page_list = [page_id]
        else:
            page_list = range(1, photos['photos']['pages'])

        print(page_list)

        for page in tqdm(page_list):
            if group_id is not None:
                photos = self.api.groups.pools.getPhotos(group_id=group_id, extras=extras, page=page)
            elif tag is not None:
                photos = self.api.photos.search(tags=tag, extras=extras, page=page, per_page=per_page)
            else:
                raise ValueError

            for photo in tqdm(photos['photos']['photo']):

                if len(images) == num_images:
                    return images

                photo_id = photo['id']

                sizes = self.api.photos.getSizes(photo_id=photo_id)

                # get the largest size available
                url = sizes['sizes']['size'][-1]['source']

                if self.download_original:
                    if not sizes['sizes']['size'][-1]['label'] == 'Original':
                        continue

                # import IPython; IPython.embed(); exit()

                try:
                    tags = self.api.photos_getExif(photo_id=photo_id)
                except:
                    tags = None
                    if self.download_with_exif_only:
                        continue

                images.append((url, str(photo['id']), tags))

            print(f'[PAGE {page}] Number of unique images so far: {len(list(set([x[0] for x in images])))}')

        print(len(images), 'will be dowloaded...')
        return images

    def download(self, group_id=None, tag=None, num_images=50000,
                 extras='url_o', destination_dir='.', page_id=None, image_ids=None):
        """
        This function downloads the images from the selected group
        Inputs:
        group_id: the group_id to download  images from
        num_images: number of images to be downloaded
        extras: the required resolution of the images
        destination_dir: the directory path to save the downloaded images
        """

        if group_id is not None:
            images = self.get_images_urls_metadata(group_id=group_id, num_images=num_images,
                                                   extras=extras, page_id=page_id)
        elif tag is not None:
            images = self.get_images_urls_metadata(tag=tag, num_images=num_images, extras=extras, page_id=page_id)
        elif image_ids is not None:
            images = self.get_images_from_ids(image_ids)
        else:
            raise ValueError

        joblib.dump(images, 'flickr_images.pkl')

        for url, id_, metadata in tqdm(images):
            # Request the image from the source
            response = requests.get(url)

            # Write the image in the selected directory
            with open(os.path.join(destination_dir, (id_ + '.jpg')), 'wb') as img:
                img.write(response.content)

            if metadata is not None:
                # Write the metadata to a json file
                with open(os.path.join(destination_dir, (id_ + '.json')), 'w') as f:
                    json.dump(metadata, f)


def scrape_and_download():
    dataset_folder = 'data/dataset_folders/pano360/flickr_pano_images'

    # For data collection
    num_images = 100000
    extras = 'url_o'
    download_type = 'tag'
    flickr = Flickr(API_KEY, API_SECRET, download_with_exif_only=True, download_original=True)

    if download_type == 'group':
        group_name = 'pano_lover'
        group_id = FLICKR_GROUPS[group_name][0]

        print(f'Downloading from group {group_name} with ID {group_id}')
        destination_dir = f'{dataset_folder}/group_{group_name}_cluster'

        os.makedirs(destination_dir, exist_ok=True)
        page_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
        print(f'Downloading page {page_id}')
        flickr.download(group_id=group_id, num_images=num_images, extras=extras,
                        destination_dir=destination_dir, page_id=page_id)
    elif download_type == 'tag':

        tag = 'people'

        print(f'Downloading from tag {tag}...')
        destination_dir = f'{dataset_folder}/tag_{tag}_cluster'
        os.makedirs(destination_dir, exist_ok=True)

        page_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
        print(f'Downloading page {page_id}')
        flickr.download(tag=tag, num_images=num_images, extras=extras,
                        destination_dir=destination_dir, page_id=page_id)


def download():
    import numpy as np
    flickr_image_ids = np.load('data/dataset_folders/pano360/flickr_photo_ids.npy')

    dataset_folder = 'data/dataset_folders/pano360/flickr_pano_images'
    os.makedirs(dataset_folder, exist_ok=True)

    # For data collection
    num_images = 100000
    extras = 'url_o'
    flickr = Flickr(API_KEY, API_SECRET, download_with_exif_only=True, download_original=True)

    flickr.download(num_images=num_images, extras=extras,
                    destination_dir=dataset_folder, image_ids=flickr_image_ids)


if __name__ == '__main__':
    download()
    # scrape_and_download()