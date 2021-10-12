'''
Source: https://github.com/Jerrypiglet/ScaleNet
'''

import sys
import time
if sys.version >= "3":
    from contextlib import ContextDecorator
else:
    from contextdecorator import ContextDecorator

import numpy as np
from PIL import Image, ImageDraw


QUIET = "QUIET" in sys.argv


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def DispWarning(text, color=bcolors.WARNING):
    print(color, text, bcolors.ENDC)


class DispDebug(ContextDecorator):
    def __init__(self, name, disp=None, stream=sys.stdout):
        self.name = name
        self.disp = disp if disp is not None else not QUIET
        self.stream = stream

    def __enter__(self):
        if self.disp:
            self.stream.write("{}...".format(self.name))
            self.stream.flush()
            self.ts = time.time()
        return self

    def __exit__(self, *exc):
        if self.disp:
            self.stream.write(' done in {0:.3f}s'.format(time.time() - self.ts))
            self.stream.flush()
        return False

if __name__ == '__main__':
    import joblib
    import os
    import random
    import numpy as np

    image_list = []

    for i in range(600):
        with open(f'/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/data/dataset_folders/20210219-preprocessed_pano_dataset/image_list/{i:05d}.txt', 'r') as f:
            image_files = [x.rstrip() for x in f.readlines()]

            if len(image_files) < 690:
                print(i)

            image_list += image_files

    joblib.dump(image_list,
                '/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/data/dataset_folders/20210219-preprocessed_pano_dataset/image_list/images.pkl')

    pano_images = list(set([os.path.dirname(x) for x in image_list]))

    val_panos = np.random.choice(pano_images, 1200)

    train_images = []
    val_images = []

    for im in image_list:
        if os.path.dirname(im) in val_panos:
            val_images.append(im)
        else:
            train_images.append(im)

    joblib.dump(train_images,
                '/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/data/dataset_folders/20210219-preprocessed_pano_dataset/image_list/train_images.pkl')

    joblib.dump(val_images,
                '/ps/scratch/mkocabas/developments/cvpr2021_projects/pare/data/dataset_folders/20210219-preprocessed_pano_dataset/image_list/val_images.pkl')

    import IPython; IPython.embed(); exit()

