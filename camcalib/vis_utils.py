"""
Script from https://github.com/Jerrypiglet/ScaleNet/tree/master/RELEASE_SUN360_camPred_minimal
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EUCLID = "EUCLID" in sys.argv


def plot_performance(output_dir, data_pitch):
    import matplotlib
    matplotlib.use('qt5agg')
    from matplotlib import pyplot as plt

    data_pitch_abs = np.abs(data_pitch)

    dpas = np.sort(data_pitch_abs)
    cdf = np.linspace(0, 1, dpas.size)

    plt.plot(dpas, cdf)

    bl, ul = plt.ylim()
    plt.plot([0.05, 0.05], [bl, ul], color=[1, 0, 0])

    plt.xlim([0, 0.4])

    plt.xlabel('Pitch error (image ratio)')

    plt.ylabel('% Correct')
    plt.savefig(os.path.join(output_dir, "pitch_error.png"), bbox_inches='tight', dpi=150)
    plt.close()


def draw_line(image, hl, hr, leftright=(None, None), color=(0,255,0), width=5):
    # hl, hr: [top: 1, bottom: 0]
    if np.isnan([hl, hr]).any():
        return image

    h, w, c = image.shape
    if image.dtype in (np.float32, np.float64):
        image = (image * 255).astype('uint8')

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)

    l = (1-hl)*h
    r = (1-hr)*h

    b = 0
    if leftright[0] is not None:
        b = leftright[0]
    if leftright[1] is not None:
        w = leftright[1]

    draw.line((b, l, w, r), fill=color, width=width) # [top: 0, bottom: 1]
    return np.array(im)


def show_horizon_line(
        image, vfov, pitch, roll, focal_length=-1,
        color=(0, 255, 0), width=5, debug=False, GT=False, text_size=16,
):
    """
    Angles should be in radians.
    """
    h, w, c = image.shape
    if image.dtype in (np.float32, np.float64):
        image = image.astype('uint8')

    if debug:
        if GT == False:
            image[0:text_size,:,:] = 0
        else:
            image[h-text_size:h,:,:] = 0

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)

    # text_size =  h // 25
    # fnt = ImageFont.truetype("/usr/share/fonts/truetype/gentium-basic/GenBasR.ttf", text_size)

    ctr = h * (0.5 - 0.5 * np.tan(pitch) / np.tan(vfov / 2))
    l = ctr - w * np.tan(roll) / 2
    r = ctr + w * np.tan(roll) / 2
    if debug:
        if GT == False:
            draw.text(
                (0, 0),
                "vfov:{0:.1f}, pitch:{1:.1f}, roll:{2:.1f}, f_pix:{3:.1f}".format(
                    np.degrees(vfov), np.degrees(pitch), np.degrees(roll), focal_length
                ),
                (255, 255, 255),
                # font=fnt,
            )
        else:
            draw.text(
                (0, h-text_size),
                "GT: vfov:{0:.1f}, pitch:{1:.1f}, roll:{2:.1f}, f_pix:{3:.1f}".format(
                    np.degrees(vfov), np.degrees(pitch), np.degrees(roll), focal_length
                ),
                (255, 255, 255),
                # font=fnt,
            )

    draw.line((0, l, w, r), fill=color, width=width)
    return np.array(im), ctr/h


def show_horizon_line_from_horizon(image, horizon, color=(0, 255, 0), width=5, debug=False, GT=False):
    """
    Angles should be in radians.
    """
    h, w, c = image.shape
    if image.dtype in (np.float32, np.float64):
        image = (image * 255).astype('uint8')

    # if debug:
    #     image[0:12,:,:] = 0

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)

    l = horizon * h
    r = horizon * h
    if debug:
        if GT == False:
            draw.text((0, 12), "v0:{0:.2f}".format(horizon), (255, 255, 255))
        else:
            draw.text((0, h-24), "GT: v0:{0:.2f}".format(horizon), (255, 255, 255))

    draw.line((0, l, w, r), fill=color, width=width)
    return np.array(im)


def get_horizon_line(vfov, pitch):
    """
    Angles should be in radians.
    """
    ctr = 0.5 - 0.5*np.tan(pitch) / np.tan(vfov/2)
    return ctr


