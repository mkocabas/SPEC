'''
Source: https://github.com/Jerrypiglet/ScaleNet
'''
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


EUCLID = "EUCLID" in sys.argv


def plotPerformance(output_dir, data_pitch):
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

    # data_roll_abs = np.abs(data_roll)*180/np.pi

    # dras = np.sort(data_roll_abs)
    # cdf = np.linspace(0, 1, dras.size)

    # ud = np.unique(dras)
    # ui = np.unique(dras, return_index=True)[1]

    # plt.plot(ud, cdf[ui])
    # plt.xlim([0, 4])
    # plt.xlabel('Roll error (degree)')
    # plt.ylabel('% Correct')
    # plt.savefig(os.path.join(output_dir, "roll_error.png"), bbox_inches='tight', dpi=150)
    # plt.close()


def drawLine(image, hl, hr, leftright=(None, None), color=(0,255,0), width=5):
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


def showHorizonLine(image, vfov, pitch, roll, focal_length=-1, color=(0, 255, 0), width=5, debug=False, GT=False):
    """
    Angles should be in radians.
    """
    h, w, c = image.shape
    if image.dtype in (np.float32, np.float64):
        image = (image * 255).astype('uint8')

    if debug:
        if GT == False:
            image[0:12,:,:] = 0
        else:
            image[h-12:h,:,:] = 0

    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)

    ctr = h*( 0.5 - 0.5*np.tan(pitch) / np.tan(vfov/2) )
    l = ctr - w*np.tan(roll)/2
    r = ctr + w*np.tan(roll)/2
    if debug:
        if GT == False:
            draw.text((0, 0), "vfov:{0:.2f}, pitch:{1:.2f}, roll:{2:.2f}, f_mm:{3:.2f}".format(vfov*180/np.pi, pitch*180/np.pi, roll*180/np.pi, focal_length), (255, 255, 255))
        else:
            draw.text((0, h-12), "GT: vfov:{0:.2f}, pitch:{1:.2f}, roll:{2:.2f}, f_mm:{3:.2f}".format(vfov*180/np.pi, pitch*180/np.pi, roll*180/np.pi, focal_length), (255, 255, 255))

    draw.line((0, l, w, r), fill=color, width=width)
    return np.array(im), ctr/h

def showHorizonLineFromHorizon(image, horizon, color=(0, 255, 0), width=5, debug=False, GT=False):
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
        # draw.text((0, 0), "vfov:{0:.2f}, pitch:{1:.2f}, roll:{2:.2f}".format(vfov*180/np.pi, pitch*180/np.pi, roll*180/np.pi), (255, 255, 255))
        # image[0:12,:,:] = 0
        if GT == False:
            draw.text((0, 12), "v0:{0:.2f}".format(horizon), (255, 255, 255))
        else:
            draw.text((0, h-24), "GT: v0:{0:.2f}".format(horizon), (255, 255, 255))

    draw.line((0, l, w, r), fill=color, width=width)
    return np.array(im)

def getHorizonLine(vfov, pitch):
    """
    Angles should be in radians.
    """
    ctr = 0.5 - 0.5*np.tan(pitch) / np.tan(vfov/2)
    return ctr



if __name__ == '__main__':
    from hdrio import imread, imsave
    import matplotlib
    matplotlib.use('qt5agg')
    from matplotlib import pyplot as plt
    import dataset
    fn, fp, data = dataset.getParameters()
    valid_idx = dataset.getValidIdx(data)
    fn = fn[valid_idx]; fp = fp[valid_idx]; data = data[valid_idx,:]
    pitch = data[:,0]; roll = data[:,2]; vfov = data[:,3]

    for i in range(100):
        print(i)
        print(fp[i], pitch[i], roll[i], vfov[i])

        im = imread(fp[i], 'native')[:,:,:3]
        imout = showHorizonLine(im, vfov[i], pitch[i], roll[i], debug=True)
        imsave('{0:03d}.png'.format(i), imout)
        #plt.clf()
        #plt.imshow(imout)
        #plt.show(block=False)
        #plt.pause(1)
