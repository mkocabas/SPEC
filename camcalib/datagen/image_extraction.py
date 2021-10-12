'''
Source: https://github.com/Jerrypiglet/ScaleNet
'''

__version__ = '1'

import numpy as np
import time
import os
import os.path
import multiprocessing
import argparse
import itertools
import sys
from math import radians, degrees
from skimage.io import imread, imsave
from scipy.ndimage.interpolation import map_coordinates, geometric_transform, zoom

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


FILL_VALUE = 0


# Source of the formulaes : mathworld.wolfram.com/GnomonicProjection.html
def latlong2rectilinear(phi, lambda_, phi0, lambda0):
    cosc = np.sin(phi0)*np.sin(phi) + np.cos(phi0)*np.cos(phi)*np.cos(lambda_-lambda0)

    return ((np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(lambda_ - lambda0)) / cosc,   # y
            np.cos(phi) * np.sin(lambda_ - lambda0) / cosc)                                                 # x


def rectilinear2latlong(x, y, phi0, lambda0):
    rho = np.sqrt(x**2 + y**2)
    c = np.arctan(rho)

    return (np.arcsin(np.cos(c) * np.sin(phi0) + y * np.sin(c) * np.cos(phi0) / (rho+1e-10)),                              # phi (elevation)
            lambda0 + np.arctan2(x * np.sin(c), rho * np.cos(phi0) * np.cos(c) - y * np.sin(phi0) * np.sin(c)))    # lambda (azimuth)


def latlong2rectilinear_cache(sinphi, cosphi, lambda_, phi0, lambda0):
    cosc = np.sin(phi0)*sinphi + np.cos(phi0)*cosphi*np.cos(lambda_-lambda0)

    return ((np.cos(phi0) * sinphi - np.sin(phi0) * cosphi * np.cos(lambda_ - lambda0)) / cosc,   # y
            cosphi * np.sin(lambda_ - lambda0) / cosc)


def extractImage(envmapPath, viewing_angles, output_height, vfov=50, ratio=4./3., mode="image", out_dtype="uint8", interp_order=1):
    """
    Extract an image from an environment map.

    :envmapPath: Either a path to or directly an environment map (image)
    :viewing_angles: phi (elevation), lambda (azimuth), and theta (roll) in radians 
    :output_height: The height of the output image
    :vfov: vertical field of view (in degrees)
    :ratio: ratio width/height
    :mode: 'image' produces an image while 'mask' produce the masked environment map in output.
    """
    # 37.8 fov == 35mm lens
    elevation = azimuth = roll = 0
    if len(viewing_angles) > 0:
        elevation = viewing_angles[0]
    if len(viewing_angles) > 1:
        azimuth = viewing_angles[1]
    if len(viewing_angles) > 2:
        roll = viewing_angles[2]

    t = time.time()

    if type(envmapPath).__module__ == np.__name__:
        envmap = envmapPath
    else:
        try:
            envmap = imread(envmapPath).astype(np.float32)
        except (TypeError, OSError, FileNotFoundError):
            print("Error with file {}".format(envmapPath))
            return

    ratiohw = 1./ratio
    fovRad = np.radians(vfov)
    fovY = np.tan(fovRad / 2.)
    fovX = fovY / ratiohw

    producedInputs, producedOutputs, anglesDesc = [], [], []
    totalAvgInputs, totalStdInputs = np.zeros((3,)), np.zeros((3,))

    if mode in ['mask', 'maskbool']:
        # We produce a latlong image, masked everywhere except in one specific zone
        t1 = time.time()
        azimPixCoords, elevPixCoords = np.meshgrid(np.arange(-envmap.shape[1]//2, envmap.shape[1]//2),
                                                    np.arange(-envmap.shape[0]//2, envmap.shape[0]//2),
                                                    indexing='xy')
        azimCoords = azimPixCoords / (envmap.shape[1]/2) * np.pi
        elevCoords = elevPixCoords / (envmap.shape[0]/2) * np.pi / 2.
        sinphi, cosphi = np.sin(elevCoords), np.cos(elevCoords)
        maskElevCoords = ((np.pi/2 < elevCoords - elevation) | (elevCoords - elevation < -np.pi/2))
        #t2 = time.time()
        #t3=time.time()
        mask = np.zeros(np.shape(azimCoords),dtype=bool)
        # 4 parts in the mask : azimuth, elevation and the wrap (quadrant) uncertainty of tan
        for azimuthOffset in range(-2,3,2):
            ycoords, xcoords = latlong2rectilinear_cache(sinphi, cosphi, azimCoords, 
                                                         elevation, azimuth+azimuthOffset*np.pi)
            # apply roll to the x and y coordinates in the image plane
            flatxcoords = np.reshape(xcoords,(1,np.product(np.shape(xcoords))))
            flatycoords = np.reshape(ycoords,(1,np.product(np.shape(ycoords))))
            i = np.stack((flatxcoords,flatycoords),axis=0)
            xform = np.array([[np.cos(roll),-np.sin(roll)],[np.sin(roll),np.cos(roll)]])
            rolled = (np.mat(i).T*np.mat(xform)).T
            xcoordsrolled = np.array(np.reshape(rolled[0],np.shape(xcoords)))
            ycoordsrolled = np.array(np.reshape(rolled[1],np.shape(ycoords)))

            mask += (((fovX < xcoordsrolled) | (xcoordsrolled < -fovX ))
                    | ((fovY < ycoordsrolled) | (ycoordsrolled < -fovY ))
                    | ((np.pi/2 < azimCoords - azimuth) | (azimCoords - azimuth < -np.pi/2))
                    | maskElevCoords)

        outimg = envmap.copy()

        for c in range(3):
            outimg[..., c][mask] = 0

        if mode == 'maskbool':
            outimg[..., c][~mask] = 1
            outimg = np.any(outimg, axis=2).astype('bool')

    else:
        nb_channels = envmap.shape[-1]

        # We produce a rectilinear image, cropped from the latlong envmap
        croppedSize = (output_height, round(output_height / ratiohw))
        if any([cs < 1 for cs in croppedSize]):
            print("Warning! negative resolution {}x{} (from aspect ratio {})".format(croppedSize[1],croppedSize[0],ratiohw))
        xcoords, ycoords = np.meshgrid( np.linspace(-fovX, fovX, croppedSize[1]),
                                        np.linspace(-fovY, fovY, croppedSize[0]),
                                        indexing='xy')

        # apply roll in the image plane before doing the gnomonic projection
        flatxcoords = np.reshape(xcoords,(1,np.product(np.shape(xcoords))))
        flatycoords = np.reshape(ycoords,(1,np.product(np.shape(xcoords))))
        i = np.stack((flatxcoords,flatycoords),axis=0)
        xform = np.array([[np.cos(roll),-np.sin(roll)],[np.sin(roll),np.cos(roll)]])
        rolled = (np.mat(i).T*np.mat(xform)).T
        xcoordsrolled = np.array(np.reshape(rolled[0],croppedSize))
        ycoordsrolled = np.array(np.reshape(rolled[1],croppedSize))

        elev, azimuth = rectilinear2latlong(xcoordsrolled, ycoordsrolled, elevation, azimuth)
        azimuth[azimuth > np.pi] -= 2*np.pi
        azimuth[azimuth < -np.pi] += 2*np.pi
        azimuthPix = azimuth / np.pi * envmap.shape[1] / 2 + envmap.shape[1] / 2
        elevPix = elev / (np.pi / 2) * envmap.shape[0] / 2 + envmap.shape[0] / 2

        outimg = np.empty(croppedSize + (nb_channels,), dtype=out_dtype)
        coordinates = np.stack((elevPix, azimuthPix), axis=0)

        for c in range(nb_channels):  # Color channels
            map_coordinates(envmap[..., c], coordinates, outimg[..., c], order=interp_order, prefilter=False, mode="wrap")

    return outimg


def demo():
    imgdir = '/home/j/Documents/GitRepos/OutdoorIllumination/data/'
    imglist = ['pano_askvbepeztrtfo.jpg', 'pano_awouoctwfnhqsv.jpg']
#    extractImage('/home/j/Documents/GitRepos/OutdoorIllumination/data/pano_askvbepeztrtfo.jpg', (0, np.pi/2), 256, 0) #, mode="maskbool")
#    exit()

#    from panotools import sun360_tools
#    img_paths = sun360_tools.getImageList('/home/j/Documents/GitRepos/OutdoorIllumination/data/pano_askvbepeztrtfo.jpg')

    os.makedirs(imgdir+'output', exist_ok=True)

    imid = 0
    for img_name in imglist:
        img_path = os.path.join(imgdir,img_name)
        img = imread(img_path).astype(np.float32)
        phi = 0
        lambda_ = 0
        for a in range(1): #phi in np.arange(-np.pi/2, np.pi/2 + np.pi/6, np.pi/2):
            for b in range(1): #lambda_ in np.arange(-np.pi, np.pi, np.pi/2):
                for roll in np.arange(-np.pi,np.pi,np.pi/4):
                    out = extractImage(img, [phi, lambda_, roll], 320)
                    mask = extractImage(img, [phi, lambda_, roll], 320, mode="maskbool")
                    basename = os.path.splitext(os.path.basename(img_path))[0]
    
                    print(os.path.join(imgdir, 'output', "{}_{}.png".format(basename, degrees(roll))))
                    imsave(os.path.join(imgdir, 'output', "{}_{}.png".format(basename, degrees(roll))), out.astype('uint8'))
                    imsave(os.path.join(imgdir, 'output', "{}_mask_{}.png".format(basename, degrees(roll))), mask.astype('uint8'))
                    #imsave(os.path.join('output', "{}_mask_{}.png".format(basename, imid)), mask.astype('uint8'))

                    imid += 1


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='SUN360 dataset preparation for Image2Lightning')
    # parser.add_argument('inputsrc', type=str, help='Input environment map directory or file.')
    # parser.add_argument('output', type=str, help='Output data directory')
    # parser.add_argument('--fov', type=float, default=60., help="FOV of the simulated camera (degrees)")
    # parser.add_argument('--coveredangle', type=int, default=360, choices=[180, 360], help='Angle covered by the reconstructed envmap (180 or 360)')
    # parser.add_argument('--mode', type=str, default='full', choices=['full', 'crop'], help='Type of input to produce: "crop" provides only the cropped (redressed) image, as a camera could take. "full" provides the full envmap to the network, but masked everywhere except in the relevant part.')
    # parser.add_argument('--overlap', type=int, default=0, help='Overlapping (in degrees) between each crop (default 0)')
    # parser.add_argument('--flip', type=int, default=0, help='Also flip the images to produce 2x the number of samples')
    # parser.add_argument('--envmapheight', type=int, default=160, help='Height of the envmap to produce')
    # parser.add_argument('--croppedheight', type=int, default=160, help='Height of the cropped region (only used if inputtype == "crop")')
    # parser.add_argument('--color', type=bool, default=True, help='Output color or gray images')
    # parser.add_argument('--interporder', type=int, default=1, help='Order of the interpolation (0=nearest neighbor, 1=bilinear, etc.)')
    # parser.add_argument('--nprocesses', type=int, default=0, help='Number of concurrent processes to use. Default is 0, meaning number of available cores.')
    # parser.add_argument('--fillwith', type=str, default='sample', choices=['zero', 'sample', 'random'], help='How to fill the masked regions : zeros (with 0), random (random data from 0-1 uniform), sample (sample pixels from the non-masked part at random)')
    # args = parser.parse_args()
    demo()

    # nprocesses = args.nprocesses if args.nprocesses > 0 else multiprocessing.cpu_count()
    # print('Using {} processes'.format(nprocesses))

    # p = multiprocessing.Pool(nprocesses)
    # r = p.map(processEnvmap, tasklist)

    # successN = sum(f[0] for f in r)
    # print('Done! Processed {} images over {} possible'.format(successN, len(tasklist)))


