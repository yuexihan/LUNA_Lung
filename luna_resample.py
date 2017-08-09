from __future__ import division, print_function, absolute_import

# All coordinates appearing in arguments or being returned are in [z, y, x] form.
# All coordinates are immediately transformed to [z, y, x] form.
# So coordinates in other form will not be exposed.

# Any variable with prefix 'np' is np.ndarray. The others are optional.

import os
import pandas as pd
import glob
import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import math
from tqdm import tqdm
import logging


def worldCoord2voxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def get_trunk(image, voxelCoord):
    """
    Extract sample trunk of shape [22, 40, 40] centered at nodule point.
    """
    z, y, x = voxelCoord
    _x = int(round(x))
    if _x > x:
        startX = _x - 20
    else:
        startX = _x - 19
    _y = int(round(y))
    if _y > y:
        startY = _y - 20
    else:
        startY = _y - 19
    _z = int(round(z))
    if _z > z:
        startZ = _z - 11
    else:
        startZ = _z - 10

    stopX = startX + 40
    stopY = startY + 40
    stopZ = startZ + 22

    shapeZ, shapeY, shapeX = image.shape
    beforeX = afterX = beforeY = afterY = beforeZ = afterZ = 0

    if startX < 0:
        beforeX = -startX
        startX = 0
    elif stopX > shapeX:
        afterX = stopX - shapeX
        stopX = shapeX
    if startY < 0:
        beforeY = -startY
        startY = 0
    elif stopY > shapeY:
        afterY = stopY - shapeY
        stopY = shapeY
    if startZ < 0:
        beforeZ = -startZ
        startZ = 0
    elif stopZ > shapeZ:
        afterZ = stopZ - shapeZ
        stopZ = shapeZ

    trunk = image[startZ:stopZ, startY:stopY, startX:stopX]

    return np.pad(trunk, ((beforeZ, afterZ), (beforeY, afterY), (beforeX, afterX)), mode='edge')


def resample(image, spacing, new_spacing=[1.2,0.7,0.7]):
    """
    Resample image to unified resolution.
    """
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest', order=1)
    return image, new_spacing


def load_mhd_image(filename):
    itkimage = sitk.ReadImage(filename)
    npImage = sitk.GetArrayFromImage(itkimage)
    npOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    npSpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return npImage, npOrigin, npSpacing


def rotate_image_points(image, points, angel):
    """
    Rotate image within XY plane.
    Calculate the new voxel coordinates of the points in the rotated image.
    """
    def rotate_point(point):
        z, y, x = point - mid
        _y = y * math.cos(theta) - x * math.sin(theta)
        _x = y * math.sin(theta) + x * math.cos(theta)
        return [z, _y, _x] + mid
    if angel % 90 == 0:
        rotated_image = np.rot90(image, angel//90, [1, 2])
    else:
        rotated_image = scipy.ndimage.rotate(image, angel, axes=(2, 1), order=1, reshape=False, mode='nearest')

    mid = np.array(image.shape) / 2
    theta = angel * np.pi / 180
    rotated_points = [rotate_point(p) for p in points]
    return rotated_image, rotated_points


class SampleAnnatations:
    def __init__(self, logger):
        self.annotations = pd.read_csv('csv_files/annotations.csv')
        self.logger = logger

    def __call__(self, image, origin, spacing, subset, pId):
        folder = os.path.join('Samples', 'annotations', subset, pId)
        os.makedirs(folder, exist_ok=True)
        # Extract and convert coordinates of nodules.
        nodules = self.annotations[self.annotations.seriesuid==pId][['coordZ', 'coordY', 'coordX']].as_matrix()
        points = [worldCoord2voxelCoord(p, origin, spacing) for p in nodules]
        for angel in tqdm(range(0, 360, 45)): # Rotate in XY plane.
            rotated_image, rotated_points = rotate_image_points(image, points, angel)
            for x in range(-3, 4): # Shift along x, y, z axes.
                for y in range(-3, 4):
                    for z in range(-2, 3):
                        for i, p in enumerate(rotated_points):
                            trunk = get_trunk(rotated_image, p + (z, x, y))
                            # assert trunk.shape == (22, 40, 40), file + '%sR.%sX.%sY.%sZ_%s' % (angel, x, y, z, i)
                            if trunk.shape != (22, 40, 40):
                                self.logger.error('%s: %sR.%sX.%sY.%sZ_%s %s', pId, angel, x, y, z, i, trunk.shape)
                                continue
                            trunk = trunk.tobytes()
                            label = np.array(1, dtype=np.int64).tobytes()
                            binFile = os.path.join(folder, '%sR.%sX.%sY.%sZ_%s.bin' % (angel, x, y, z, i))
                            with open(binFile, 'wb') as f:
                                f.write(label + trunk)


class SampleCandidates:
    def __init__(self, logger):
        self.candidates = pd.read_csv('csv_files/candidates_V2.csv')
        self.logger = logger

    def __call__(self, image, origin, spacing, subset, pId):
        folder = os.path.join('Samples', 'candidates', subset, pId)
        os.makedirs(folder, exist_ok=True)
        # Extract and convert coordinates of nodules.
        nodules = self.candidates[self.candidates.seriesuid==pId][['coordZ', 'coordY', 'coordX']].as_matrix()
        points = [worldCoord2voxelCoord(p, origin, spacing) for p in nodules]
        classes = self.candidates[self.candidates.seriesuid==pId]['class'].tolist()
        for angel in tqdm(range(0, 360, 90)):
            rotated_image, rotated_points = rotate_image_points(image, points, angel)
            for i, (p, c) in enumerate(zip(rotated_points, classes)):
                trunk = get_trunk(rotated_image, p)
                if trunk.shape != (22, 40, 40):
                    self.logger.error('%s: %s_%s %s', pId, angel, i, trunk.shape)
                    continue
                trunk = trunk.tobytes()
                label = np.array(c, dtype=np.int64).tobytes()
                binFile = os.path.join(folder, '%sR_%s.bin' % (angel, i))
                with open(binFile, 'wb') as f:
                    f.write(label + trunk)

if __name__ == '__main__':
    # Initialize logger.
    fh = logging.FileHandler('resample.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    annotations_logger = logging.getLogger('resample.annotations')
    annotations_logger.setLevel(logging.INFO)
    annotations_logger.addHandler(fh)
    candidates_logger = logging.getLogger('resample.candidates')
    candidates_logger.setLevel(logging.INFO)
    candidates_logger.addHandler(fh)

    sample_annotations = SampleAnnatations(annotations_logger)
    sample_candidates = SampleCandidates(candidates_logger)

    files = glob.glob('LUNA_data/subset[0-9]/*.mhd')
    for file in tqdm(files):
        # Load and resample image.
        image, origin, spacing = load_mhd_image(file)
        image, spacing = resample(image, spacing)

        subset, pId = file.split('/')[-2:]
        pId = pId[:-4]
        sample_annotations(image, origin, spacing, subset, pId)
        sample_candidates(image, origin, spacing, subset, pId)

