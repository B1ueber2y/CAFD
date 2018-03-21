"""
Random noise hacking
Apply random noise uniformly distributed on each pixel

Params:
-- ORIGIN_IMAGE_PATH: The path containing original images
-- amp              : The amplitude of noise

Retturns:
-- NOISE_IMAGE_PATH : Noised images.
"""
import numpy as np
import tensorflow as tf
from scipy.misc import *
import pathlib
from utils import *
import random

ORIGIN_IMAGE_PATH = './hack/'
NOISE_IMAGE_PATH = './noise/'
path = pathlib.Path(ORIGIN_IMAGE_PATH)

if not os.path.exists(NOISE_IMAGE_PATH):
    os.makedirs(NOISE_IMAGE_PATH)
os.system('rm ' + NOISE_IMAGE_PATH + '*')

amp = 33
files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
images = [imread(str(fn)) for fn in files]
for i in range(len(images)):
    np.random.seed(i)
    noise = np.round((2*np.random.rand(64,64,3) - 1)*amp)
    images[i] += noise
    images[i][images[i] >= 255] = 255
    images[i][images[i] <= 0] = 0
    imsave(NOISE_IMAGE_PATH + str(i) + '.png', np.uint8(images[i]))

