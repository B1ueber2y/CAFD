"""
Get celebA images
Same to DCGAN, we first crop and resize celebA images before hacking.
We use 10000 images for hacking.
Params:
-- IMAGE_PATH: The path containing original celebA images

Returns:
-- HACK_PATH : Images after dropped and resized 
"""
import numpy as np
import tensorflow as tf
from scipy.misc import *
import pathlib
from utils import *


IMAGE_PATH = './celebA/'
HACK_IMAGE_PATH = './hack/'

if not os.path.exists(HACK_IMAGE_PATH):
    os.makedirs(HACK_IMAGE_PATH)
os.system('rm ' + HACK_IMAGE_PATH + '*')

path = pathlib.Path(IMAGE_PATH)
files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
images = [get_image(str(fn), 108, 108, 64, 64, 1) for fn in files]
mean_image = np.empty(images[0].shape)
mean_value = np.zeros(3)
for i in range(10000):
    imsave(HACK_IMAGE_PATH + str(i) + '.png', np.uint8(images[i]))
