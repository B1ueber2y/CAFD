"""
Random shelter hacking
Each image is divided into m^2 regions and n of them are 
sheltered by a pixel sampled from the face

Params:
-- ORIGIN_IMAGE_PATH : The path containing original images.
-- split_size        : m
-- shelter_num       : n

Returns: 
-- SHELTER_IMAGE_PATH : Random sheltered images.
"""
import numpy as np
import tensorflow as tf
from scipy.misc import *
import pathlib
from utils import *
import random

#ORIGIN_IMAGE_PATH = './hack/'
ORIGIN_IMAGE_PATH = '/home/weiy/Evalgan/Section_3.1/hack/image/test/' 
SHELTER_IMAGE_PATH = './shelter/'

if not os.path.exists(SHELTER_IMAGE_PATH):
    os.makedirs(SHELTER_IMAGE_PATH)
os.system('rm ' + SHELTER_IMAGE_PATH + '*')

path = pathlib.Path(ORIGIN_IMAGE_PATH)
files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
images = [imread(str(fn)) for fn in files]
split_size = 8
shelter_num = 7
for i in range(len(images)):
    mean_value = images[i][16,32,:]
    choice_list_candidate = [m for m in range(split_size**2)]
    choice_list = random.sample(choice_list_candidate, int(shelter_num))
    for j in range(int(shelter_num)):
          
        index2 = np.mod(choice_list[j],split_size)
        index1 = choice_list[j]//split_size
        for k in range(3):
            images[i][index1*round(64/split_size):(index1+1)*round(64/split_size), index2*round(64/split_size):(index2+1)*round(64/split_size), k] = mean_value[k]
    print(i)
    imsave(SHELTER_IMAGE_PATH + str(i) + '.jpg', np.uint8(images[i]))

