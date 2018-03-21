"""
Random exchange hacking
Each image is divided into m^2 regions and 
randomly exchange two of them for n times.

Params:
-- ORIGIN_IMAGE_PATH  : The path containing original images
-- split_size         : m
-- change_num         : n

Returns:
-- EXCHANGE_IMAGE_PATH: Random exchanged images.
"""
import numpy as np
import tensorflow as tf
from scipy.misc import *
import pathlib
from utils import *
import random

ORIGIN_IMAGE_PATH = './hack/'
EXCHANGE_IMAGE_PATH = './exchange/'

if not os.path.exists(EXCHANGE_IMAGE_PATH):
    os.makedirs(EXCHANGE_IMAGE_PATH)
os.system('rm ' + EXCHANGE_IMAGE_PATH + '*')

path = pathlib.Path(ORIGIN_IMAGE_PATH)
files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
images = [imread(str(fn)) for fn in files]
split_size = 4
change_num = 2
for i in range(len(images)):
    for j in range(change_num):
        choice_list_candidate = [m for m in range(split_size**2)]
        choice_list = random.sample(choice_list_candidate, 2)
        index0_2 = np.mod(choice_list[0],split_size)
        index0_1 = choice_list[0]//split_size
        index1_2 = np.mod(choice_list[1],split_size)
        index1_1 = choice_list[1]//split_size
        temp0 = images[i][index0_1*round(64/split_size):(index0_1+1)*round(64/split_size), index0_2*round(64/split_size):(index0_2+1)*round(64/split_size), :].copy()
        temp1 = images[i][index1_1*round(64/split_size):(index1_1+1)*round(64/split_size), index1_2*round(64/split_size):(index1_2+1)*round(64/split_size), :].copy()
        images[i][index1_1*round(64/split_size):(index1_1+1)*round(64/split_size), index1_2*round(64/split_size):(index1_2+1)*round(64/split_size), :] = temp0
        images[i][index0_1*round(64/split_size):(index0_1+1)*round(64/split_size), index0_2*round(64/split_size):(index0_2+1)*round(64/split_size), :] = temp1
    imsave(EXCHANGE_IMAGE_PATH + str(i) + '.png', np.uint8(images[i]))

