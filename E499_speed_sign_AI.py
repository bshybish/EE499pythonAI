# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:50:17 2020

@author: Bassam
"""
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np 
from PIL import Image 
from os import listdir
from os.path import isfile, join
from skimage import io, filters

mypath          = "C:\\Users\\Bassam\\Documents\\training_data\\first_set\\"
t_files         = listdir(mypath)
y_test          = [30, 40, 30, 30, 40, 50, 50, 50, 70, 80, 80, 80, 80,
                   80, 80, 80, 80, 80, 100, 100, 80, 90, 90,
                   90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
x_test          = np.zeros((0,128,128))

for i in t_files:
    x4 = cv2.imread(mypath + i, cv2.IMREAD_GRAYSCALE)
    x4 = cv2.resize(x4, (128,128))
    x4 = tf.keras.utils.normalize(x4)
    x_test = np.append(x_test,[x4],axis=0)

