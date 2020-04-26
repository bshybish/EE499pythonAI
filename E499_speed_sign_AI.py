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
x_test          = np.array([])

print(len(y_test))
print(type(y_test))

x1 = np.array(tf.keras.utils.normalize(cv2.imread(mypath + t_files[1], cv2.IMREAD_GRAYSCALE)))



for i in t_files:
    x_test = np.append(x_test,tf.keras.utils.normalize(cv2.imread(mypath + i, cv2.IMREAD_GRAYSCALE)))

