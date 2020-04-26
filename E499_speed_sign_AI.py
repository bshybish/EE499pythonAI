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


def shuffle_c(xx_test, yy_test):
    tempRand = np.arange(34)
    tempRand = np.random.permutation(tempRand)
    j = 0
    for k in tempRand:
        (xx_test[k,:,:], xx_test[j,:,:]) = (xx_test[j,:,:], xx_test[k,:,:])
        (yy_test[k], yy_test[j]) = (yy_test[j], yy_test[k])
        j = j + 1
    return (xx_test, yy_test)
    
    
#%%


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
    


#%%

(zx_test, zy_test) = shuffle_c(x_test, y_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation= tf.nn.softmax))

model.compile(optimizer='adam',
              loss     ='sparse_categorical_crossentropy'
              )

history = model.fit(x_test, y_test, epochs=5)

#predss = model.predict([x_test])

#%%
print(zy_test[1])
zz = zx_test[1,:,:]

#%%
print(y_test[5])
zz = x_test[5,:,:]


