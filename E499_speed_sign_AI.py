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



#%%


#mypath          = "C:\\Users\\Bassam\\Documents\\training_data\\first_set\\"
mypath = "C:\\Users\\mm\\Documents\\GitHub\\EE499pythonAI\\Train_Arabic_Traffic_Signs_24_2200\\"
t_files         = listdir(mypath)

y_test = np.empty(519)

for i in range(519):
    if (i<=61):
        y_test[i]=100
    if (i>61 and i<=167):
        y_test[i]=30
    if (i>167 and i<=260):
        y_test[i]=40
    if (i>260 and i<=350):
        y_test[i]=50
    if (i>377 and i<=439):
        y_test[i]=80
    if (i>439 and i<=519):
        y_test[i]=100
        
#y_test          = [30, 40, 30, 30, 40, 50, 50, 50, 70, 80, 80, 80, 80,
#                   80, 80, 80, 80, 80, 100, 100, 80, 90, 90,
##                   90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
x_test          = np.zeros((0,128,128))

def shuffle_c(xx_test, yy_test):
    tempRand = np.arange(128)
    tempRand = np.random.permutation(tempRand)
    j = 0
    for k in tempRand:
        (xx_test[k,:,:], xx_test[j,:,:]) = (xx_test[j,:,:], xx_test[k,:,:])
        (yy_test[k], yy_test[j]) = (yy_test[j], yy_test[k])
        j = j + 1
    return (xx_test, yy_test)
    
    
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
model.add(tf.keras.layers.Dense(1000, activation= tf.nn.softmax))

model.compile(optimizer='adam',
              loss     ='sparse_categorical_crossentropy'
              )
y_test = np.array(y_test)
history = model.fit(x_test, y_test, epochs=100)

predss = model.predict([x_test])

#%%
print(zy_test[1])
zz = zx_test[1,:,:]

#%%
print(y_test[5])
zz = x_test[5,:,:]

#%%

print("the prediction value:"+str(np.argmax(predss[1])))
print("the true value: "+str(y_test[1]))
plt.imshow(x_test[1], cmap= 'gray')
plt.show()


