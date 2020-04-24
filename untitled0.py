# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 00:18:38 2020

@author: Bassam
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import confusion_matrix

print(tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation= tf.nn.softmax))

model.compile(optimizer='adam',
              loss     ='sparse_categorical_crossentropy',
              matrices =['accuracy'])

history = model.fit(x_train, y_train, epochs=5)

predss = model.predict([x_test])


#%%
print("the prediction value:"+str(np.argmax(predss[110])))
print("the true value: "+str(y_test[110]))
plt.imshow(x_test[110], cmap= plt.cm.binary)
plt.show()
#%%

rang = []
for i in range(1,len(history.history['loss'])+1):
    rang.append(i)
print(history.history.keys())
plt.bar(rang,history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("loss")
sd_predss = list()
sd_y_test = np.asarray(y_test).tolist()
for i in range(0,10000):
    sd_predss.append(np.argmax(predss[i]))
    
#print(np.argmax(predss[3]))
print(type( sd_y_test ))
print(type(sd_predss))
klConMat = confusion_matrix(sd_y_test, sd_predss)
#%%

sum_true = 0

for i in range(10):
    sum_true += klConMat[i,i]

print(sum_true)
print(klConMat.sum(0).sum(0))
print()
#%%

index49 = []

for i in range(10000):
    if(sd_y_test[i] == 4 and sd_predss[i] == 9):
        index49.append(i)

#%%
for i in index49:
    
    plt.imshow(x_test[i], cmap= plt.cm.binary)
    plt.show()




