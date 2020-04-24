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


