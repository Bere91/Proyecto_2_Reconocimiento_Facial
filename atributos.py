import tensorflow as tf
#import tensorflow_datasets as tfds
import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop
import pathlib
import datetime
import os
from turtle import pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#Este archivo lo saqué de la explicación que dio en la clase de red convolucional, sobre cómo cargar las imágenes de la
#base de datos en un dataset. 
'''with open('list_attr_celeba.txt', 'r') as f:
    print("skipping : " + f.readline())
    print("skipping headers : " + f.readline())
    with open('attr_prepceleba.txt' , 'w') as newf:
        for line in f:
            new_line = ' '.join(line.split())
            newf.write(new_line)
            newf.write('\n')'''

df = pd.read_csv('attr_prepceleba.txt', sep=' ', header = None)
df= df.replace([-1],0)
print('----------')
#print(df[0].head())
files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files, attributes))


path_to_images = 'C:/Users/Berec/Downloads/Redes/img_align_celeba/img_align_celeba/'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, attributes

labeled_images = data.map(process_file)


#En clase se imprimieron alguans imágenes, esto para corroborar que se hubieran cargado apropiadamente en un dataset. En esta ocasión
#omitiremos este paso.
'''for image, attri in labeled_images.take(2):
    plt.imshow(image)
    plt.show()
'''
#Encontré esta función para dividir el dataset en 3 partes. 
def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    
    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = get_dataset_partitions_tf(labeled_images, 202599)



train_ds = train_ds.batch(32).repeat(4)
val_ds = val_ds.batch(32).repeat(4)
tstep = 2000
vstep = 1000


model = Sequential()

model.add(Conv2D(10, (3, 3), input_shape=(192,192,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#Se pone una capa densa de 40 neuronas, una por cada atributo.
model.add(Dense(40))
model.add(Activation('sigmoid'))
opt = keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")



print("Logs:")
print(log_dir)
print("__________")



model.fit(
                train_ds,
                steps_per_epoch=tstep,
                epochs=2,
                validation_data=val_ds,
                validation_steps=vstep,
                #callbacks=[tbCallBack]
                )


model.save("red.h5")







