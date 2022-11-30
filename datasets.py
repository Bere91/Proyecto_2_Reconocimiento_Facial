import tensorflow as tf
import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop
import datetime
from tensorflow.keras.models import load_model
import os
from turtle import pd
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Fotos de CelebA

df_no = pd.read_csv('attr_prepceleba.txt', sep=' ', header = None)
df_no= df_no.replace([-1],0)
df_no = df_no.replace([1], 0)
list=[x for x in range(2, 41)]
df_no = df_no.drop(df_no.columns[list], axis='columns')
df_no = df_no.drop(range(5000, 202599, 1),axis=0)
archivos_no = tf.data.Dataset.from_tensor_slices(df_no[0])


atributos_celeba = tf.data.Dataset.from_tensor_slices(df_no.iloc[:,1:].to_numpy())
data_celeba = tf.data.Dataset.zip((archivos_no, atributos_celeba))

direccion_de_imagenes = "C:/Users/Berec/Downloads/Redes/img_align_celeba/img_align_celeba/"
def process_file(file_name, attributes):
    imagen = tf.io.read_file(direccion_de_imagenes + file_name)
    imagen = tf.image.decode_jpeg(imagen, channels=3)
    imagen = tf.image.resize(imagen, [192, 192])
    imagen /= 255.0
    return imagen, attributes

imagenes_celeba = data_celeba.map(process_file)

#Fotos de mi rostrpo
nombres_archivos_yo = os.listdir("C:/Users/Berec/Downloads/Proyecto_2_Reconocimiento_Facial/fotosaumentadas")
atributos_fm = [1] * 6642
df_me = pd.DataFrame((zip(nombres_archivos_yo, atributos_fm)))

archivos_yo = tf.data.Dataset.from_tensor_slices(df_me[0])
atributos_fm = tf.data.Dataset.from_tensor_slices(df_me.iloc[:,1:].to_numpy())
data_si = tf.data.Dataset.zip((archivos_yo, atributos_fm))

direccion_de_imagenes = "C:/Users/Berec/Downloads/Proyecto_2_Reconocimiento_Facial/fotosaumentadas/"
def process_file(file_name, attributes):
    image = tf.io.read_file(direccion_de_imagenes + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, attributes

imagenes_si = data_si.map(process_file)



concat_ds = imagenes_si.concatenate(imagenes_celeba)

def partir_dataset(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=1000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = partir_dataset(concat_ds, 11642)



t_step=9313//32 
v_step=1164//32 

train_ds = train_ds.batch(32).repeat(8)
val_ds = val_ds.batch(32).repeat(8)

modelo_pre_entrenado=load_model('red2.h5')
model = tf.keras.Sequential()
model.add(modelo_pre_entrenado.layers[0])
model.add(modelo_pre_entrenado.layers[1])
model.add(modelo_pre_entrenado.layers[2])
model.add(modelo_pre_entrenado.layers[3])
model.add(modelo_pre_entrenado.layers[4])
model.add(modelo_pre_entrenado.layers[5])
model.add(modelo_pre_entrenado.layers[6])
model.add(modelo_pre_entrenado.layers[7])
model.add(modelo_pre_entrenado.layers[8])
model.add(modelo_pre_entrenado.layers[9])
model.add(Dense(1))
for layer in model.layers[:10]:
    layer.trainable = False
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

print("Logs:")
print(log_dir)
print("__________")
model.fit(
                train_ds,
                steps_per_epoch=t_step,
                epochs=8,
                validation_data=val_ds,
                validation_steps=v_step,
                
                )


model.save("red3.h5")