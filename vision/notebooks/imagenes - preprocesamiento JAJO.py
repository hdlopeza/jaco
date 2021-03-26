#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:35:42 2020

@author: @hdlopeza (Hernan Dario Lopez Archila, hernand.lopeza@gmail.com)

Se debe ejecutar en Agente/vision
referencias:
    https://towardsdatascience.com/working-with-tfrecords-and-tf-train-example-36d111b3ff4d
"""
#%%
import sys
sys.path += ['code']

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import utils

#%% Cambiar forma de ver las imagenes
plt.rcParams["figure.figsize"] = (10,10)


#%% Visualizar una imagen
file = 'data/1.jpg'
path = pl.Path('data/invoices')

dpi=150
alto=np.int(11*dpi)
ancho=np.int(8.5*dpi)

#%% Visualizar una imagen
plt.imshow(
    tf.image.decode_jpeg(
        tf.io.read_file(file)))

#%% Visualizar una imagen
img_1 = tf.io.read_file(file)
img_1 = tf.image.decode_jpeg(img_1)
img_1 = tf.image.resize(img_1 , size=[alto, ancho], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#img_1
plt.imshow(img_1.numpy())
#img_1.numpy  #2750,2125
#img_1 = tf.image.resize(img_1, size=[2750,2125])
#plt.imshow(img_1)
plt.imsave('example.jpeg', img_1[:,:,0], cmap=plt.cm.gray)

#%% Visualizar una imagen
img_1 = tf.io.read_file(file)
img_1 = tf.image.decode_jpeg(img_1)
img_1= tf.expand_dims(img_1, 0)
[plt.imshow(e) for e in img_1]

#%% Visualizar una imagen
img_1 = tf.io.read_file(file)
img_1 = tf.image.decode_jpeg(img_1)
plt.imshow(tf.keras.preprocessing.image.array_to_img(img_1))

#%% Visualizar una imagen
img_1 = tf.io.read_file(file)
img_1 = tf.image.decode_jpeg(img_1)
img_1 = tf.image.resize_with_crop_or_pad(img_1, target_height=900, target_width=700)
plt.imshow(img_1)

#%% Visualizar una imagen
img_1 = tf.io.read_file(file)
img_1 = tf.image.decode_jpeg(img_1)
img_1 = tf.image.resize_with_pad(img_1, target_height=alto, target_width=ancho, method="nearest")
plt.imshow(img_1)
plt.imsave('example.jpeg', img_1[:,:,0], cmap=plt.cm.gray)

#%% Visualizar una imagen
""" Usar keras para levantar las imagenes desde un directorio"""

img_1 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
img_1 = img_1.flow_from_directory(directory='data/invoices', class_mode='sparse', shuffle=True) #batch_size=10

sample, label = next(img_1)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
plotImages(sample[:5])
[print(e) for e in label]
sample[:5].shape
plt.imshow(sample[0])

#%% Visualizar una imagen
img_1 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
img_1 = img_1.flow_from_directory(directory='data/invoices', class_mode='sparse', shuffle=True, batch_size=10)
sample, label = next(img_1)
label.shape
label
sample.shape
sample[0].shape
plt.imsave('example.jpeg', sample[0])
sample[0]
plt.imshow(sample[0])
plt.imshow(np.array(sample[0]))

#%% Visualizar una imagen
plt.imshow(plt.imread(file))
plt.imsave('example.jpeg', plt.imread(file))

#%% Leer los archivos de un directorio
[print(e) for e in path.glob('*')]

list_ds = tf.data.Dataset.list_files(str(path/'*/*'))
for e in list_ds.take(50):
  print(e.numpy())

#%% Leer los archivos de un directorio
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
x = datagen.flow_from_directory(directory='data/invoices', class_mode='sparse', shuffle=True, batch_size=10, target_size=(alto, ancho))  #este metodo setea automaticamente a 256,256
sample, label = next(x)

plotImages(sample[:5])
plt.imshow(sample[2])
plt.imsave('example.jpeg', sample[2])
label
#%%  Leer un archivo tfrecords
ds = tf.data.TFRecordDataset('data/invoices_train.tfrecord')
ds

#%% Navegar por un dataset proveniente de un tfrecords
for e in ds.take(1):
    print(e) # print(repr(e))
# Se puede notar que la informacion esta codificada
    
#%% Decodificar la informacion
dsi = ds.map(utils._decode)
dsi

for e in dsi.take(1):
    print(e['image'], e['label'])
#    print(tf.image.decode_jpeg(e['image']))
#    print(plt.imshow(tf.image.decode_jpeg(e['image'])))
#    plt.imshow(tf.image.decode_jpeg(e['image']))

#%% Preprocesamiento de la informacion
def preprocess_data(img_dic):
    
#    image = tf.io.parse_tensor(img_dict["image"], tf.string)
#    image = tf.io.parse_tensor(serialized=img_dic["image"], out_type=tf.uint8) # era tf.string
    
    image = tf.image.decode_jpeg(img_dic["image"])

#    image.set_shape([None, None, None]) #+

    image = tf.image.resize(image, [alto, ancho])
#    image = tf.image.resize(image, size=[alto, ancho], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    image = tf.math.divide(image, 255)
    image = tf.reshape(image, [alto*ancho*3])

#    return dict(image = image, label = img_dic["label"])
    return image, img_dic['label']

#%%  Funciona con preprocess data
dsf = dsi.map(utils._preprocess_image)
#b = []
#for e in dsf.take(1):
#    b.append(e)
#    print(e)
#
#plt.imshow(np.array(b[0]['image']))
#plt.imsave('example.jpeg', np.array(b[0]['image']))
#plt.imsave('example.jpeg', e.numpy)
#plt.imsave('example.jpeg', a[:,:,0], cmap=plt.cm.gray)

#%%
#model1 = tf.keras.models.Sequential([
#  tf.keras.layers.Flatten(input_shape=(28, 28)),
#  tf.keras.layers.Dense(128, activation='relu'),
#  tf.keras.layers.Dropout(0.2),
#  tf.keras.layers.Dense(10, activation='softmax')
#])

#%%
inputs = tf.keras.Input(shape=(alto*ancho*3), name='Hiraganas')
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.BatchNormalization(renorm=True)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.BatchNormalization(renorm=True)(x)
outputs = tf.keras.layers.Dense(9, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
# "sparse me sirve para clasificacion mutuamente excluyente un label --- multilabel"
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%%
model.summary()

#%%
dsff = dsf.shuffle(buffer_size=64).batch(32)
history = model.fit(dsff, epochs=120)

#dmodel.save('Jaco.h5')

#"""
#model.fit(x_train, y_train, epochs=5)
#model.evaluate(x_test,  y_test, verbose=2)
#
#with tf.device('/device:gpu:0'):
#    model.fit(dsff, epochs=3)
#
#"""
#%%
#dsv = tf.data.TFRecordDataset('data/invoices_val.tfrecord')
#dsiv = dsv.map(ppTFRecords._decode)
#dsfv = dsiv.map(preprocess_data)
#test_loss, test_acc = model.evaluate(dsfv, verbose=2)
#
#print('\nTest accuracy:', test_acc)

#%% Split information in data sets

#DATASET_SIZE = 12
#
#train_size = int(0.5 * DATASET_SIZE)
#val_size = int(0.3 * DATASET_SIZE)
#test_size = int(0.2 * DATASET_SIZE)
#
#dataset = tf.data.Dataset.range(DATASET_SIZE) 
#ds_train = dataset.take(train_size)
#ds_test = dataset.skip(train_size)
#ds_val = ds_test.skip(test_size)
#ds_test = ds_test.take(test_size)
#list(ds_train.as_numpy_iterator())
#list(ds_val.as_numpy_iterator())
#list(ds_test.as_numpy_iterator())
