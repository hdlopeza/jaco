""" 
This module create a model to recognize bill images
and predict the numer of NIT to each supplier
x are images of features
y are nit of each supplier or labels
author: @hdlopeza (Hernan Lopez Archila, hernand.lopeza@gmail.com)
 """
#%%
import sys
sys.path += ['code']

import tensorflow as tf
import utils

#%% Variables

#%% Bug Tensorflow
file = 'data/1.jpg'
tf.image.resize(
        tf.image.decode_jpeg(tf.io.read_file(file)),
        size=[utils.alto, utils.ancho], 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

#%%  Crear los dataset dese archivos tfrecord
ds_val = tf.data.TFRecordDataset('/vision/data/invoices_val.tfrecord')
ds_train = tf.data.TFRecordDataset('/vision/data/invoices_train.tfrecord')
ds_test = tf.data.TFRecordDataset('/vision/data/invoices_test.tfrecord')

ds_val = ds_val.map(utils._decode)
ds_train = ds_train.map(utils._decode)
ds_test = ds_test.map(utils._decode)

#%%  Preprocesar la informacion
ds_val = ds_val.map(utils._preprocess_image)
ds_train = ds_train.map(utils._preprocess_image)
ds_test = ds_test.map(utils._preprocess_image)

#%% Arquitectura de la MLP
inputs = tf.keras.Input(shape=(utils.alto*utils.ancho*3,), name='Hiraganas')
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.BatchNormalization(renorm=True)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.BatchNormalization(renorm=True)(x)
outputs = tf.keras.layers.Dense(9, activation='softmax')(x)  #probar a cambiar 9 por 210

model = tf.keras.Model(inputs=inputs, outputs=outputs)
# "sparse me sirve para clasificacion mutuamente excluyente un label --- multilabel"
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#%% Inicializacion de los datos

ds_train = ds_train.shuffle(buffer_size=64).batch(32)
ds_test = ds_test.batch(64)
ds_val = ds_val.batch(64)

history = model.fit(ds_train, epochs=35)
model.save('Jaco.h5')
print(history)
model.evaluate(ds_val)