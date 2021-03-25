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

import os
import tensorflow as tf
#import tensorflow_hub as tfhub
import utils

# Variables

# Crear los dataset dese archivos tfrecord
ds_val = tf.data.TFRecordDataset('/vision/data/invoices_val.tfrecord')
ds_train = tf.data.TFRecordDataset('/vision/data/invoices_train.tfrecord')
ds_test = tf.data.TFRecordDataset('/vision/data/invoices_test.tfrecord')

ds_val = ds_val.map(utils.parse_invoices)
ds_train = ds_train.map(utils.parse_invoices)
ds_test = ds_test.map(utils.parse_invoices)


# Preprocesar la informacion
def preprocess_data(image_dic):
    """  
    Function that takes a image_dic, normalized (divided by 255) and returns a tupe (x, y)
    """

    image, label = image_dic['image'], image_dic['label']

    # image = tf.image.decode_jpeg(image)
    image.set_shape([None, None, None]) #+
    image = tf.math.divide(image, 255)
    image = tf.image.resize(image, [224, 224])
    image = tf.reshape(image, [224*224*3])

    return image, label

ds_val = ds_val.map(preprocess_data)
ds_train = ds_train.map(preprocess_data)
ds_test = ds_test.map(preprocess_data)

inputs = tf.keras.Input(shape=(224*224*3,), name='Hiraganas')
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.BatchNormalization(renorm=True)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.BatchNormalization(renorm=True)(x)
outputs = tf.keras.layers.Dense(210, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
# "sparse me sirve para clasificacion mutuamente excluyente un label --- multilabel"
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

#ds_train = ds_train.shuffle(buffer_size=300).batch(32)
#ds_test = ds_test.batch(64)
#ds_val = ds_val.batch(64)

history = model.fit(ds_train, epochs=3)
#print(history)
#model.evaluate(ds_val)