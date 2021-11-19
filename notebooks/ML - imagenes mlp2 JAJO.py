#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 21:35:42 2020

@author: hdla
"""
#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pathlib as pl
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


#%%
#Visualizar una imagen
file = 'data/invoices/1/16.jpg'
plt.rcParams["figure.figsize"] = (10,10)
plt.imshow(
    tf.image.decode_jpeg(
        tf.io.read_file(file)))

#%%
ds = tf.data.TFRecordDataset('data/invoices_train.tfrecord')
ds

for e in ds.take(1):
    print(repr(e))
    
#%%
def parse_function(tensor_string):
    """ 
    Decode a serialized tensor
    That takes a image_dicts normalize and returns a tuple (x,y)
    Returns a x(image) to decode
    """
    feature_decode = {
        'image_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)}

    img_dic = tf.io.parse_single_example(tensor_string, feature_decode)
    
    label= img_dic['label']
#    image = tf.io.parse_tensor(serialized=img_dic["image_raw"], out_type=tf.uint8) # era tf.string
    image = img_dic['image_raw']

    return {'image':image, 'label':label}

#%%
dsi = ds.map(parse_function)
dsi

for e in dsi.take(1):
    print(plt.imshow(tf.image.decode_jpeg(e['image'])))

#%%

def preprocess_data1(image_dic):

    """  
    Function that takes a image_dic, normalized (divided by 255) and returns a tupe (x, y)
    """

    image, label = image_dic['image'], image_dic['label']

    image = tf.image.decode_jpeg(e['image'])
    image = tf.image.resize(image, size=[900, 700], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.reshape(image, [900*700*3])
    image = tf.math.divide(image, 255)
    
    return image, label
    
#%%
def preprocess_data(image_dic):
    """  
    Function that takes a image_dic, normalized (divided by 255) and returns a tupe (x, y)
    """

    image, label = image_dic['image'], image_dic['label']

    image = tf.image.decode_jpeg(image)
    image.set_shape([None, None, None]) #+
    image = tf.math.divide(image, 255)
    image = tf.image.resize(image, [224, 224])
    image = tf.reshape(image, [224*224*3])

    return image, label

#%%
dsf = dsi.map(preprocess_data1)
dsf

for e, y in dsf.take(10):
    print(repr(e))

#%%
for e, y in dsf.take(1):
    print(e.numpy())

#%%
inputs = tf.keras.Input(shape=(900*700*3), name='Hiraganas')
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.BatchNormalization(renorm=True)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.BatchNormalization(renorm=True)(x)
outputs = tf.keras.layers.Dense(9, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
# "sparse me sirve para clasificacion mutuamente excluyente un label --- multilabel"
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

#%%
model.summary()

#%%
dsff = dsf.shuffle(buffer_size=300).batch(32)
history = model.fit(dsff, epochs=60)