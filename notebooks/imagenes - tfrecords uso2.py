#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 17:09:05 2020

Crear TfRecords

@author:  @hdlopeza (Hernan Dario Lopez Archila, hernand.lopeza@gmail.com)
"""

#%%
import sys
sys.path += ['code']

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import utils_split
import numpy as np
import os

#%% Realizar listado de imagenes
FOLDER = 'data/invoices'
DATA_DIR = 'data'

y, files = utils_split.ratio(folder=FOLDER)
files = np.array(files)

#%% Funciones basicas para serializar (codificar) los datos
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#%% Función de codificacion de la data
def _encode(image, label):
    """
    Sale un diccionario con los atributos en feature_encode
    Creates a tf.Example message ready to be written to a file.
    """
    
    #Preprocesamiento de informacion
#    image = tf.io.read_file(filename=image)
#    image = tf.image.decode_jpeg(image)
#    image = tf.io.serialize_tensor(image).numpy()
    
    feature_encode = {'image_raw'   : _bytes_feature(image),
                      'label'       : _int64_feature(label)}

    example_proto = tf.train.Example(features=tf.train.Features(
            feature=feature_encode))

    return example_proto.SerializeToString()

def _decode(tensor_string):
    """ 
    Decode a serialized tensor
    That takes a dicts and returns a tuple (x,y) where x(image) needs to decode
    """
    feature_decode = {'image_raw'   : tf.io.FixedLenFeature([], tf.string, default_value=''),
                      'label'       : tf.io.FixedLenFeature([], tf.int64, default_value=0)}

    img_dic = tf.io.parse_single_example(tensor_string, feature_decode)
    
    label = img_dic['label']
    image = img_dic['image_raw']

    return {'image': image, 'label': label}

#%% Realiza separacion entre los grupos de trabajo
datasets =['val', 'test', 'train']

for ds in datasets:
    files_temp = files[files[:,3]==ds]
    ds_temp = tf.data.Dataset.from_tensor_slices((files_temp[:, 0], np.int_(files_temp[:, 2])))

    with tf.io.TFRecordWriter(os.path.join(DATA_DIR, 'invoices_' + ds + '.tfrecord')) as writer:
        for image, label in ds_temp:
            img = tf.io.read_file(image)
            example = _encode(image=img, label=label)
            writer.write(example)


# %%
