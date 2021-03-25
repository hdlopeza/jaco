""" 
Create a tfrecords from invoces dataset, using tf.dataset
author: @hdlopeza (Hernan Lopez Archila, hernand.lopeza@gmail.com)
"""
import utils_split
from utils import serialize_example
import numpy as np
import tensorflow as tf
import os

tf.compat.v1.enable_eager_execution()

# Variables
FOLDER = '/data/invoices'
DATA_DIR = '/data/'

# Entrega listado del listado unico de carpetas y 
# listado de carpetas dividido entre train/test/val
y, files = utils_split.ratio(folder=FOLDER)

# Split el listado entre train/test/val
files = np.array(files)


files_train = files[files[:,3]=='train']
files_test = files[files[:,3]=='test']
files_val = files[files[:,3]=='val']

ds_train = tf.data.Dataset.from_tensor_slices((files_train[:, 0], np.int_(files_train[:, 2])))
ds_test = tf.data.Dataset.from_tensor_slices((files_test[:, 0], np.int_(files_test[:, 2])))
ds_val = tf.data.Dataset.from_tensor_slices((files_val[:, 0], np.int_(files_val[:, 2])))


with tf.io.TFRecordWriter(os.path.join(DATA_DIR, 'invoices_train.tfrecord')) as writer:
    for image, label in ds_train:
        example = serialize_example(
            image=tf.image.decode_jpeg(
                tf.io.read_file(image)), 
            label=label)
        writer.write(example)

with tf.io.TFRecordWriter(os.path.join(DATA_DIR, 'invoices_test.tfrecord')) as writer:
    for image, label in ds_test:
        example = serialize_example(
            image=tf.image.decode_jpeg(
                tf.io.read_file(image)), 
            label=label)
        writer.write(example)

with tf.io.TFRecordWriter(os.path.join(DATA_DIR, 'invoices_val.tfrecord')) as writer:
    for image, label in ds_val:
        example = serialize_example(
            image=tf.image.decode_jpeg(
                tf.io.read_file(image)), 
            label=label)
        writer.write(example)
