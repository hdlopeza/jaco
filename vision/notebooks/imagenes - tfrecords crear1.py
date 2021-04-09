""" 
Create a tfrecords from invices dataset using numpy
author: @hdlopeza (Hernan Lopez Archila, hernand.lopeza@gmail.com)
"""
import utils_split
from utils import serialize_example
import numpy as np
import tensorflow as tf
import os

tf.enable_eager_execution()

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


with tf.io.TFRecordWriter(os.path.join(DATA_DIR, 'invoices_train.tfrecord')) as writer:
    for invoice in files_train:
        example = serialize_example(
            image=tf.io.read_file(str(invoice[0])), 
            label=np.array(invoice[2]))
        writer.write(example)

with tf.io.TFRecordWriter(os.path.join(DATA_DIR, 'invoices_test.tfrecord')) as writer:
    for invoice in files_test:
        example = serialize_example(
            image=tf.io.read_file(str(invoice[0])), 
            label=np.array(invoice[2]))
        writer.write(example)

with tf.io.TFRecordWriter(os.path.join(DATA_DIR, 'invoices_val.tfrecord')) as writer:
    for invoice in files_val:
        example = serialize_example(
            image=tf.io.read_file(str(invoice[0])), 
            label=np.array(invoice[2]))
        writer.write(example)