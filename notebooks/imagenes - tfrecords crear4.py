""" 
Create a tfrecords from invoces dataset, using tf.dataset
author: @hdlopeza (Hernan Lopez Archila, hernand.lopeza@gmail.com)
"""
import sys
sys.path += ['code']

import utils_split
import utils
import numpy as np
import tensorflow as tf
import os

# Variables
FOLDER = '/vision/data/invoices'
DATA_DIR = '/vision/data'

# Entrega listado del listado unico de carpetas y 
# listado de carpetas dividido entre train/test/val
y, files = utils_split.ratio(folder=FOLDER)

# Split el listado entre train/test/val
files = np.array(files)

datasets =['val', 'test', 'train'] 

for ds in datasets:
    files_temp = files[files[:,3]==ds]
    ds_temp = tf.data.Dataset.from_tensor_slices((files_temp[:, 0], np.int_(files_temp[:, 2])))

    with tf.io.TFRecordWriter(os.path.join(DATA_DIR, 'invoices_' + ds + '.tfrecord')) as writer:
        for image, label in ds_temp:
            img = tf.io.read_file(image)
            example = utils._encode(image=img, label=label)
            writer.write(example)