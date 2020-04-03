"""
Create a tfrecords from kmnist dataset
author: @hdlopeza (Hernan Lopez, hernand.lopeza@gmail.com)
"""
import numpy as np
import os
import tensorflow as tf

from utils import serialize_example

tf.enable_eager_execution()


DATA_DIR = '/data/' # el directorio de nuestros datos
SEED = 1227

with np.load(os.path.join(DATA_DIR, 'kmnist-train-imgs.npz')) as img_container, \
    np.load(os.path.join(DATA_DIR, 'kmnist-train-labels.npz')) as label_container:

    images = img_container['arr_0']
    labels = label_container['arr_0']

    assert images.shape[0]==labels.shape[0], 'Images do not have certain labels'

kmnist_ds =  tf.data.Dataset.from_tensor_slices((images, labels))
kmnist_ds.shuffle(70000, seed=SEED)
kmnist_train_ds = kmnist_ds.skip(1000)
kmnist_dev_ds = kmnist_ds.take(1000)

with tf.io.TFRecordWriter(os.path.join(DATA_DIR, 'kmnist_dev.tfrecord')) as writer:
    for image, label in kmnist_dev_ds:
        example = serialize_example(image, label)
        writer.write(example)

with tf.io.TFRecordWriter(os.path.join(DATA_DIR, 'kmnist_train.tfrecord')) as writer:
    for image, label in kmnist_train_ds:
        example = serialize_example(image, label)
        writer.write(example)
