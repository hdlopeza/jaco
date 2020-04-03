"""
Module that trains a smple MLP using tfrecors
author: @hdlopeza
"""

import tensorflow as tf
from utils import  parse_tfrecords
tf.enable_eager_execution()

train_ds = tf.data.TFRecordDataset('/data/kmnist_train.tfrecord')
dev_ds = tf.data.TFRecordDataset('/data/kmnist_dev.tfrecord')

train_ds = train_ds.map(parse_tfrecords)
dev_ds = dev_ds.map(parse_tfrecords)

def preprocess_data(image_dic):
    "function that takes a image_dicts normalize and returns a tuple (x,y)"

    image, label = image_dic['image'], image_dic['label']
    image = tf.dtypes.cast(image, tf.float32)
    image_norm = tf.math.divide(image, 255)
    image_norm = tf.reshape(image_norm,shape=[28*28])

    return (image_norm, label)

train_ds = train_ds.map(preprocess_data)
dev_ds = dev_ds.map(preprocess_data)

inputs = tf.keras.Input(shape=(28*28,), name='Hiraganas')
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.BatchNormalization(renorm=True)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.BatchNormalization(renorm=True)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
"sparse me sirve para clasificacion mutuamente excluyente un label --- multilabel"
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

train_ds = train_ds.shuffle(buffer_size=30000).batch(32)
dev_ds = dev_ds.batch(64)

history = model.fit(train_ds, epochs=3)

model.evaluate(dev_ds)
