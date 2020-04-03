"""
keywords_train : Classification, MLP, DS_train Numpy, Keras-Sequential, No validation
This module that trains a smple MLP using tf.dataset
In this model tf.keras uses Input instead .layers.Input
Notice that input_shape do not uses a tuple,  don't know why the code work's because should be (28*28,)
author: @hdlopeza
"""
import numpy as np
import tensorflow as tf

# Dataset
with np.load('/data/kmnist-train-imgs.npz') as img_container, \
    np.load('/data/kmnist-train-labels.npz') as label_container:
    images = img_container['arr_0']
    labels = label_container['arr_0']

assert images.shape[0]==labels.shape[0]

images_flatten = tf.keras.layers.Flatten()(images)

ds_train = tf.data.Dataset.from_tensor_slices((images_flatten, labels)).batch(32).repeat()

# Model construct
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(28*28), name='Hiraganas'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.BatchNormalization(renorm=True))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.BatchNormalization(renorm=True))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

# Model learning process
model.compile(
    optimizer='Adam',
    loss='sparse_categorical_crossentropy', #sparse me sirve para clasificacion mutuamente excluyente un label --- multilabel
    metrics=['acc'])

# Model train
model.fit(
    ds_train,
    epochs=3,
    steps_per_epoch=200)