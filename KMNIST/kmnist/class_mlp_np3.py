"""
keywords : Classification, MLP, DS Numpy, Keras-Sequential, Validation
This module that trains a smple MLP using numpy dataset
In this model tf.keras uses Input instead .layers.Input
Notice that input_shape do not uses a tuple,  don't know why the code work's because should be (28*28,)
author: @hdlopeza
"""
import numpy as np
import tensorflow as tf

# Dataset
with np.load('/data/kmnist-train-imgs.npz') as img_container, \
    np.load('/data/kmnist-train-labels.npz') as label_container:
    images_train = img_container['arr_0']
    labels_train = label_container['arr_0']

with np.load('/data/kmnist-test-imgs.npz') as img_container, \
    np.load('/data/kmnist-test-labels.npz') as label_container:
    images_test = img_container['arr_0']
    labels_test = label_container['arr_0']


assert images_train.shape[0]==labels_train.shape[0]
assert images_test.shape[0]==labels_test.shape[0]

images_flatten_train = np.array([_.reshape((28*28,)) for _ in images_train])
images_flaten_test = np.array([_.reshape((28*28,)) for _ in images_test])

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
    x=images_flatten_train,
    y=labels_train,
    batch_size=32,
    epochs=3,
    steps_per_epoch=200,
    validation_data=(images_flaten_test, labels_test))