""" 
keywords : Classification, MLP, DS TfRecords, Keras-Sequential, No validation
This model implements a linear model in a single layer perceptron with np dataset.
In this model tf.keras uses Input instead .layers.Input
Notice that input_shape do not uses a tuple,  don't know why the code work's because should be (28*28,)
author: @hdlopeza
"""
import numpy as np
import tensorflow as tf

with np.load('/data/kmnist-train-imgs.npz') as img_container, \
    np.load('/data/kmnist-train-labels.npz') as label_container:
    images = img_container['arr_0']
    labels = label_container['arr_0']

assert images.shape[0] == labels.shape[0]

images_flatten = tf.keras.layers.Flatten()(images)
labels_one_hot = tf.keras.utils.to_categorical(labels)

linear_model = tf.keras.Sequential()
linear_model.add(tf.keras.Input(shape=(28*28))) #tf.keras.layers.InputLayer(input_shape=(28*28)) 
linear_model.add(tf.keras.layers.Dense(units=10))
linear_model.add(tf.keras.layers.Softmax())

linear_model.summary()

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy())

linear_model.fit(
    x=images_flatten,
    y=labels_one_hot,
    batch_size=32,
    epochs=3,
    steps_per_epoch=200)