"""
This model contains a implentation for a linear model for our data
author: @hdlopeza
"""

import tensorflow as tf
import numpy as np

def main():
    with np.load('/data/kmnist-train-imgs.npz') as img_container, \
         np.load('/data/kmnist-train-labels.npz') as label_container:
            images = img_container['arr_0']
            labels = label_container['arr_0']
    
    assert images.shape[0]==labels.shape[0], 'Images do not have certain labels'

    ## Model code

    images_flatten = tf.keras.layers.Flatten()(images)
    labels_one_hot = tf.keras.utils.to_categorical(labels)

    linear_model = tf.keras.Sequential()
    linear_model.add(tf.keras.layers.InputLayer(input_shape=(28*28,)))
    linear_model.add(tf.keras.layers.Dense(units=10))
    linear_model.add(tf.keras.layers.Softmax())

    linear_model.summary()

    linear_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy())
    linear_model.fit(x=images_flatten, y=labels_one_hot, batch_size=32, epochs=3, steps_per_epoch=200)

    return linear_model

if __name__ == "__main__":
    main()
