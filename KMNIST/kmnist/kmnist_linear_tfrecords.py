"""
Module who uses TFrecords to train a ML Lineal
author: @hdlopeza (Hernan Lopez, hernand.lopeza@gmail.com) 
"""
import os
import tensorflow as tf

def main():
    DATA_DIR = '/data/' # el directorio de nuestros datos

    
    # def parse_features(kmnist_ds):
    #     def _transform_img(img_dict):
    #         image = tf.io.parse_tensor(img_dict["image_raw"], tf.uint8)
    #         return dict(
    #             image = tf.reshape(image, [28, 28]) ,
    #             label = img_dict["label"]
    #         )

    kmnist_ds = tf.data.TFRecordDataset(os.path.join(DATA_DIR, 'kmnist_dev.tfrecord'))

    for ds in kmnist_ds.take(10):
        print(repr(ds))

    #img_dict = {
    #    'image_raw': tf.io.FixedLenFeature([], tf.int64),
    #    'label': tf.io.FixedLenFeature([], tf.int64)
    #    }
    
    #def _parser(example_proto):
    #    return tf.io.parse_single_example(example_proto, img_dict)
    
    #raw_data_parsed = kmnist_ds.map(_parser)
    #raw_data_parsed = parse_features(kmnist_ds)


    # ## Model code

    # images_flatten = tf.keras.layers.Flatten()(images)
    # labels_one_hot = tf.keras.utils.to_categorical(labels)

    # linear_model = tf.keras.Sequential()
    # linear_model.add(tf.keras.layers.InputLayer(input_shape=(28*28,)))
    # linear_model.add(tf.keras.layers.Dense(units=10))
    # linear_model.add(tf.keras.layers.Softmax())

    # linear_model.summary()

    # linear_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy())
    # linear_model.fit(x=images_flatten, y=labels_one_hot, batch_size=32, epochs=3, steps_per_epoch=200)

    # return linear_model

if __name__ == "__main__":
    main()