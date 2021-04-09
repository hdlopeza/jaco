""" 
This module create a model to recognize bill images
and predict the numer of NIT to each supplier
x are images of features
y are nit of each supplier or labels
author: @hdlopeza (Hernan Lopez Archila, hernand.lopeza@gmail.com)
 """
import os
import tensorflow as tf
import tensorflow_hub as tfhub
import utils

# Variables
TFHUB_CACHE_DIR = '/model/tfhub'
os.environ['TFHUB_CACHE_DIR'] = TFHUB_CACHE_DIR
#os.makedirs(TFHUB_CACHE_DIR, exist_ok=True)

# Importacion del modulo de transfer learning para imagenes
def transfer_model():
    if not os.path.isdir(TFHUB_CACHE_DIR):
        tfh_module = tfhub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2")
        return tfh_module
        # tfh_module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/feature_vector/2")
        # tfh_module = hub.Module("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/1")
        # tfh_module = tfhub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2")
    else:
        tfh_module = tfhub.Module(os.path.join(TFHUB_CACHE_DIR, 'adfe0cf8d843e3588bfb9602e32a718b12212904'))
        return tfh_module

# Trae de tfhub el tamaño de la imagen esperada segun el modulo cargado,
# _IMAGESIZE = tfhub.get_expected_image_size(transfer_model())

# Funcion para preprocesar la informacion
def preprocess_data(image_dic):
    """  
    Function that takes a image_dic, normalized (divided by 255) and returns a tupe (x, y)
    """

    image, label = image_dic['image'], image_dic['label']

    # image = tf.image.decode_jpeg(image)
    image.set_shape([None, None, None]) #+
    image = tf.math.divide(image, 255)
    image = tf.image.resize(image, [224, 224])
    # image = tf.reshape(image, [224*224*3])

    return image, label

# Funciones de paso de datos a la red
def inputfn_train():
    # Modifica los dataset con el preprocesamiento de la informacion
    ds = tf.data.TFRecordDataset('/data/invoices_train.tfrecord')
    ds = ds.map(utils.parse_tfrecords)
    ds = ds.map(preprocess_data)
    ds = ds.shuffle(buffer_size=300)
    ds = ds.repeat()
    ds = ds.batch(32)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def inputfn_eval():
    # Modifica los dataset con el preprocesamiento de la informacion
    ds = tf.data.TFRecordDataset('/data/invoices_val.tfrecord')
    ds = ds.map(utils.parse_tfrecords)
    ds = ds.map(preprocess_data)
    # ds = ds.shuffle(buffer_size=300)
    ds = ds.repeat(1)
    ds = ds.batch(64)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def inputfn_pred():
    pass

def inputfn_serving():
    pass


# Definicion del modelo 
def model_fn(features, labels, mode):
    # Transfer learning
    tfh_module = transfer_model()

    transformed_features = tfh_module(features)
    logits = tf.layers.dense(transformed_features, 210) #La cantidad de proveedores
    probabilities = tf.nn.softmax(logits)

    if (mode != tf.estimator.ModeKeys.PREDICT):
        one_hot_labels = tf.one_hot(labels, 210)
        loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
        optimizer = tf.train.AdamOptimizer()

        train_op = tf.contrib.training.create_train_op(loss, optimizer)
        accuracy = tf.metrics.accuracy(labels, tf.argmax(probabilities, axis=-1))
        metrics = {'acc': accuracy}
    else:
        loss = optimizer = train_op = metrics = None

    model =  tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions={
            'proba': probabilities,
            'class': tf.argmax(probabilities, axis=-1)},
        eval_metric_ops=metrics)
    
    return model

# Definicion de la configuracion de entrenamiento
run_config = tf.estimator.RunConfig(
    model_dir='./models/tlearning',
    save_summary_steps=10,
    log_step_count_steps=10)

model = tf.estimator.Estimator(
    model_fn=model_fn,
    config=run_config)

train_spec = tf.estimator.TrainSpec(
    input_fn=inputfn_train,
    max_steps=150)

eval_spec = tf.estimator.EvalSpec(
    input_fn=inputfn_eval)

out = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
print(out)





