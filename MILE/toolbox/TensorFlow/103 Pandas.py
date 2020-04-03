#%% [markdown]
# # Estimadores
# Los features que utilizare
# ## Fase 1: deteccion de factura:
#  Feature: imagen de la factura 
#  Feature: nit de la imagen
# ## Fase 2: deteccion de la cuenta
#  Feature: El texto
#  Feature: otros parametros de numero y texto
#  #


#%% Importaciones de datos
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil

#%% Variables
tf.logging.set_verbosity(tf.logging.INFO) # nivel de verbosity en la ejecucion
OUTDIR = 'Toolbox/TensorFlow/model_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # Elimina la carpeta, inicia fresco cada vez

#%% Data disponible
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
LABEL = CSV_COLUMNS[0]

df_train = pd.read_csv('./Toolbox/TensorFlow/data/taxi-train.csv', header = None, names = CSV_COLUMNS)
df_valid = pd.read_csv('./Toolbox/TensorFlow/data/taxi-valid.csv', header = None, names = CSV_COLUMNS)
df_test = pd.read_csv('./Toolbox/TensorFlow/data/taxi-test.csv', header = None, names = CSV_COLUMNS)

#%% 1 Creacion de Features
# Aqui solo se nombra las columnas#
# Ademas existen: bucketized, embedding, crossed, categorical_with_hash
featcols = [
    tf.feature_column.numeric_column('sq_footage'),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'type', ['house', 'apt'])
        ]

#%% 2 Estimador del modelo a usar
model = tf.estimator.LinearRegressor(featcols, model_dir=OUTDIR)
# El DNN no funciona
# model = tf.estimator.DNNRegressor(
#                                     featcols,
#                                     hidden_units=[3, 2],
#                                     activation_fn=tf.nn.relu,
#                                     dropout=0.2,
#                                     optimizer='Adam'
#                                     )

#%% 3 Entrenamiento con DF
def train_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x = df,
        y = df['price'],
        batch_size = 128,
        num_epocs = 10,
        shuffle = True,
        queue_capacity=1000
        )
#%%
model.train(train_input_fn(df), steps=2000)

#%% 3 Entrenamiento con Numpy
def train_input_fn(sqft, prop_type, price):
    return tf.esttimator.inputs.numpy_input_fn(
        x = {'sq_footage': sqft, 'type': prop_type},
        y = price,
        batch_size = 128,
        num_epocs = 10,
        shuffle = True,
        queue_capacity=1000
    )

#%% 4 Predicciones
def predict_input_fn():
    features = {
        'sq_footage': [1500, 1800],
        'type': ['house', 'apt']}
    
    return features

predictions = model.predict(predict_input_fn)
print(next(predictions))
print(next(predictions))

#%%
# 4.1 Predicciones a partir de un modelo salvado
trained_model = tf.estimator.LinearRegressor(featcols, 'Toolbox/TensorFlow/model_trained')
predictions = trained_model.predict(predict_input_fn)
print(next(predictions))
print(next(predictions))

