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
import itertools

#%% Variables
tf.logging.set_verbosity(tf.logging.INFO) # nivel de verbosity en la ejecucion
OUTDIR = 'Toolbox/TensorFlow/model_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # Elimina la carpeta, inicia fresco cada vez

#%% Hyperparametros
EPOCS = 10

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
def make_feature_cols():
    return [tf.feature_column.numeric_column(i) for i in FEATURES]

#%% 2 Estimador del modelo a usar
model = tf.estimator.LinearRegressor(
    feature_columns=make_feature_cols(), 
    model_dir=OUTDIR)

#%% 3.1 Pipeline data to Entrenamiento con DF
def make_train_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x = df,
        y = df[LABEL],
        batch_size = 128,
        num_epochs = EPOCS,
        shuffle = True,
        queue_capacity=1000
        )

#%% 3.1 Pipeline data to Evaluacion con DF
def make_eval_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x = df,
        y = df[LABEL],
        batch_size = 128,
        num_epochs = EPOCS,
        shuffle = False,
        queue_capacity=1000
        )

#%% Ejecucion del entrenamiento
model.train(
    input_fn=make_train_input_fn(df_train))

#%% Evaluar el modelo
def print_rmse(model, name, df):
  metrics = model.evaluate(input_fn = make_eval_input_fn(df))
  print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))
print_rmse(model, 'validation', df_valid)

#%% 4 Predicciones
def make_prediction_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x = df,
        y = None,
        batch_size = 128,
        shuffle = False,
        queue_capacity=1000
        )
predictions = model.predict(
    input_fn=make_prediction_input_fn(df_test))
for items in predictions:
    print(items)

#%%
# 4.1 Predicciones a partir de un modelo salvado
trained_model = tf.estimator.LinearRegressor(
    feature_columns=make_feature_cols(), 
    model_dir=OUTDIR)

predictions = trained_model.predict(
    input_fn=make_prediction_input_fn(df_test))

print([
    pred['predictions'][0] for pred in
    list(itertools.islice(predictions,5))])

#%% DNNRegression
tf.logging.set_verbosity(tf.logging.INFO)
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
model = tf.estimator.DNNRegressor(
    hidden_units = [32, 8, 2],
    feature_columns = make_feature_cols(), 
    model_dir = OUTDIR)
model.train(
    input_fn = make_train_input_fn(df_train))
print_rmse(model, 'validation', df_valid)


#%%
