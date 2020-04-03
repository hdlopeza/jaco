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
import numpy as np
import shutil
import itertools

#Variables
tf.logging.set_verbosity(tf.logging.WARN) # nivel de verbosity en la ejecucion
OUTDIR = 'Toolbox/TensorFlow/model_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # Elimina la carpeta, inicia fresco cada vez

#Hyperparametros
EPOCS = 10

# Data disponible
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]
LABEL_COLUMN = CSV_COLUMNS[0]

#% 1 Pipe Data en lotes
def read_dataset(filename, mode, batch_size = 512):
 def _input_fn():
    def decode_csv(row):
        columns = tf.decode_csv(row, record_defaults = DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        features.pop('key') # Se descarta porque no es un atributo a tomar
        label = features.pop(LABEL_COLUMN)
        return features, label
    
    # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
    filenames_dataset = tf.data.Dataset.list_files(filename)
    
    #Lee cada linea desde cada archivo
    # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
    textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
    
    # Parse text lines as comma-separated values (CSV)
    # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)
    dataset = textlines_dataset.map(decode_csv)

    if mode == tf.estimator.ModeKeys.TRAIN:
        EPOCS = None #Loop indefinidamente
        dataset = dataset.shuffle(buffer_size = 10 * batch_size)
    else:
        EPOCS = 1 #end-of-input after this
    
    dataset = dataset.repeat(EPOCS).batch(batch_size = batch_size)
    
    return dataset.make_one_shot_iterator().get_next()
 return _input_fn

#% 1.1 Pipeline data to Entrenamiento
def make_train_input_fn():
    return read_dataset('./Toolbox/TensorFlow/data/taxi-train.csv', mode = tf.estimator.ModeKeys.TRAIN)

#% 1.1 Pipeline data to Evaluacion
def make_eval_input_fn():
    return read_dataset('./Toolbox/TensorFlow/data/taxi-valid.csv', mode = tf.estimator.ModeKeys.EVAL)

#% 1.1 Pipeline data to Predicciones
def make_prediction_input_fn():
    return read_dataset('./Toolbox/TensorFlow/data/taxi-test.csv', mode = tf.estimator.ModeKeys.EVAL)

#% 2 Creacion de Features
# Aqui solo se nombra las columnas#
# Ademas existen: bucketized, embedding, crossed, categorical_with_hash
def make_feature_cols():
    return [tf.feature_column.numeric_column(i) for i in CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]]

#% 3 Estimador del modelo a usar
model = tf.estimator.LinearRegressor(feature_columns=make_feature_cols(), model_dir=OUTDIR)

#% 4 Ejecucion del entrenamiento
model.train(input_fn = make_train_input_fn(), steps = 200)

#%% 5  Evaluar el modelo
def print_rmse(model, name):
  metrics = model.evaluate(input_fn = make_eval_input_fn(), steps = None)
  print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['average_loss'])))
print_rmse(model, 'validation')


#%% 6
predictions = model.predict(input_fn = make_prediction_input_fn())
for items in predictions:
    print(items)

#%% 6.1
# 4.1 Predicciones a partir de un modelo salvado
trained_model = tf.estimator.LinearRegressor(feature_columns = make_feature_cols(), model_dir = OUTDIR)

predictions = trained_model.predict(input_fn = make_prediction_input_fn())

print([
    pred['predictions'][0] for pred in
    list(itertools.islice(predictions,5))])

#%% 7 DNNRegression Se demora mucho
tf.logging.set_verbosity(tf.logging.WARN)
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time

model = tf.estimator.DNNRegressor(
    hidden_units = [32, 8, 2],
    feature_columns = make_feature_cols(), 
    model_dir = OUTDIR)
model.train(input_fn = make_train_input_fn())
print_rmse(model, 'validation')
