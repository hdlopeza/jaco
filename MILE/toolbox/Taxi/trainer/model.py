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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#%% Importaciones de datos
import tensorflow as tf

#Variables
tf.logging.set_verbosity(tf.logging.INFO) # nivel de verbosity en la ejecucion

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
    # use tf.data.Dataset.map to apply one to one  transformations (here: text line -> feature list)
    dataset = textlines_dataset.map(decode_csv)

    if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = None #Loop indefinidamente
        dataset = dataset.shuffle(buffer_size = 10 * batch_size)
    else:
        num_epochs = 1 #end-of-input after this
    
    dataset = dataset.repeat(num_epochs).batch(batch_size = batch_size)
    return dataset.make_one_shot_iterator().get_next()
 return _input_fn

#% 1.1 Pipeline data to Entrenamiento
def train_input_fn(train_filename, train_batch):
    return read_dataset(
        filename = train_filename, 
        batch_size = train_batch, 
        mode = tf.estimator.ModeKeys.TRAIN)

#% 1.1 Pipeline data to Evaluacion
def eval_input_fn(eval_filename, eval_batch):
    return read_dataset(
        filename = eval_filename, 
        batch_size = eval_batch,
        mode = tf.estimator.ModeKeys.EVAL)

#% 1.1 Pipeline data to Predicciones
def prediction_input_fn():
    return read_dataset(
        './Toolbox/TensorFlow/data/taxi-test.csv', 
        mode = tf.estimator.ModeKeys.EVAL)

#% 1.1 Pipeline data to serving
def serving_input_fn():
  # Defines the expected shape of the JSON feed that the model
  # will receive once deployed behind a REST API in production.
  def _input_fn():
    json_feature_placeholders = {
        'pickuplon' : tf.placeholder(tf.float32, [None]),
        'pickuplat' : tf.placeholder(tf.float32, [None]),
        'dropofflat' : tf.placeholder(tf.float32, [None]),
        'dropofflon' : tf.placeholder(tf.float32, [None]),
        'passengers' : tf.placeholder(tf.float32, [None]),}
    
    # feature_placeholders = {
    #     column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS
    # }    
    
    # You can transforma data here from the input format to the format expected by your model.
    features = json_feature_placeholders # no transformation needed
    
    return tf.estimator.export.ServingInputReceiver(features, json_feature_placeholders)
  return _input_fn

#% 2 Creacion de Features
def make_feature_cols():
    # Aqui solo se nombra las columnas
    # Ademas existen: bucketized, embedding, crossed, categorical_with_hash
    return [tf.feature_column.numeric_column(i) for i in CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]]

def train_and_evaluate(args):
    #% 3 Estimador del modelo a usar
    run_config = tf.estimator.RunConfig(
        model_dir = args['output_dir'], # Ouput directory for checkpoint
        save_summary_steps = 100) #, save_checkpoints_step = 100

    # Define los aspectos del modelo
    model = tf.estimator.LinearRegressor(
        feature_columns = make_feature_cols(), 
        config = run_config)

    # Define los aspectos del entrenamiento y la entrada de datos
    train_spec = tf.estimator.TrainSpec(
        input_fn = train_input_fn(train_filename = args['train_data_paths'], train_batch = args['train_batch_size']), 
        max_steps = args['train_steps'])

    # Define los aspectos del uso en produccion con ML Engine
    export_latest = tf.estimator.LatestExporter(
        'exporter', #folder to export
        serving_input_receiver_fn = serving_input_fn())

    # Define los aspectos de la evaluacion, cada cuanto se graba para tensorboard y la entrada de datos
    eval_spec = tf.estimator.EvalSpec(
        input_fn = eval_input_fn(eval_filename = args['eval_data_paths'], eval_batch = 500),
        steps = None,
        start_delay_secs = args['eval_delay_secs'], # start evaluating after N seconds
        throttle_secs = args['throttle_secs'], # evaluate every N seconds
        exporters = export_latest)

    # Ejecuta el modelo
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
