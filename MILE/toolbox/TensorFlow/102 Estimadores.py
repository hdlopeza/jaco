#%% [markdown]
# # Estimadores
# Los features que utilizare
# Fase 1: deteccion de factura:
#  la imagen de la factura y el nit de la imagen,
#  ?cuantas imagenes por nit necesito?
# Fase 2: deteccion de la cuenta
#  El texto y otros parametros de numero y texto, es un
#  reto de clasificacion#


#%%
import tensorflow as tf

#%% 1 Creacion de Features
# Ademas existen: bucketized, embedding, crossed, categorical_with_hash
featcols = [
    tf.feature_column.numeric_column('sq_footage'),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'type', ['house', 'apt'])
        ]

#%% 2 Estimador del modelo a usar
model = tf.estimator.LinearRegressor(featcols, 'Toolbox/TensorFlow/model_trained')
# El DNN no funciona
# model = tf.estimator.DNNRegressor(
#                                     featcols,
#                                     hidden_units=[3, 2],
#                                     activation_fn=tf.nn.relu,
#                                     dropout=0.2,
#                                     optimizer='Adam'
#                                     )

#%% 3 Entrenamiento
def train_input_fn():
    features = {
        'sq_footage': [1000, 2000, 3000, 1000, 2000, 3000],
        'type': ['house', 'house', 'house', 'apt', 'apt', 'apt']}

    labels = [500, 1000, 1500, 700, 1300, 1900]

    return features, labels

model.train(train_input_fn, steps=2000)

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

