"""
Modulo que contiene utils para la app to Vision 
author Google
"""
#%%
import sys
sys.path += ['code']

import tensorflow as tf
import numpy as np

#%% Variables
dpi   = 150
alto  = np.int(11*dpi)
ancho = np.int(8.5*dpi)

#%% Funciones para serializacion de datos
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#%% Función de codificacion/decodificacion de la data

def _encode(image, label):
    """
    Sale un diccionario con los atributos en feature_encode
    Creates a tf.Example message ready to be written to a file.
    """
    
    #Preprocesamiento de informacion
#    image = tf.io.read_file(filename=image)
#    image = tf.image.decode_jpeg(image)
#    image = tf.io.serialize_tensor(image).numpy()
    
    feature_encode = {'image_raw'   : _bytes_feature(image),
                      'label'       : _int64_feature(label)}

    example_proto = tf.train.Example(features=tf.train.Features(
            feature=feature_encode))

    return example_proto.SerializeToString()

def _decode(tensor_string):
    """ 
    Decode a serialized tensor
    That takes a dicts and returns a tuple (x,y) where x(image) needs to decode
    """
    feature_decode = {'image_raw'   : tf.io.FixedLenFeature([], tf.string, default_value=''),
                      'label'       : tf.io.FixedLenFeature([], tf.int64, default_value=0)}

    img_dic = tf.io.parse_single_example(tensor_string, feature_decode)
    
    label = img_dic['label']
    image = img_dic['image_raw']

    return {'image': image, 'label': label}

#%% Funcion para preprocesamiento de las imagenes
def _preprocess_image(img_dic):
    """  
    Function that takes a image_dic, normalized (divided by 255) and returns a tupe (x, y)
    """

    # image, label = image_dic['image'], image_dic['label']

#    image = tf.io.parse_tensor(img_dict["image"], tf.string)
#    image = tf.io.parse_tensor(serialized=img_dic["image"], out_type=tf.uint8) # era tf.string
    
    image = tf.image.decode_jpeg(img_dic["image"])

#    image.set_shape([None, None, None]) #+

    image = tf.image.resize(image, [alto, ancho])
#    image = tf.image.resize(image, size=[alto, ancho], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    image = tf.math.divide(image, 255)
    image = tf.reshape(image, [alto*ancho*3])

#    return dict(image = image, label = img_dic["label"])
    return image, img_dic['label']
