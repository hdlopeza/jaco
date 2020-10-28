"""
Modulo que contiene utils para la app to KMNIST 
author Google
"""
import tensorflow as tf

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

def serialize_example(image, label):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    image = tf.io.read_file(filename=image)
    image = tf.image.decode_jpeg(image)
    serlialized_image = tf.io.serialize_tensor(image).numpy() #Se convierte el tensor(image) en string y luego en numpy

    feature_encode = {
        'image_raw': _bytes_feature(serlialized_image),
        'label': _int64_feature(label.numpy())}

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_encode))
    return example_proto.SerializeToString()

def serialize_invoices(image, label):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    feature_encode = {
      'image_raw': _bytes_feature(image),
      'label': _int64_feature(label)
      }

    example_proto = tf.train.Example(
      features=tf.train.Features(
        feature=feature_encode
        ))

    return example_proto.SerializeToString()

def parse_tfrecords(tensor_string):
    """ 
    Decode a serialized tensor
    That takes a image_dicts normalize and returns a tuple (x,y)
    Returns a x(image) to decode
    """
    feature_decode = {
        'image_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)}

    img_dic = tf.io.parse_single_example(tensor_string, feature_decode)
    
    label = img_dic['label']
    image = tf.io.parse_tensor(serialized=img_dic["image_raw"], out_type=tf.uint8) # era tf.string

    return {'image': image, 'label': label}

def parse_invoices(tensor_string):
    """ 
    Decode a serialized tensor
    That takes a image_dicts normalize and returns a tuple (x,y)
    Returns a x(image) to decode
    """
    feature_decode = {
        'image_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)}

    img_dic = tf.io.parse_single_example(tensor_string, feature_decode)
    
    label= img_dic['label']
    image = tf.image.decode_jpeg(img_dic['image_raw'])

    return {'image': image, 'label': label}