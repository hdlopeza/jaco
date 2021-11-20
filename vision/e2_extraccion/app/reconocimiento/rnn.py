"""Modulo que realiza el reconocimiento de la factura
"""

#%%

import numpy as np

import tensorflow.compat.v2 as tf
import matplotlib.pyplot as plt

from object_detection.utils import label_map_util, visualization_utils

#%%

DIR_MODEL_EXPORTED = 'app/reconocimiento/model'
PATH_TO_SAVED_MODEL = DIR_MODEL_EXPORTED + "/saved_model"
LABEL_MAP_FILE = DIR_MODEL_EXPORTED + '/label_map.pbtxt'

# %%inicializacion en memoria del modelo

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_FILE, use_display_name=True)


# %%  Load saved model and build the detection function

def nit_imagen(FILE):


    # Cargar y preprocesar la imagen
    image_np = np.array(plt.imread(FILE))
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]


    # Realizar la deteccion de objetos en la imagen
    detections = detect_fn(input_tensor)

    # Extractar las detecciones
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return (category_index[detections['detection_classes'][0]]['name'], detections['detection_scores'][0], FILE)


# %%

# # Visualizar la imagen

# image_np_with_detections = image_np.copy()
# visualization_utils.visualize_boxes_and_labels_on_image_array(
#       image_np_with_detections,
#       detections['detection_boxes'],
#       detections['detection_classes'],
#       detections['detection_scores'],
#       category_index,
#       use_normalized_coordinates=True,
#       max_boxes_to_draw=200,
#       min_score_thresh=.15,
#       agnostic_mode=False)

# plt.figure(figsize= (20,10))
# plt.imshow(image_np_with_detections)
# plt.show()

