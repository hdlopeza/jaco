#%%

import cv2
import os
from cv2 import dnn


# %%

net = cv2.dnn.readNetFromTensorFlow(
    os.path.join('dnn', 'saved_model.pb'),
    os.path.join('dnn', 'tfhub_module.pb')
)

# %%

net = dnn.readNetFromTensorflow(os.path.join('dnn', 'tfhub_module.pb'),
                                os.path.join('dnn', 'saved_model.pb') 
                                )

# %%
