
import cv2
import os
from matplotlib import pyplot as plt


def mostrar(image, boxes):
    for box in boxes:
        (x, y, w, h) = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 1)
    return plt.imshow(image)

def recorre_carpeta(folder):
    """Recorre una carpeta y regresa un generador para usar de uno en uno

    Args:
        folder ([type]): [description]

    Yields:
        [type]: [description]
    """

    for (path, dirs, files) in os.walk(folder):
        for file in files:
            yield folder, file