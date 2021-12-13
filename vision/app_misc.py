
import cv2
from matplotlib import pyplot as plt


def mostrar(image, boxes):
    for box in boxes:
        (x, y, w, h) = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 1)
    return plt.imshow(image)