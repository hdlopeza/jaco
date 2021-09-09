#%%
import os
import re
import cv2
import pytesseract
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# %%

def mostrar1(titulo, image):
    # Using cv2.imshow() method 
    # Displaying the image 
    cv2.imshow(titulo, image)
    
    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0) 
    
    #closing all open windows 
    cv2.destroyAllWindows() 

#%%

ARCHIVO = "test2.png"
in_file = os.path.join("data", ARCHIVO)
img = cv2.imread(in_file, 0)
img = cv2.medianBlur(img, 5)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)


mostrar1('prueba', img)

# %%
