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
t,im = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
im = 255-im

mostrar1('prueba', im)

# %%
kernel_len = np.array(img).shape[1]//100
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# %%
