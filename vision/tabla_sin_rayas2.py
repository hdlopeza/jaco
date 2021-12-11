
#%%
import os
import re
import cv2
from numpy.core.numeric import NaN
import pytesseract
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
plt.rcParams['figure.figsize'] = [30, 10]

import app_db as db


# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# https://github.com/UB-Mannheim/tesseract/wiki


def mostrar(image, boxes):
    for box in boxes:
        (x, y, w, h) = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 1)
    return plt.imshow(image)

#%%

# lista de entrada a la funcion
lista = [('8001304263', 0.84001195, 'data/8001304263_new.jpg'),
        ('8002126635', 0.6552293, 'data/8002126635_new.jpg'),
        ('8110213537', 0.8046723, 'data/2-029_new.jpg'),
        ('8001539937', 0.6190442, 'data/8001539937_new.jpg')]

# selecciona un registro y el nombre del archivo
lista = lista[3]
file = lista[2]

# abre la imagen y toma sus dimensiones
img = cv2.imread(os.path.join(file))
height, weight, _ = img.shape

# Busca en la base de datos la factura en cuestion 
# y trae los datos de todos los campos
record = db.busca_nit(int(lista[0]))
record = record[0].get('campos')

col = {}
for i in record:
    # crea la lista de columnas con ancho para la seccion de detalle
    if 'columna' in i.get('campo'):
        col[i.get('campo')] = [{
            "x_min": int(i.get("x_min")*weight), 
            "x_max": int(i.get("x_max")*weight)
            }]

    elif 'detalle' in i.get('campo'):
        y_min = int(i.get('y_min')*height)
        y_max = int(i.get('y_max')*height)
        x_min = int(0)
        x_max = int(i.get('x_max')*weight)
col

ERODE_ITERATIONS = 3
DILATE_ITERATIONS = 1
HIGHT_MAX = 20
HIGHT_MIN = 3
THRESHOLD_COLUMN = 30
THRESHOLD_ROW = 12

img = img[y_min:y_max , x_min:x_max] # Fraccion de la imagen
img_frame = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0, 0)
img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
img1 = cv2.bitwise_not(img)
struct = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
img1 = cv2.erode(img1, struct, iterations=ERODE_ITERATIONS)
img1 = cv2.dilate(img1, struct, anchor=(-1, -1), iterations=DILATE_ITERATIONS)
contours, hierarchy = cv2.findContours(img1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


mostrar(img_frame.copy(), [(0,0,0,0)])
dataframe = pytesseract.image_to_data(img_frame, output_type=pytesseract.Output.DATAFRAME, config='--oem 1 --psm 6')

dataframe = dataframe.dropna()
columnas = col

# Eliminar los datos NaN de cualquier columna
dataframe.loc[:, 'columna'] = None

# A cada fila o grupo de palabras encontrada dependiendo de los limites del template
# de las columnas del area de detalle, la clasifica
for i in columnas:
    x_min = columnas.get(i)[0].get('x_min')
    x_max = columnas.get(i)[0].get('x_max')
    columna_nombre = i

    for ii, row in dataframe.iterrows():
        xmin = row['left']
        xmax = xmin + row['width']

        if (xmin >= x_min) and (xmax <= x_max):
            dataframe.loc[ii, 'columna'] = columna_nombre
        else:
            pass

dataframe = dataframe.dropna()

# Define las filas y las ordena para tener un parametro de recorrido del
# dataframe
df_lineas = sorted(dataframe.loc[:, 'line_num'].unique())

# contenedor de todas las filas
todo = []

# Recorre cada una de las filas
for x in df_lineas:

    # Ordena por la posicion en el eje x(left) para agrupar de izquierda a derecha
    dfxlinea = dataframe[dataframe.loc[:, 'line_num']==x].sort_values(by=['left'])

    # Extracta los nombres de las columnas identificadas
    df_columnas = sorted(dfxlinea.loc[:, 'columna'].unique())

    # Contenedor de cada fila
    registro = {}
    for i in df_columnas:

        # Filtra los elementos comunes a cada columna porque pytesseract 
        # reconoce por grupos de palabras
        dfxlinea_columna = dfxlinea[dfxlinea.loc[:, 'columna']==i]

        # Concatena los textos de la misma linea y categoria
        text_final = ''
        for ii, row in dfxlinea_columna.iterrows():
            text_final = text_final + row['text'] + ' '

        registro[i] = (
            # 'left':  dfxlinea_columna.left.min(),
            # 'top': dfxlinea_columna.top.max(),
            # 'width': dfxlinea_columna.width.sum(),
            # 'height': dfxlinea_columna.height.sum(),
            # 'conf': dfxlinea_columna.conf.max(),
            text_final
            )
    todo.append(registro)
todo

dataframe = pd.DataFrame(todo).replace("[\n]|[\x0c]|[,]|[|]|[)({}]|[\]\[]", "", regex=True) #.to_dict('records')
dataframe = dataframe[dataframe['columna_valor'].notna()]
dataframe.to_dict('records')

# %%
