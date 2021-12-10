
#%%
import os
import re
import cv2
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

lista = [('8001304263', 0.84001195, 'data/8001304263_new.jpg'),
        ('8002126635', 0.6552293, 'data/8002126635_new.jpg'),
        ('8110213537', 0.8046723, 'data/2-029_new.jpg'),
        ('8001539937', 0.6190442, 'data/8001539937_new.jpg')]

lista = lista[2]

record = db.busca_nit(int(lista[0]))
record = record[0].get('campos')
record = record[0]
file = lista[2]

ERODE_ITERATIONS = 3
DILATE_ITERATIONS = 1
HIGHT_MAX = 20
HIGHT_MIN = 3
THRESHOLD_COLUMN = 30
THRESHOLD_ROW = 12

img = cv2.imread(os.path.join(file))
height, weight, _ = img.shape

y_min = int(record.get('y_min')* height)
y_max = int(record.get('y_max')* height)
x_min = int(record.get('x_min')* weight)
x_max = int(record.get('x_max')* weight)

img = img[y_min+70:y_max-800 , x_min+150:x_max] # Fraccion de la imagen
img_frame = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0, 0)
img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
img1 = cv2.bitwise_not(img)
struct = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
img1 = cv2.erode(img1, struct, iterations=ERODE_ITERATIONS)
img1 = cv2.dilate(img1, struct, anchor=(-1, -1), iterations=DILATE_ITERATIONS)
contours, hierarchy = cv2.findContours(img1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Toma los contornos de lo reconocido y lo deja en
# boxes de x,y y w,h
boxes = []
for contour in contours:
    # Convierte coordenads de x,y con
    # alto y ancho
    box = cv2.boundingRect(contour)
    h = box[3]

    if HIGHT_MIN < h < HIGHT_MAX:
        boxes.append(box)

mostrar(img_frame.copy(), boxes)



boxes = list(sorted(boxes, key=lambda r: (r[0], r[1])))
rows = {}
for box in boxes:
    (x, y, w, h) = box
    row_key = y // 10
    rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]
mostrar(img_frame.copy(), boxes)

boxes = []
htemp = []
for i in rows:
    boxes.append(rows.get(i)[0])
    htemp = [rows.get(i)[0]]
    for box in rows.get(i):
        (x1, y1, w1, h1) = htemp[-1]
        (x2, y2, w2, h2) = box
        (x3, y3, w3, h3) = boxes[-1]
        if ((x2 - x1) == 0):
            pass
        elif ((x2 - (x1 + w1)) > THRESHOLD_COLUMN):
            boxes.append(box)
        else:
            boxes[-1] = (x3, y3, w2+w3 + (x2 - (x1 + w1)), h2)
        htemp.append(box)
mostrar(img_frame.copy(), boxes)
dataframe = pytesseract.image_to_data(img_frame, output_type=pytesseract.Output.DATAFRAME, config='--oem 1 --psm 6')


#%%
#%%


boxes = list(sorted(boxes, key=lambda r: (str(r[0])[:2], r[1])))
cols = {}
for box in boxes:
    (x, y, w, h) = box
    col_key = x // 10
    cols[col_key] = [box] if col_key not in cols else cols[col_key] + [box]
mostrar(img_frame.copy(), boxes)

boxes = []
htemp = []
for i in cols:
    boxes.append(cols.get(i)[0])
    htemp = [cols.get(i)[0]]
    for box in cols.get(i):
        (x1, y1, w1, h1) = htemp[-1]
        (x2, y2, w2, h2) = box
        (x3, y3, w3, h3) = boxes[-1]
        if (y2 - y1) == 0:
            pass
        elif ((y2 - (y1 + h1)) > THRESHOLD_ROW):
            boxes.append(box)
        else:
            boxes[-1] = (x2, y1, w1, h1+h2)
        htemp.append(box)
mostrar(img_frame.copy(), boxes)

boxes = list(sorted(boxes, key=lambda r: (r[1], r[0])))
cols1 = {}
for box in boxes:
    (x, y, w, h) = box
    row_key = x // 10
    cols1[row_key] = [box] if row_key not in cols1 else cols1[row_key] + [box]
# cols1

punto = []
for i in cols1:
    x = []
    y = []
    for box in cols1.get(i):
        x.append(box[0])
        y.append(box[0] + box[2])

    punto.append((min(x), max(y)))

punto = sorted(punto)
# punto

punto_columna = []
for i in range(len(punto)):
    if i==0:
        punto_columna.append(punto[i][0])
    else:
        punto_columna.append((punto[i-1][1] + punto[i][0])//2)
punto_columna.append(punto[-1][1])
# punto_columna

boxes = list(sorted(boxes, key=lambda r: (r[1], r[0])))
rows1 = {}
for box in boxes:
    (x, y, w, h) = box
    row_key = y // 10
    rows1[row_key] = [box] if row_key not in rows1 else rows1[row_key] + [box]

# rows1

punto = []
for i in rows1:
    x = []
    y = []
    for box in rows1.get(i):
        x.append(box[1])
        y.append(box[1] + box[3])

    punto.append((min(x), max(y)))

punto = sorted(punto)
# punto


punto_fila = []
for i in range(len(punto)):
    if i==0:
        punto_fila.append(punto[i][0])
    else:
        punto_fila.append((punto[i-1][1] + punto[i][0])//2)
punto_fila.append(punto[-1][1])
# punto_fila


#----------
_img = img.copy()
for line in punto_fila:
    x1 = punto_columna[0]
    y1 = line
    x2 = punto_columna[-1]
    y2 = line
    cv2.line(_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

for line in punto_columna:
    x1 = line
    y1 = punto_fila[0]
    x2 = line
    y2 = punto_fila[-1]
    cv2.line(_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

#----------
mostrar(_img, [(0,0,0,0)])


celda = np.empty((len(punto_fila) - 1,len(punto_columna) - 1), dtype='object')
celdax = np.empty((len(punto_fila) - 1,len(punto_columna) - 1), dtype='object')
for i in range(len(punto_fila)):
    for ii in range(len(punto_columna)):
        try:
            text = re.sub(
                r"[\n]|[\x0c]|[,]", "", 
                pytesseract.image_to_string(
                    img[punto_fila[i]-3:punto_fila[i+1], 
                    punto_columna[ii]:punto_columna[ii+1]], 
                    config='--oem 1 --psm 6')
                    )
            celda[i, ii] = text
            celdax[i, ii] = (punto_fila[i],punto_fila[i+1], '--' , punto_columna[ii],punto_columna[ii+1])
        except:
            pass
pd.DataFrame(celda)
pd.DataFrame(celdax)
# pytesseract.image_to_string(im, config='--psm 6')




# %%
