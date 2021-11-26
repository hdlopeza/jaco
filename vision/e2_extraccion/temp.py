

#%%

"""
Toma un cuadro de un sector y lo convierte

https://www.pyimagesearch.com/2020/09/07/ocr-a-document-form-or-invoice-with-tesseract-opencv-and-python/
"""

import os
import cv2
import pytesseract

import app.db.database as db


res = [('8001304263', 0.84001195, 'data/8001304263_new.jpg'),
 ('8002126635', 0.6552293, 'data/8002126635_new.jpg'),
 ('8909049961', 0.8228462, 'data/8909049961_new.jpg'),
 ('8110213537', 0.8046723, 'data/8000213537_new.jpg'),
 ('8110142325', 0.21848114, 'data/345-77_new.jpg'),
 ('8001304263', 0.76955444, 'data/19-104_new.jpg'),
 ('8110142325', 0.21848114, 'data/8110142325_new.jpg'),
 ('33289265', 0.38963863, 'data/33289265_new.jpg'),
 ('8909299223', 0.36123875, 'data/8909299223_new.jpg'),
 ('8110213537', 0.8046723, 'data/2-029_new.jpg'),
 ('8001539937', 0.6190442, 'data/8001539937_new.jpg')]


# %%

def ocr_imagenes(lista):
    """Toma la lista de imagenes que sale de clasificar imagenes
    y hace un recorrido, reconociendo cada campo

    regresa las imagenes alineadas en la ruta original de funcionamiento

    Args:
        lista ([type]): [description]
    """

    for i in res[0:2]:
        try:

            results = {}

            img = cv2.imread(i[2])
            height, weight, _ = img.shape

            record = db.busca_nit(int(i[0]))[0].get('campos')
            for ii in record:
                roi = img[
                    int(ii.get('y_min')* height):int(ii.get('y_max')*height),
                    int(ii.get('x_min')* weight):int(ii.get('x_max')*weight)
                    ]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                text = pytesseract.image_to_string(roi)
                results[ii.get('campo')] = text
            print(results)


        except IndexError:
            print('error en el registro {}'.format(i))

ocr_imagenes(res)

# %%
