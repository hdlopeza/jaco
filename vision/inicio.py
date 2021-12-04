
# usar de pillow im.verify() para hacer una verificacion de imagen no corrompida
# os.path.basename('data/5.pdf')

#%%
import os
import cv2
import re
import pytesseract
from app.vision.convert_pdf import pdf_a_imagen, imagen_a_imagen
import app.vision.imghdr as imghdr
import app.vision.convert_pdf as cpdf
import app.reconocimiento.rnn as reconoce
import app.db.database as db
import app.vision.align as align

import matplotlib.pyplot as plt
# %matplotlib inline


# creacion de variables globales

CARPETA = '../data'
CARPETA_DESTINO = 'data'
CARPETA_ERRORES = 'data_error'
CARPETA_IMAGES_TEMPLATES = 'app/db/templates'


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

def preprocesamiento_a_imagen(folder, folder_out):
    """Ejecuta la operacion de recorrer una carpeta
    y validar si el archivo pasa las pruebas para generar el documento
    """

    _ = recorre_carpeta(folder)

    while True:
        try:
            _folder, _file = next(_)

            _path = os.path.splitext(_file)
            _in = os.path.join(folder, _file)
            _out = os.path.join(folder_out, _path[0] + '_new.jpg')

            a = imghdr.what(os.path.join(_folder, _file))

            if _file.lower().endswith(('pdf')):
                pdf_a_imagen(file_in=_in, file_out=_out)
                print('(pdf) se crea el archivo {}'.format(_file))

            elif str(a) in ('jpeg'):
                imagen_a_imagen(file_in=_in, file_out=_out)
                print('archivo sigue derecho es jpeg, pero se asegura resolucion')

            elif a != None:
                imagen_a_imagen(file_in=_in, file_out=_out)
                print('formato de archivo cambiado {} y se asegura la resolucion'.format(_file))

            else:
                print('archivo {} no se puede procesar formato no es jpeg o pdf'.format(_file))

        except StopIteration:
            print('fin del lote')
            break

def reconocimiento_imagenes(folder, folder_out):

    _ = recorre_carpeta(folder)

    lista = []

    while True:
        try:
            _folder, _file = next(_)
            _in = os.path.join(_folder, _file)

            lista.append(reconoce.nit_imagen(_in))

        except StopIteration:
            print('--- fin del lote ---')
            return lista

def clasificar_imagenes(lista, folder_out, limite=0.2):

    lista_imagenes = []

    for i in lista:
        if i[1] < limite:
            # Mover el archivo a la carpeta de errores
            os.rename(i[2], folder_out + '/' + os.path.basename(i[2]))
        else:
            lista_imagenes.append(i)

    return lista_imagenes

def alinear_imagenes(lista):
    """Toma la lista de imagenes que sale de clasificar imagenes
    y hace un recorrido, alineandolas con el template

    regresa las imagenes alineadas en la ruta original de funcionamiento

    Args:
        lista ([type]): [description]
    """

    for i in lista:
        try:
            record = db.busca_nit(int(i[0]))
            path_template = os.path.join(
                CARPETA_IMAGES_TEMPLATES,
                record[0].get('img_template'))
            path_target = i[2]
            image_new = align.align_images(
                image=path_target,
                template=path_template)
            # image_new = cpdf.change_resolution(image_new)

            cv2.imwrite(path_target, image_new)

        except IndexError:
            print('error en el registro {}'.format(i))

def ocr_imagenes(lista):
    """Toma un cuadro de un sector y lo convierte
    https://www.pyimagesearch.com/2020/09/07/ocr-a-document-form-or-invoice-with-tesseract-opencv-and-python/

    lista = [('8001304263', 0.84001195, 'data/8001304263_new.jpg'),
        ('8002126635', 0.6552293, 'data/8002126635_new.jpg'),
        ('8110213537', 0.8046723, 'data/2-029_new.jpg'),
        ('8001539937', 0.6190442, 'data/8001539937_new.jpg')]
    """

    for i in lista:
        try:

            results = {}

            img = cv2.imread(i[2])
            img_bound = img.copy()
            height, weight, _ = img.shape

            record = db.busca_nit(int(i[0]))[0].get('campos')

            for ii in record:

                y_min = int(ii.get('y_min')* height)
                y_max = int(ii.get('y_max')* height)
                x_min = int(ii.get('x_min')* weight)
                x_max = int(ii.get('x_max')* weight)

                roi = img[y_min:y_max , x_min:x_max]

                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                if ii.get('campo') == "detalle":
                    print("se realiza con tabla sin rayas")

                else:
                    text = re.sub(
                        r"[\n]|[\x0c]|[,]"
                        , ""
                        , pytesseract.image_to_string(roi, config='--oem 1 --psm 6')
                        )
                    results[ii.get('campo')] = text

                # img_bound = cv2.rectangle(img_bound, (x_min, y_min), (x_max,y_max), (0,255,0), 2)

            print(results)
            # plt.imsave('hh.jpg', img_bound)


        except IndexError:
            print('error en el registro {}'.format(i))



preprocesamiento_a_imagen(CARPETA, CARPETA_DESTINO)
lista_imagenes = reconocimiento_imagenes(CARPETA_DESTINO, CARPETA_ERRORES)
lista_imagenes = clasificar_imagenes(lista=lista_imagenes, folder_out=CARPETA_ERRORES)
alinear_imagenes(lista_imagenes)
ocr_imagenes(lista_imagenes)


# %%
