
# usar de pillow im.verify() para hacer una verificacion de imagen no corrompida
# os.path.basename('data/5.pdf')

#%%
import os
import cv2
from app_vision_convertpdf import pdf_a_imagen, imagen_a_imagen
from app_misc import recorre_carpeta
import app_vision
import app_vision_ocr
import app_vision_imghdr
import app_vision_convertpdf as cpdf
# import app_rnn as reconoce
import app_db as db
from datetime import datetime
import matplotlib.pyplot as plt
# %matplotlib inline

#%%

# creacion de variables globales

CARPETA = 'data_origen'
CARPETA_DESTINO = 'data'
CARPETA_ERRORES = 'data_error'
CARPETA_IMAGES_TEMPLATES = 'app/db/templates'

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

            a = app_vision_imghdr.what(os.path.join(_folder, _file))

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
            image_new = app_vision.align_images(
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
 ('8909049961', 0.8228462, 'data/8909049961_new.jpg'),
 ('8110213537', 0.8046723, 'data/8000213537_new.jpg'),
 ('8110142325', 0.21848114, 'data/345-77_new.jpg'),
 ('8001304263', 0.76955444, 'data/19-104_new.jpg'),
 ('8110142325', 0.21848114, 'data/8110142325_new.jpg'),
 ('33289265', 0.38963863, 'data/33289265_new.jpg'),
 ('8909299223', 0.36123875, 'data/8909299223_new.jpg'),
 ('8110213537', 0.8046723, 'data/2-029_new.jpg'),
 ('8001539937', 0.6190442, 'data/8001539937_new.jpg')]
    """

    for i in lista:
        try:
            # print(i[2])

            results = {}

            img = cv2.imread(i[2])
            height, weight, _ = img.shape
            record_original = db.busca_nit(int(i[0]))[0]
            record = record_original.get('campos')
            record1 = record_original.get('detalle')

            results['nit'] = record_original.get('nit')
            results['fecha_procesamiento'] = datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
            results['file'] = i[2]

            results['campos'] = [app_vision_ocr.ocr_campos(
                imagen=img
                , record=record
                , height=height
                , weight=weight)]

            results['detalle'] = [app_vision_ocr.ocr_detalle(
                imagen=img
                , record=record1
                , height=height
                , weight=weight)]

            db.inserta_dic_alone(results)

        except IndexError:
            print('error en el registro {}'.format(i))


preprocesamiento_a_imagen(CARPETA, CARPETA_DESTINO)
lista_imagenes = reconocimiento_imagenes(CARPETA_DESTINO, CARPETA_ERRORES)
lista_imagenes = clasificar_imagenes(lista=lista_imagenes, folder_out=CARPETA_ERRORES)
alinear_imagenes(lista_imagenes)
ocr_imagenes(lista_imagenes)


# %%
# Eliminar imagenes temporales
# !rm -r data/*
# !rm -r data_error/*


# %%

