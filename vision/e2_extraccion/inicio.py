
# usar de pillow im.verify() para hacer una verificacion de imagen no corrompida
# os.path.basename('data/5.pdf')

#%%
import os
import app.vision.imghdr as imghdr
from app.vision.convert_pdf import pdf_a_imagen, imagen_a_imagen
import app.reconocimiento.rnn as reconoce


# creacion de variables globales

CARPETA = '../data'
CARPETA_DESTINO = 'data'
CARPETA_ERRORES = 'data_error'

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


# %%

preprocesamiento_a_imagen(CARPETA, CARPETA_DESTINO)
lista_imagenes = reconocimiento_imagenes(CARPETA_DESTINO, CARPETA_ERRORES)
lista_imagenes = clasificar_imagenes(lista=lista_imagenes, folder_out=CARPETA_ERRORES)
# %%
