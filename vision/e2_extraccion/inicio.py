
# usar de pillow im.verify() para hacer una verificacion de imagen no corrompida
# os.path.basename('data/5.pdf')

#%%
import os
import imghdr
from convert_pdf import pdf_a_imagen, imagen_a_imagen

# creacion de variables globales

CARPETA = 'data'
CARPETA_DESTINO = 'data_imagenes'

def recorre_carpeta(folder):
    """Recorre una carpeta y regresa un generador para usar de uno en uno

    Args:
        folder ([type]): [description]
    """

    for (path, dirs, files) in os.walk(folder):
        for file in files:
            yield folder, file

def ejecuta(folder, folder_out):
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
                print('formato de archivo cambiado {}'.format(_file))

            else:
                print('archivo {} no se puede procesar formato no es jpeg o pdf'.format(_file))

        except StopIteration:
            print('fin del lote')
            break

ejecuta(CARPETA, CARPETA_DESTINO)


# %%
