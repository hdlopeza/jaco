
# usar de pillow im.verify() para hacer una verificacion de imagen no corrompida
# os.path.basename('data/5.pdf')

#%%
import os
import imghdr
from convert_pdf import pdf_a_imagen

# creacion de variables globales

CARPETA = 'data'

def recorre_carpeta(folder):
    """Recorre una carpeta y regresa un generador para usar de uno en uno

    Args:
        folder ([type]): [description]
    """

    for (path, dirs, files) in os.walk(folder):
        for file in files:
            yield folder, file

def valida_archivo(folder, file):
    """Funcion que valida el tipo de archivo
    regresa si es pdf, imagen jpeg, otro tipo de imagen, o archivo erroneo

    Args:
        file ([str]): Archivo con la ruta completa
    """

    a = imghdr.what(os.path.join(folder, file))

    if file.lower().endswith(('pdf')):
        pdf_a_imagen(folder=folder, file=file)
        return print('archivo pdf')
    elif str(a) in ('jpeg'):
        return print('archivo sigue derecho es jpeg')
    elif a != None:
        return print('cambiar formato al archivo')
    else:
        return print('archivo {} no se puede procesar formato no es jpeg o pdf'.format(file))

def ejecuta(folder):

    _ = recorre_carpeta(folder)

    while True:
        try:
            _folder, _file = next(_)
            valida_archivo(_folder, _file)
        except StopIteration:
            print('fin del lote')
            break

ejecuta(CARPETA)


# %%
