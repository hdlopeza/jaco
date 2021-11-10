
# usar de pillow im.verify() para hacer una verificacion de imagen no corrompida

#%%
import os
import imghdr


#%%
# creacion de variables globales

CARPETA = os.path.join('data')


def valida_archivo(file):
    """Funcion que valida el tipo de archivo
    regresa si es pdf, imagen jpeg, otro tipo de imagen, o archivo erroneo

    Args:
        file ([type]): [description]
    """

    a = imghdr.what(file)

    if file.lower().endswith(('pdf')):
        return print('archivo pdf')
    elif str(a) in ('jpeg'):
        return print('archivo sigue derecho es jpeg')
    elif a != None:
        return print('cambiar formato al archivo')
    else:
        return print('archivo no se puede procesar formato no es jpeg o pdf')


for (path, dirs, files) in os.walk(CARPETA):
    for file in files:
        valida_archivo(os.path.join(CARPETA, file))


# %%
