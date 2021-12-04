"""Modulo que maneja la base de datos de la aplicacion
https://pypi.org/project/tinydb/
https://github.com/pysonDB/pysonDB
"""

# %% Importar modulos

from tinydb import TinyDB, Query

# %% Se instancia la base de datos

DIR_TEMPLATE = 'app/db/template'

db = TinyDB('app/db/db.json')
factura = Query()

# %% Funciones de ayuda

def inserta_registro(id, img_template, campos):
    """Inserta un registro en la base de datos
    verifica si esta duplicado lo sobreescribe

    Args:
        id ([type]): [description]
        campos ([type]): [description]

    Returns:
        [type]: [description]
    """

    dic = {
        'nit':id,
        'img_template': img_template,
        'campos':[{'campo':i[0], 'x_min':i[1], 'y_min':i[2], 'x_max':i[3], 'y_max':i[4]} for i in campos]
    }

    db.upsert(dic, factura.nit == id)

def inserta_dic(dic):
    """Inserta un registro completo de diccionario

    Args:
        dic ([type]): [description]
    """

    db.upsert(dic, factura.nit == dic.get('nit'))

def busca_nit(nit:int):
    try:
        # return db.search(factura.nit == nit)[0]['campos']
        return db.search(factura.nit == nit)
    except  IndexError:
        print('no hay registros')
