"""Modulo que maneja la base de datos de la aplicacion
https://pypi.org/project/tinydb/
"""

# %% Importar modulos

from typing import List
from tinydb import TinyDB, Query

# %% Se instancia la base de datos

DIR_TEMPLATE = 'app/db/template'

db = TinyDB('db.json')
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
        'campos':[{'campo':i[0], 'x_min':i[1], 'y_min':i[2], 'width':i[3], 'height':i[4]} for i in campos]
    }

    db.upsert(dic, factura.nit == id)

def busca_nit(nit:int):

    return db.search(factura.nit == nit)[0]['campos']


# %%
