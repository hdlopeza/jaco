"""
Libreria que toma un archivo XML generado con labelImg y extrae la informacion de
campos que será objeto de reconocimiento para posterior ingresarla en la base de datos

OJO!!!
en labelImg cuando se realice las areas de las columnas en la seccion de detalle, NO se pueden traslapar 
areas, deben quedar independientes, asi sea por un pixel
"""

# %% Importar librerias

import os
import xml.etree.ElementTree as ET
from app_misc import recorre_carpeta
from app_db import inserta_dic


# %%

def ingresa_xml(folder):

    _ = recorre_carpeta(folder)

    while True:
        try:
            _folder, _file = next(_)
            if os.path.splitext(_file)[1] == '.xml':

                file = _file

                xml_list = [] # Contenedor de los campos de factura
                xml_list1 = [] # Contenedor de los campos de la seccion detalle
                tree = ET.parse(os.path.join(_folder, file))
                root = tree.getroot()
                filename = root.find('filename').text
                width = int(root.find('size').find('width').text)
                height = int(root.find('size').find('height').text)

                for member in root.findall('object'):
                    bndbox = member.find('bndbox')

                    value = ({
                        'campo': member.find('name').text,
                        'x_min': int(bndbox.find('xmin').text) / width,
                        'y_min': int(bndbox.find('ymin').text) / height,
                        'x_max': int(bndbox.find('xmax').text) / width,
                        'y_max': int(bndbox.find('ymax').text) / height,
                        })

                    if (member.find('name').text == 'logo'): 
                        continue
                    elif ('columna' in member.find('name').text) or ('detalle' in member.find('name').text):
                        xml_list1.append(value)
                    else:
                        xml_list.append(value)

                dic ={
                    'nit': int(os.path.splitext(filename)[0])
                    ,'img_template': filename
                    ,'campos': xml_list
                    ,'detalle': xml_list1
                }

                inserta_dic(dic)

        except StopIteration:
            print('fin del lote')
            break


ingresa_xml('app/db/templates')
# %%
