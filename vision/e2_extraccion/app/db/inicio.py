"""
Libreria que toma un archivo XML generado con labelImg y extrae la informacion de
campos que será objeto de reconocimiento para posterior ingresarla en la base de datos
"""

# %% Importar librerias

import os
import xml.etree.ElementTree as ET
from database import inserta_dic


# %%

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

def ingresa_xml(folder):

    _ = recorre_carpeta(folder)

    while True:
        try:
            _folder, _file = next(_)
            if os.path.splitext(_file)[1] == '.xml':

                file = _file

                xml_list = []
                tree = ET.parse(os.path.join(_folder, file))
                root = tree.getroot()
                filename = root.find('filename').text
                width = int(root.find('size').find('width').text)
                height = int(root.find('size').find('height').text)

                for member in root.findall('object'):
                    bndbox = member.find('bndbox')
                    if (member.find('name').text == 'logo'): continue
                    value = ({
                        'campo': member.find('name').text,
                        'x_min': int(bndbox.find('xmin').text) / width,
                        'y_min': int(bndbox.find('ymin').text) / height,
                        'x_max': int(bndbox.find('xmax').text) / width,
                        'y_max': int(bndbox.find('ymax').text) / height,
                        })
                    xml_list.append(value)

                dic ={
                    'nit': int(os.path.splitext(filename)[0]),
                    'img_template': filename,
                    'campos':xml_list
                }

                inserta_dic(dic)

        except StopIteration:
            print('fin del lote')
            break


ingresa_xml('templates')
# %%
