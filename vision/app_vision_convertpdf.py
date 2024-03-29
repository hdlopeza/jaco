"Convierte un documento de pdf a jpg"

# conda install -c conda-forge poppler
# pip install pillow
# conda install -c conda-forge pdf2image

# %%
import os
import cv2
import numpy as np

from pdf2image import (convert_from_path, convert_from_bytes)

# import matplotlib.pyplot as plt
# from io import BytesIO
# from PIL import Image

def pdf_a_imagen(file_in, file_out):
    """Crea una imagen a partir de un documento pdf

    Args:
        file_in ([type]): [description]
        file_out ([type]): [description]
    """

    pages = convert_from_bytes(pdf_file=open(file_in, 'rb').read())
    image_new = np.asarray(pages[0])

    image_new = change_resolution(imagen=image_new)

    cv2.imwrite(file_out, image_new)

def imagen_a_imagen(file_in, file_out):
    """Se asegura que la resolucion sea la definida

    Args:
        folder ([type]): [description]
        file ([type]): [description]
        folder_out ([type]): [description]
    """

    image_new = cv2.imread(file_in)
    image_new = change_resolution(imagen=image_new)

    cv2.imwrite(file_out, image_new)

def change_resolution(imagen):
    """Se asegura que la imagen final siempre quede en la resolucion
    objetivo

    Args:
        imagen ([type]): [description]

    Returns:
        [type]: [description]
    """

    image_new = imagen

    ancho_fijo = 768
    alto, ancho, canales = image_new.shape
    ancho_porcentaje = (ancho_fijo / ancho)
    alto_objetivo = int(float(alto) * float(ancho_porcentaje))
    image_new = cv2.resize(image_new, (ancho_fijo, alto_objetivo))
    
    return image_new


# %% Convertir imagenes desde una ruta dada

# Opcion 1
# pages = convert_from_path(in_file)
# pages[0].save(out_file, format="JPEG")

# Opcion 2
# pages = convert_from_path(in_file)
# [page.save(out_file, "JPEG") for page in pages]

# Opcion 3
# pages = convert_from_path(in_file)
# with BytesIO() as f:
#     pages[0].save(f, format="jpeg")
#     f.seek(0)
#     img_page = Image.open(f)

# Opcion 4
# pages = convert_from_bytes(pdf_file=open(in_file, 'rb').read())


# %% Pasar las imagenes a OpenCv

# Opcion 1
# image = cv2.imread(out_file)

# Opcion 2
# image = Image.open(out_file)

# Opcion 3
# image_new = np.asarray(pages[0])

# %% Definir la escala en funcion del alto o del ancho

# Opcion con alto
# alto_fijo = 1024
# alto_porcentaje = (alto_fijo / float(image.size[1]))
# ancho_objetivo = int(float(image.size[0])* float(alto_porcentaje))

# Opcion con ancho
# ancho_fijo = 768
# alto, ancho, canales = image_new.shape
# ancho_porcentaje = (ancho_fijo / ancho)
# alto_objetivo = int(float(alto) * float(ancho_porcentaje))

# %% Hacer la rescala de la imagen

# Opcion 1
# image_new = image.resize((ancho_fijo, alto_objetivo))

# Opcion 2
# image_new = cv2.resize(image_new, (ancho_fijo, alto_objetivo))

# %% Salvar la imagen

# Opcion 1
# image_new.save("hh1.jpg")

# Opcion 2
# cv2.imwrite(out_file, image_new)
