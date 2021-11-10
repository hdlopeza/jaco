"Convierte un documento de pdf a jpg"

# conda install -c conda-forge poppler
# pip install pillow
# conda install -c conda-forge pdf2image

#%%
import os
import cv2
from pdf2image import convert_from_path
# import matplotlib.pyplot as plt

# from io import BytesIO
# from PIL import Image

#%%

in_file = os.path.join('../', "data", r'8.pdf')
out_file = os.path.join('../', "data", r'8_out.jpg')
pages = convert_from_path(in_file)
pages[0].save(out_file, format="JPEG")

# pages[0]
# [page.save(out_file, "JPEG") for page in pages]

# with BytesIO() as f:
#     pages[0].save(f, format="jpeg")
#     f.seek(0)
#     img_page = Image.open(f)


#%%

image = cv2.imread(out_file)
# image = Image.open(out_file)

# %%

# alto_fijo = 1024
# alto_porcentaje = (alto_fijo / float(image.size[1]))
# ancho_objetivo = int(float(image.size[0])* float(alto_porcentaje))
# image_new = image.resize((ancho_objetivo, alto_fijo))
# image_new.save("hh.jpg")


ancho_fijo = 768
ancho_porcentaje = (ancho_fijo / float(image.shape[1]))
alto_objetivo = int(float(image.shape[0]) * float(ancho_porcentaje))

image_new = cv2.resize(image, (ancho_fijo, alto_objetivo))
# image_new = image.resize((ancho_fijo, alto_objetivo))
# image_new.save("hh1.jpg")

# %%

cv2.imwrite(out_file, image_new)
# %%
