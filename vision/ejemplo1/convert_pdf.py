# conda install -c conda-forge poppler
# pip install pillow

import os
from pdf2image import convert_from_path
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt


in_file = os.path.join("data", r'8.pdf')
out_file = os.path.join("data", r'8_out.jpg')
pages = convert_from_path(in_file)
[page.save(out_file, "JPEG") for page in pages]
pages[0]

with BytesIO() as f:
    pages[0].save(f, format="jpeg")
    f.seek(0)
    img_page = Image.open(f)

image = Image.open(out_file)
alto_fijo = 1024
alto_porcentaje = (alto_fijo / float(image.size[1]))
ancho_objetivo = int(float(image.size[0])* float(alto_porcentaje))
image_new = image.resize((ancho_objetivo, alto_fijo))
image_new.save("hh.jpg")

ancho_fijo = 768
ancho_porcentaje = (ancho_fijo / float(image.size[0]))
alto_objetivo = int(float(image.size[1]) * float(ancho_porcentaje))
image_new = image.resize((ancho_fijo, alto_objetivo))
image_new.save("hh1.jpg")

pages[0].save("hh1.jpg", format="JPEG")

plotting = plt.imshow(img_bin)
plt.show()