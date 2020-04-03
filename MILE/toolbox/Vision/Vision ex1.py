import os
from google.cloud import vision
from google.cloud.vision import types

#Pasar la cadena de conexion al cliente
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./connection/apikey.JSON"

#Crear el cliente
client = vision.ImageAnnotatorClient()

#Dar la imagen y leerla
image_to_open = './images/5_1.jpg'

with open(image_to_open, 'rb') as image_file:
    content = image_file.read()

image = vision.types.Image(content=content)

text_response = client.document_text_detection(image=image)

# print(text_response)

# Filtrar los campos que realmente queremos de respuesta
text = ""
for page in text_response.full_text_annotation.pages:
    for block in page.blocks:
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                for symbol in word.symbols:
                    min_x=min(symbol.bounding_box.vertices[0].x,symbol.bounding_box.vertices[1].x,symbol.bounding_box.vertices[2].x,symbol.bounding_box.vertices[3].x)
                    max_x=max(symbol.bounding_box.vertices[0].x,symbol.bounding_box.vertices[1].x,symbol.bounding_box.vertices[2].x,symbol.bounding_box.vertices[3].x)
                    min_y=min(symbol.bounding_box.vertices[0].y,symbol.bounding_box.vertices[1].y,symbol.bounding_box.vertices[2].y,symbol.bounding_box.vertices[3].y)
                    max_y=max(symbol.bounding_box.vertices[0].y,symbol.bounding_box.vertices[1].y,symbol.bounding_box.vertices[2].y,symbol.bounding_box.vertices[3].y)
                    if(min_x >= 42 and max_x <= 117 and min_y >= 16 and max_y <= 38):
                        text+=symbol.text
                        if(symbol.property.detected_break.type==1 or 
                            symbol.property.detected_break.type==3):
                            text+=' '
                        if(symbol.property.detected_break.type==2):
                            text+='\t'
                        if(symbol.property.detected_break.type==5):
                            text+='\n'
print(text)