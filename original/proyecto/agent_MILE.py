import os, re, fnmatch, shutil, json, firebase_admin
import numpy as np
from google.cloud import vision
from google.cloud import storage
from google.cloud.vision import types
from google.protobuf import json_format
from enum import Enum
from PIL import ImageDraw, Image
from pdf2image import convert_from_path
from firebase_admin import credentials, firestore

# Definicion de variables con google [START]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './proyecto/connection/hdlaml-vision.json'
# Definicion de variables con google [END]

# Definicion de variables con firestore [START]
cred = credentials.Certificate("./proyecto/connection/hdlaml-firebase.json")
firebase_admin.initialize_app(cred)
# Definicion de variables con firestore [END]

# Definicion de variables [START]
path_source_client = './proyecto/images/' # OK debe ir el / al final
path_source = path_source_client + 'MILE/'  # OK debe ir el / al final
path_source_responses_from_vision = path_source + 'vision/'
# Definicion de variables [END]

#1 Funcion que toma los archivos los convierte a imagen y los deja en carpeta final
def file_to_MILE():
    '''
    Crea la carpeta de donde se tomaran las imagenes y si hay archivos 'pdf' convierte la primera hoja a imagen
    ** Referencias
        https://docs.python.org/2/library/filesys.html
        https://stackoverflow.com/questions/5899497/checking-file-extension
        https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-path_source-in-python
        https://stackoverflow.com/questions/18383384/python-copy-files-to-a-new-path_source-and-rename-if-file-name-already-exists
        https://realpython.com/working-with-files-in-python/

    Convierte la primera pagina de un archivo pdf a imagen jpg de 250 pdi
    ** Referencias
        https://stackoverflow.com/questions/2693820/extract-images-from-pdf-without-resampling-in-python
        https://github.com/Belval/pdf2image
        pip install {pooper, pdf2image}

    '''
    # Si no esta creado la carpeta de destino la crea
    if not os.path.exists(path_source):
        os.makedirs(path_source)


    [
        # Si en la carpeta de origen hay archivos png o jpg y los copia a la nueva ubicacion
        shutil.copy(path_source_client + file, path_source + file)
        # Itera sobre todos los archivos que hay en la carpeta de origen 
        for file in os.listdir(path_source_client)
        if fnmatch.fnmatch(file, '*.png') or fnmatch.fnmatch(file, '*.jpg')
    ]

    # Si en la carpeta de origen hay archivos pdf los convierte a imagen jpg y los copia a la nueva ubicacion
    [   convert_from_path(pdf_path=path_source_client + file, output_folder=path_source, output_file=file[0:len(file)-4], dpi=250, fmt='JPEG', first_page=1, last_page=1)
        for file in os.listdir(path_source_client)
        if fnmatch.fnmatch(file, '*.pdf')
    ]
    return 'Archivos entregados a M.I.L.E'

#2 cloud pub o storage entrega a ML la imagen, este paso entrega localmente el nit 
# del docuento y la imagen en si misma, uno a uno 
def ml_invoices_numbers_and_paths():
    '''
    Trae de Google ML el NIT de la factura(imagen)
    y el archivo de la imagen, asociados en un array
    '''

    # Si en la carpeta hay arhivos jpg o png los manda a reconocimiento de NIT
    # OJO aqui se esta asumiendo que todas las facturas tienen el mismo codigo, pero
    # eso sera una respuesta de ML ---- esto es la BB1 (BLACKBOX1)

    tmp_numbers_and_paths = [['1088000426', path_source + file]
        for file in os.listdir(path_source)
        if fnmatch.fnmatch(file, '*.png') or fnmatch.fnmatch(file, '*.jpg')
    ]

    return np.array(tmp_numbers_and_paths)

#3.1.b 
def text_recogn(document_annotated_by_vision, x1, x2, y1, y2, namefield):
    '''
    https://cloud.google.com/vision/docs/libraries#client-libraries-install-python
    https://cloud.google.com/vision/docs/libraries#client-libraries-install-python
    https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/vision/cloud-client/detect/detect.py
    pip install google-cloud
    pip install google-cloud-vision
    google-cloud-pubsub==0.38.0
    google-cloud-storage==1.13.0
    google-cloud-translate==1.3.1
    google-cloud-vision==0.35.0
    
    Performs text detection on the image file y devuelve en json todo el resultado
    texts = client.document_text_detection(image=image)
    response = client.document_text_detection(image=image)
    texts = response.text_annotations
    print(texts) en format json
     
    '''

    text = ""
    for page in document_annotated_by_vision.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        min_x=min(symbol.bounding_box.vertices[0].x,symbol.bounding_box.vertices[1].x,symbol.bounding_box.vertices[2].x,symbol.bounding_box.vertices[3].x)
                        max_x=max(symbol.bounding_box.vertices[0].x,symbol.bounding_box.vertices[1].x,symbol.bounding_box.vertices[2].x,symbol.bounding_box.vertices[3].x)
                        min_y=min(symbol.bounding_box.vertices[0].y,symbol.bounding_box.vertices[1].y,symbol.bounding_box.vertices[2].y,symbol.bounding_box.vertices[3].y)
                        max_y=max(symbol.bounding_box.vertices[0].y,symbol.bounding_box.vertices[1].y,symbol.bounding_box.vertices[2].y,symbol.bounding_box.vertices[3].y)
                        if(min_x >= x1 and max_x <= x2 and min_y >= y1 and max_y <= y2):
                            text+=symbol.text
                            if(symbol.property.detected_break.type==1 or 
                                symbol.property.detected_break.type==3):
                                text+=' '
                            if(symbol.property.detected_break.type==2):
                                text+='\t'
                            if(symbol.property.detected_break.type==5):
                                text+='\n'
    print('{}: {}'.format(namefield, text))

#3.1.a.1.1
def assemble_word(word):
    '''
    Funcion que ensambla las palabras
    '''
    assembled_word = ""
    for symbol in word.symbols:
        assembled_word += symbol.text
    return assembled_word

#3.1.a.1
def find_word_location(document_annotated_by_vision, word_to_find):
    '''
    Funcion que busca la palabra y se da los poligonos
    '''
    for page in document_annotated_by_vision.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    assembled_word = assemble_word(word=word)
                    if (assembled_word == word_to_find):
                        return word.bounding_box.vertices[0]

#3.1.a 
def fs_invoice_fields_and_boundigs(invoice_number, document_annotated_by_vision):
    '''
    Hace una conexion a la base de datos y trae todos los documentos relacionados con el numero de factura
    '''
    tmp_detail=[]

    db = firestore.client() # Se inicializa el admin SDK

    collection = db.collection(invoice_number).order_by('type').get()
    for doc in collection:
        tmp_fields = doc.to_dict()

        if tmp_fields['type'] == 0:
            tmp_dot_reference = find_word_location(
                                    document_annotated_by_vision=document_annotated_by_vision, 
                                    word_to_find=tmp_fields['name']) 


        # busca el campo 0= misc, 1 principal, 2 detalle
        if tmp_fields['type'] != 0:
            tmp_detail.append(
                    #Pasa los datos del poligono al campo para adicionarlo al documento factura
                    text_recogn(
                        document_annotated_by_vision = document_annotated_by_vision,
                        namefield = doc.id, 
                        x1 = tmp_dot_reference.x + tmp_fields['bounding_box']['x1'], 
                        x2 = tmp_dot_reference.x + tmp_fields['bounding_box']['x2'],
                        y1 = tmp_dot_reference.y + tmp_fields['bounding_box']['y1'], 
                        y2 = tmp_dot_reference.y + tmp_fields['bounding_box']['y2']
                    ))

#3.1 Reconocimiento de una sola imagen por Google Vision
def ml_invoice_decode(path_source_image, invoice_number):
    '''
    Toma la ruta de una imagen y la pasa a google vision
    para que la reconozca y retorne el documento con las anotaciones
    y se guarde en una carpeta diferente

    ** Referencias
        https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
    '''

    # a. Cargar la imagen en la memoria, rb reading binary
    # y la pasa al objeto de vision
    with open(path_source_image, 'rb') as image_file:
        tmp_image_opened = image_file.read()
    tpm_image_opened_to_vision = vision.types.Image(content=tmp_image_opened)

    # b. Crear el cliente de google
    tpm_vision_client = vision.ImageAnnotatorClient()

    # c. Recibir la respuesta de google con todo lo maximo que encontro en la imagen
    tpm_response_from_vision = tpm_vision_client.document_text_detection(image=tpm_image_opened_to_vision)

    # d. Documento a trabajar con la respuesta
    tpm_document_annotated_by_vision = tpm_response_from_vision.full_text_annotation

    # Si no esta creado la carpeta de destino la crea
    if not os.path.exists(path_source_responses_from_vision):
        os.makedirs(path_source_responses_from_vision)

    tmp_path_file, tmp_file = os.path.split(path_source_image)

    with open(path_source_responses_from_vision + invoice_number + '_' + tmp_file[0:len(tmp_file)-4] + '.json', 'w') as f:  
        f.write(
            json_format.MessageToJson(
                tpm_response_from_vision))
    
    # Como aun no se como hacer el proceso asincronico, tengo que unir las funciones,
    # Aqui llamo la funcion para buscar el punto de referencia, y luego la funcion para
    # que busca los puntos de cada campo y los reconoce document['pages'] document.pages
    
    fs_invoice_fields_and_boundigs(
        invoice_number=invoice_number, 
        document_annotated_by_vision=tpm_document_annotated_by_vision)
    
#3. Reconocimiento de todas las imagenes por Google Vision
def invoices_decode(numbers_and_paths=[]):
    '''
    Toma la ruta de las imagenes, lee cada imagen, la pasa a Google Vision
    y recibe de alli, cada factura decodificada de imagen a texto
    ml_invoice_decode(x[1]) y este ultimo lo guarda en una carpeta
    '''
    # Si no esta creado la carpeta de destino la crea
    if not os.path.exists(path_source_responses_from_vision):
        os.makedirs(path_source_responses_from_vision)

    [map(lambda x: (
        x[0], 
        ml_invoice_decode(
            path_source_image=str(x[1]),
            invoice_number=x[0])), 
        numbers_and_paths)]
    
    return 'Facturas decodificadas'
