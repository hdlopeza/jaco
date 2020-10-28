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

# Definicion de variables [START]
path_source_client = './proyecto/images/' # OK debe ir el / al final
path_source = path_source_client + 'MILE/'  # OK debe ir el / al final
path_source_responses_from_vision = path_source + 'vision/'
path_source_toolbox = path_source + 'toolbox/'
# Definicion de variables [END]

# Definicion de variables visibles par el programa tomados del codigo [START]
document_annotated_by_vision_full_text_annotation = None
document_annotated_by_vision_text_annotations = None
file_source = None
# Definicion de variables visibles par el programa tomados del codigo [START]

#1.2.c
class FeatureType(Enum):
    '''
    Define la tipologia de elementos a dibujar en la imagen de la factura
    '''
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5

#1.3
def draw_boxes(bounds, color, witdth=5):
    '''
    De acuerdo a unos limites recibidos dibuja los cuadros en una imagen

    ** Referencias
        https://github.com/monark12/vision-api/blob/master/vision_api.ipynb
        https://medium.com/searce/tips-tricks-for-using-google-vision-api-for-text-detection-2d6d1e0c6361
    '''
    img = Image.open(path_source_client + file_source)
    #Define el objeto PIL a dibujar en el
    tmp_draw = ImageDraw.Draw(img)
    for bound in bounds:
        tmp_draw.line([
            #Dibuja un cuadrado con los cuatro vertices en la imagen
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[3].y,
            bound.vertices[3].x, bound.vertices[3].y,
            bound.vertices[0].x, bound.vertices[0].y
        ],
            fill=color, width=witdth
        )

    img.save(path_source_toolbox + file_source[0:len(file_source)-4] + '_drawed' + '.jpg' ,'JPEG')

#1.2
def get_document_bounds(feature):
    '''
    Dependiendo de lo deseado consulta los boundries bloques, para, words, symbol
    '''
    # Define un array para guardar los limites (bounds)
    tmp_bounds = []
    #busca en el documento que respondio google lo que deseamos dibujar y los vertices los agrega al objeto bounds
    for i, page in enumerate(document_annotated_by_vision_full_text_annotation.pages):
        for block in page.blocks:
            if feature == FeatureType.BLOCK:
                tmp_bounds.append(block.bounding_box)
            for paragraph in block.paragraphs:
                if feature == FeatureType.PARA:
                    tmp_bounds.append(paragraph.bounding_box)
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if feature == FeatureType.SYMBOL:
                            tmp_bounds.append(symbol.bounding_box)
                    if feature == FeatureType.WORD:
                        tmp_bounds.append(word.bounding_box)
     
    return tmp_bounds

#1.1. Crea la instancia de vision a usar
def vision_client_instance(image_to_use):
    '''
    Define un objeto de vision y crea las respuestas de una sola imagen pasada
    El archivo a pasar debe ser una imagen jpg
    '''

    global document_annotated_by_vision_full_text_annotation
    global document_annotated_by_vision_text_annotations

    with open(image_to_use, 'rb') as image_file:
        tmp_image_opened = image_file.read()
    tpm_image_opened_to_vision = vision.types.Image(content=tmp_image_opened)

    tpm_vision_client = vision.ImageAnnotatorClient()
    tpm_response_from_vision = tpm_vision_client.document_text_detection(image=tpm_image_opened_to_vision)

    # Deja el documento visible para todo el modulo
    document_annotated_by_vision_full_text_annotation = tpm_response_from_vision.full_text_annotation
    document_annotated_by_vision_text_annotations = tpm_response_from_vision.text_annotations
    if not os.path.exists(path_source_toolbox):
        os.makedirs(path_source_toolbox)

    with open(path_source_toolbox + file_source[0:len(file_source)-4] + '_fulltext.json', 'w') as f:  
        f.write(
            json_format.MessageToJson(
                document_annotated_by_vision_full_text_annotation))

#1. Funcion que frontea para dibujar en la imagen
def draw_images(file_source1, color='green'):
    '''
    Funcion que dependiendo de lo deseado dibuja cuadros para bloques, para, words, symbol
    '''
    global file_source
    file_source = file_source1

    vision_client_instance(image_to_use = path_source_client + file_source1)

    tmp_bounds = get_document_bounds(feature = FeatureType.WORD)
    tmp_image_drawed = draw_boxes(bounds = tmp_bounds, color = color)    

#2. Funcion que extrae los boundrys de los textos reconocidos
def text_and_boundy(file_source1):
    '''
    Funcion que extrae los boundrys de los textos reconocidos
    Se asume que la imagen esta ubicada en la carpeta del cliente
    *** Referencias
        https://cloud.google.com/vision/docs/libraries#client-libraries-install-python
        https://cloud.google.com/vision/docs/libraries#client-libraries-install-python
        https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/vision/cloud-client/detect/detect.py
        pip install google-cloud
        pip install google-cloud-vision
        google-cloud-pubsub==0.38.0
        google-cloud-storage==1.13.0
        google-cloud-translate==1.3.1
        google-cloud-vision==0.35.0
        #print('bounds: {}'.format(','.join(vertices)))
        #print('\n"{}"'.format(text.description.encode('utf-8').strip()))
        #print(texts) cuando texts = client.document_text_detection(image=image).text_annotations
    '''

    global file_source
    file_source = file_source1

    vision_client_instance(image_to_use = path_source_client + file_source1)

    f = open(path_source_toolbox + file_source[0:len(file_source)-4] + '_boundies.txt', 'w')
    for text in document_annotated_by_vision_text_annotations:
        # captura los limites del poligono y los pasa a la variable vertice
        vertices = (['({},{})'.format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        
        #Solo imprime el campo descripcion y los limites del poligono
        f.write('\n{} --- {}'.format(text.description.encode('utf-8').strip(), 
            str.replace(str.replace((','.join(vertices)),'(','['),')',']')
                ))
    f.close()

#3. Carga a a firestore la informacion de cada un campo de la factura.
def invoice_field(number, field, level, p00, f11, f12):
    """ 
    Crea los trabajos con firebase
    simpre debe existir por cada tipo de factura
    word_to_find : {type=0, name='', bounding_box: {x1:'', x2:'', y1:'', y2=''}}
    ** Referencia firestore
        https://pythonspot.com/json-encoding-and-decoding-with-python/
        https://github.com/GoogleCloudPlatform/python-docs-samples/blob/19f7f65c7badc37e23ad9f0663da8bd78823a1d7/firestore/cloud-client/snippets.py#L150-L160
        https://googleapis.github.io/google-cloud-python/latest/firestore/index.html
        https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/storage/cloud-client/snippets.py
        https://www.youtube.com/watch?v=yylnC3dr_no
        https://firebase.google.com/docs/reference/admin/?authuser=0
        https://github.com/GoogleCloudPlatform/python-docs-samples/blob/19f7f65c7badc37e23ad9f0663da8bd78823a1d7/firestore/cloud-client/snippets.py#L150-L160
        pip install firebase-admin
    ** Referncia OOP
        https://github.com/CoreyMSchafer/code_snippets/blob/master/Object-Oriented/4-Inheritance/oop-finish.py
    """
    # Definicion de variables con firestore [START] OJO CON ESTO, SE DEBE DESHABILITAR CUANDO FUNCIONA CONJUNTAMENTE CON MILE
    # cred = credentials.Certificate("./proyecto/connection/hdlaml-firebase.json")
    # firebase_admin.initialize_app(credential = cred, name = 'tolbox')
    # firebase_admin.initialize_app(cred)
    db = firestore.client()
    # Definicion de variables con firestore [END]

    # Esta es la ruta de documento en firestore
    document = '{}/{}'.format(number, field)
    
    # Hace las operaciones para calcular la distancia hacia el punto de referencia
    p0_from_vision = np.array(p00)
    p0 = np.append([p0_from_vision.min(axis=0)], [p0_from_vision.min(axis=0)], axis=0)
    f1_from_vision = np.append(np.array(f11), np.array(f12), axis=0)
    f1 = np.append([f1_from_vision.min(axis=0)], [f1_from_vision.max(axis=0)], axis=0)
    pf = f1-p0

    y = ['x1', 'y1', 'x2', 'y2']
    x = pf.flatten()

    data = {
        'bounding_box': dict(zip(y,x)),
        'type': level,
        'name': field
        }

    db.document(document).set(data)
    pass

#4. Detecta los boundries from a word
def find_word_location(file_source1, word_to_find):
    '''
    Funcion que busca la palabra y se da los poligonos
    '''
    global file_source
    file_source = file_source1
    vision_client_instance(image_to_use = path_source_client + file_source1)
    
    for page in document_annotated_by_vision_full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    #Ensambla las palabras
                    assembled_word = ''.join(symbol.text for symbol in word.symbols)
                    if (assembled_word == word_to_find):
                        bounds = [[p.x, p.y] for p in word.bounding_box.vertices]
                        return bounds

#5.1. Detecta los boundries from a word
def find_word_location1(word_to_find):
    '''
    Funcion que busca la palabra y se da los poligonos
    '''
    for page in document_annotated_by_vision_full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    #Ensambla las palabras
                    assembled_word = ''.join(symbol.text for symbol in word.symbols)
                    if (assembled_word == word_to_find):
                        bounds = [[p.x, p.y] for p in word.bounding_box.vertices]
                        return bounds

#5. Carga a a firestore la informacion de cada un campo de la factura, a partir de los
# nombres de las letras a buscar.
def invoice_field1(number, field, level, wr, wf11, wf12, file_source1):
    """ 
    Crea los trabajos con firebase
    simpre debe existir por cada tipo de factura
    word_to_find : {type=0, name='', bounding_box: {x1:'', x2:'', y1:'', y2=''}}
    ** Referencia firestore
        https://pythonspot.com/json-encoding-and-decoding-with-python/
        https://github.com/GoogleCloudPlatform/python-docs-samples/blob/19f7f65c7badc37e23ad9f0663da8bd78823a1d7/firestore/cloud-client/snippets.py#L150-L160
        https://googleapis.github.io/google-cloud-python/latest/firestore/index.html
        https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/storage/cloud-client/snippets.py
        https://www.youtube.com/watch?v=yylnC3dr_no
        https://firebase.google.com/docs/reference/admin/?authuser=0
        https://github.com/GoogleCloudPlatform/python-docs-samples/blob/19f7f65c7badc37e23ad9f0663da8bd78823a1d7/firestore/cloud-client/snippets.py#L150-L160
        pip install firebase-admin
    ** Referencia OOP
        https://github.com/CoreyMSchafer/code_snippets/blob/master/Object-Oriented/4-Inheritance/oop-finish.py
    """
    # Definicion de variables con firestore [START] OJO CON ESTO, SE DEBE DESHABILITAR CUANDO FUNCIONA CONJUNTAMENTE CON MILE
    # cred = credentials.Certificate("./proyecto/connection/hdlaml-firebase.json")
    # firebase_admin.initialize_app(credential = cred, name = 'tolbox')
    # firebase_admin.initialize_app(cred)
    db = firestore.client()
    # Definicion de variables con firestore [END]

    # Esta es la ruta de documento en firestore
    document = '{}/{}'.format(number, field)

    global file_source
    file_source = file_source1
    vision_client_instance(image_to_use = path_source_client + file_source)

    if level == 0:
        p00 = [[0,0],[0,0],[0,0],[0,0]]
        f11 = find_word_location1(word_to_find=wr)
        f12 = f11
    else:
        p00 = find_word_location1(word_to_find=wr)
        f11 = find_word_location1(word_to_find=wf11)
        f12 = find_word_location1(word_to_find=wf12)

    # Hace las operaciones para calcular la distancia hacia el punto de referencia
    p0_from_vision = np.array(p00)
    p0 = np.append([p0_from_vision.min(axis=0)], [p0_from_vision.min(axis=0)], axis=0)
    f1_from_vision = np.append(np.array(f11), np.array(f12), axis=0)
    f1 = np.append([f1_from_vision.min(axis=0)], [f1_from_vision.max(axis=0)], axis=0)
    pf = f1-p0

    y = ['x1', 'y1', 'x2', 'y2']
    x = pf.flatten()

    data = {
        'bounding_box': dict(zip(y,x)),
        'type': level,
        'name': field
        }

    db.document(document).set(data)
    pass

def draw_boxes1(invoice_number, file_source1):
    '''
    De acuerdo a unos limites recibidos dibuja los cuadros en una imagen

    ** Referencias
        https://github.com/monark12/vision-api/blob/master/vision_api.ipynb
        https://medium.com/searce/tips-tricks-for-using-google-vision-api-for-text-detection-2d6d1e0c6361
    '''
    global file_source
    file_source = file_source1
    color = 'red'
    witdth=5

    #Define el objeto PIL a dibujar en el
    img = Image.open(path_source_client + file_source)
    tmp_draw = ImageDraw.Draw(img)

    #Define los objetos de la base de datos
    tmp_detail=[]
    db = firestore.client() # Se inicializa el admin SDK
    collection = db.collection(str(invoice_number)).order_by('type').get()

    #Empieza a iterar sobre los resultados de la base de datos
    for doc in collection:
        tmp_fields = doc.to_dict()

        if tmp_fields['type'] == 0:
            tmp_dot_reference = [tmp_fields['bounding_box']['x1'],
                                tmp_fields['bounding_box']['y1']]
        else:
            tmp_draw.line([
                #Dibuja un cuadrado con los cuatro vertices en la imagen
                tmp_dot_reference[0]+tmp_fields['bounding_box']['x1'], tmp_dot_reference[1]+tmp_fields['bounding_box']['y1'],
                tmp_dot_reference[0]+tmp_fields['bounding_box']['x2'], tmp_dot_reference[1]+tmp_fields['bounding_box']['y1'],
                tmp_dot_reference[0]+tmp_fields['bounding_box']['x2'], tmp_dot_reference[1]+tmp_fields['bounding_box']['y2'],
                tmp_dot_reference[0]+tmp_fields['bounding_box']['x1'], tmp_dot_reference[1]+tmp_fields['bounding_box']['y2'],
                tmp_dot_reference[0]+tmp_fields['bounding_box']['x1'], tmp_dot_reference[1]+tmp_fields['bounding_box']['y1']
            ],
                fill=color, width=witdth
            )

    img.save(path_source_toolbox + file_source[0:len(file_source)-4] + '_drawed' + '.jpg' ,'JPEG')
    pass
