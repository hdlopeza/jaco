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

path_source = './proyecto/images/' # OK debe ir el / al final
path_destination = './proyecto/images/' # debe ir el / al final
directory = path_source + 'M.I.L.E/'  # OK
file_source = '7.pdf' 
file_destination = re.search(r'(\w+).(\w+)',file_source).group(1)
image_source = re.search(r'(\w+).(\w+)',file_source).group(1) + '_1.jpg'
gcs_source_uri = 'gs://hdlaml_bucket/9.pdf'
gcs_destination_uri = 'gs://hdlaml_bucket/PREFIX/'

# [INICIO SECCION FIREBASE] Usa a service account
cred = credentials.Certificate("./proyecto/connection/hdlaml-firebase.json")
firebase_admin.initialize_app(cred)
# Se inicializa el admin SDK
db = firestore.client()
# [FIN SECCION FIREBASE] 

# Seccion
# 1 Pasar la cadena de conexion al ento0rno para trabajar con google
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './proyecto/connection/hdlaml-vision.json'

# 2 Definir los parametros del archivo a dibujar o trabajar
path_image_to_open = path_source + image_source
image_opened = Image.open(path_image_to_open)

# 21 Cargar la imagen en la memoria, rb reading binary
with open(path_image_to_open, 'rb') as image_file:
    content = image_file.read()

# 3 Crear el cliente de google
client = vision.ImageAnnotatorClient()

# 31 Pasar la imagen que esta en momoria al objeto de google
image = vision.types.Image(content=content)

# 32 Recibir la respuesta de google con todo lo maximo que encontro en la imagen
response = client.document_text_detection(image=image)

# 33 Documento a trabajar con la respuesta
document = response.full_text_annotation

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5
class Campo:
    def __init__(self, name, x1, x2, y1, y2):
        self.name = name
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
class Document(Campo):
    def __init__(self, Number, vision_document, dot_reference=None): #Como funciona el superinit?
        self.Number = str(Number)
        self.Detail=[]
        self.dot_reference = dot_reference
        self.vision_document = vision_document

        # Hace una conexion a la base de datos y trae todos los documentos relacionados con el numero de factura
        # Tarea como hacer las dos consultas con un solo get()
        collection = db.collection(self.Number).order_by('type').get()
        for doc in collection:
            fields = doc.to_dict()

            if fields['type'] == 0:
                self.dot_reference = find_word_location(self.vision_document, fields['name']) 


            # busca el campo 0= misc, 1 principal, 2 detalle
            if fields['type'] == 1:
                self.Detail.append(
                        #Pasa los datos del poligono al campo para adicionarlo al documento factura
                        Campo(
                            doc.id, 
                            self.dot_reference.x + fields['bounding_box']['x1'], 
                            self.dot_reference.x + fields['bounding_box']['x2'],
                            self.dot_reference.y + fields['bounding_box']['y1'], 
                            self.dot_reference.y + fields['bounding_box']['y2']
                        ))
class invoice:
    """ 
    Crea los trabajos con firebase 

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
    def __init__(self, number, field, level, p00, f11, f12):
        self.number = number
        self.field = field
        self.level = level
        self.document = '{}/{}'.format(self.number, self.field)
        self.p00 = p00
        self.f11 = f11
        self.f12 = f12
    
        self.p0_from_vision = np.array(self.p00)
        self.p0 = np.append([self.p0_from_vision.min(axis=0)], [self.p0_from_vision.min(axis=0)], axis=0)
        self.f1_from_vision = np.append(np.array(self.f11), np.array(self.f12), axis=0)
        self.f1 = np.append([self.f1_from_vision.min(axis=0)], [self.f1_from_vision.max(axis=0)], axis=0)
        self.pf = self.f1-self.p0

        y = ['x1', 'y1', 'x2', 'y2']
        x = self.pf.flatten()

        self.data = {'bounding_box': dict(zip(y,x)),
                'type': self.level,
                'name': self.field
                }

    def to_upload(self):
        db.document(self.document).set(self.data)

    # OJO!!!! Esta sentencia  __repr__ es propensa a eliminar despues que todo este funcionando
    def __repr__(self):
        return u'invoice(number={}, field={}, level={})'.format(
            self.number, self.field, self.level)

#1 Funcion que toma los archivos los convierte a imagen y los deja en carpeta final
def file_to_MILE():
    '''
    Crea la carpeta de donde se tomaran las imagenes y si hay archivos 'pdf' convierte la primera hoja a imagen
    ** Referencias
        https://docs.python.org/2/library/filesys.html
        https://stackoverflow.com/questions/5899497/checking-file-extension
        https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
        https://stackoverflow.com/questions/18383384/python-copy-files-to-a-new-directory-and-rename-if-file-name-already-exists

    Convierte la primera pagina de un archivo pdf a imagen jpg de 250 pdi
    ** Referencias
        https://stackoverflow.com/questions/2693820/extract-images-from-pdf-without-resampling-in-python
        https://github.com/Belval/pdf2image
        pip install {pooper, pdf2image}

    '''
    # Si no esta creado la carpeta de destino la crea
    if not os.path.exists(directory):
        os.makedirs(directory)


    [
        # Si en la carpeta de origen hay archivos png o jpg y los copia a la nueva ubicacion
        shutil.copy(path_source + file, directory + file)
        # Itera sobre todos los archivos que hay en la carpeta de origen 
        for file in os.listdir(path_source)
        if fnmatch.fnmatch(file, '*.png') or fnmatch.fnmatch(file, '*.jpg')
    ]

    # Si en la carpeta de origen hay archivos pdf los convierte a imagen jpg y los copia a la nueva ubicacion
    [   convert_from_path(pdf_path=path_source + file, output_folder=directory, output_file=file, dpi=250, fmt='JPEG', first_page=1, last_page=1)
        for file in os.listdir(path_source)
        if fnmatch.fnmatch(file, '*.pdf')
    ]
file_to_MILE()

#2 cloud pub o storage entrega a ML la imagen, este paso entrega localmente el nit 
# del docuento y la imagen en si misma, uno a uno 
def image_classification_invoice():
    '''
    Trae de Google ML el NIT de la factura(imagen)
    y el archivo de la imagen
    '''
    return (71314590, 'image')



## ******[FUNCIONES SIN ORGANIZAR START]
# Funcion que ensambla las palabras
def assemble_word(word):
    assembled_word = ""
    for symbol in word.symbols:
        assembled_word += symbol.text
    return assembled_word

#Funcion que busca la palabra y se da los poligonos
def find_word_location(document, word_to_find):
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    assembled_word = assemble_word(word)
                    if (assembled_word == word_to_find):
                        return word.bounding_box.vertices[0]

# Funcion que frontea para dibujar en la imagen
def draw_images(path_source, path_destination, file_source, color='yellow'):
    bounds = get_document_bounds(response, FeatureType.WORD)
    image_drawed = draw_boxes(image_opened, bounds, color)    
    image_drawed.save(path_destination + file_destination + '_draw' + '.jpg' ,'JPEG')

# Funcion que extrae los boundrys de los textos reconocidos
def text_and_boundy(path_source, file_source):
    # https://cloud.google.com/vision/docs/libraries#client-libraries-install-python
    # https://cloud.google.com/vision/docs/libraries#client-libraries-install-python
    # https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/vision/cloud-client/detect/detect.py
    # pip install google-cloud
    # pip install google-cloud-vision
    # google-cloud-pubsub==0.38.0
    # google-cloud-storage==1.13.0
    # google-cloud-translate==1.3.1
    # google-cloud-vision==0.35.0

    # Performs text detection on the image file y devuelve en json todo el resultado
    texts = response.text_annotations
    
    for text in texts:
        # captura los limites del poligono y los pasa a la variable vertice
        vertices = (['({},{})'.format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
        
        #Solo imprime el campo descripcion y los limites del poligono
        print(text.description.encode('utf-8').strip() + '\n{}'.format(','.join(vertices)))
        #print('bounds: {}'.format(','.join(vertices)))
        #print('\n"{}"'.format(text.description.encode('utf-8').strip()))
        #print(texts) cuando texts = client.document_text_detection(image=image).text_annotations

# Funcion que devuelve el texto reconocido de un poligono en particular    
def text_recogn(path_source, file_source, x1, x2, y1, y2, namefield):
    # https://cloud.google.com/vision/docs/libraries#client-libraries-install-python
    # https://cloud.google.com/vision/docs/libraries#client-libraries-install-python
    # https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/vision/cloud-client/detect/detect.py
    # pip install google-cloud
    # pip install google-cloud-vision
    # google-cloud-pubsub==0.38.0
    # google-cloud-storage==1.13.0
    # google-cloud-translate==1.3.1
    # google-cloud-vision==0.35.0
    
    # Performs text detection on the image file y devuelve en json todo el resultado
    #texts = client.document_text_detection(image=image)
    #response = client.document_text_detection(image=image)
    #texts = response.text_annotations
    #print(texts) en format json

    text = ""
    for page in document.pages:
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
#text_recogn(path_source=path_source, file_source=image_source, x1=1978, x2=2032, y1=1312, y2=1344)


# # Reconoce una factura completa
def recogn_invoice(number, vision_document, path_source, file_source):
    factura = Document(number, vision_document=vision_document)
    for fac in factura.Detail:
        text_recogn(path_source=path_source, file_source=image_source, x1=fac.x1, x2=fac.x2, y1=fac.y1, y2=fac.y2, namefield=fac.name)
# recogn_invoice(number=71314590, vision_document=document, path_source=path_source, file_source=file_source)

# ******[FUNCIONES SIN ORGANIZAR END]


# Muestra el texto reconocido y los limites
text_and_boundy(path_source=path_source, file_source=image_source)


# [START add_example_data]
# Carga datos en firestore
invoice(number = 71314580, 
    field = u'Campo3', 
    level = 6, 
    p00 = [[791,42],[877,42],[877,76],[791,76]], 
    f11 = [[975,86],[983,86],[983,117],[975,117]], 
    f12 = [[1107,86],[1143,86],[1143,117],[1107,117]]
    ).to_upload()
# [END add_example_data]


# *****************************************************************************************


## ******[MIS SECCTION START]
# Misc 1 Funcion que dependiendo de lo deseado dibuja cuadros para bloques, para, words, symbol
def get_document_bounds(response, feature):
    # Define un array para guardar los limites (bounds)
    bounds = []
    #busca en el documento que respondio google lo que deseamos dibujar y los vertices los agrega al objeto bounds
    for i, page in enumerate(document.pages):
        for block in page.blocks:
            if feature == FeatureType.BLOCK:
                bounds.append(block.bounding_box)
            for paragraph in block.paragraphs:
                if feature == FeatureType.PARA:
                    bounds.append(paragraph.bounding_box)
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if feature == FeatureType.SYMBOL:
                            bounds.append(symbol.bounding_box)
                    if feature == FeatureType.WORD:
                        bounds.append(word.bounding_box)
    return bounds

# Misc 2 Funcion que devuelve la imagen con la informacion graficada
def draw_boxes(image, bounds, color, witdth=5):
    # https://github.com/monark12/vision-api/blob/master/vision_api.ipynb
    # https://medium.com/searce/tips-tricks-for-using-google-vision-api-for-text-detection-2d6d1e0c6361

    #Define el objeto PIL a dibujar en el
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        draw.line([
            #Dibuja un cuadrado con los cuatro vertices en la imagen
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[3].y,
            bound.vertices[3].x, bound.vertices[3].y,
            bound.vertices[0].x, bound.vertices[0].y
        ],
            fill=color, width=witdth
        )
    return image
# # Dibuja sobre una imagen los bloques y guarda la imagen
# draw_images(path_source=path_source,path_destination=path_destination,file_source=image_source)

# Misc 3 Funcion que convierte un pdf a imagen ej gcs
def async_detect_document1(gcs_source_uri, gcs_destination_uri):
    """OCR with PDF/TIFF as source files on GCS"""

    feature = vision.types.Feature(type=vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION)
    
    gcs_source = vision.types.GcsSource(uri=gcs_source_uri)
    input_config = vision.types.InputConfig(gcs_source=gcs_source, mime_type='application/pdf')

    gcs_destination = vision.types.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.types.OutputConfig(gcs_destination=gcs_destination, batch_size=2) # How many pages should be grouped into each json output file.

    async_request = vision.types.AsyncAnnotateFileRequest(features=[feature], input_config=input_config,output_config=output_config)

    # Aqui escribe
    operation = client.async_batch_annotate_files(requests=[async_request])

    print('Waiting for the operation to finish.')
    operation.result(timeout=180)
        # Once the request has completed and the output has been
        # written to GCS, we can list all the output files.
    storage_client = storage.Client()

    match = re.match(r'gs://([^/]+)/(.+)', gcs_destination_uri)
    bucket_name = match.group(1)
    prefix = match.group(2)

    bucket = storage_client.get_bucket(bucket_name=bucket_name)

        # List objects with the given prefix.
    blob_list = list(bucket.list_blobs(prefix=prefix))
    print('Output files:')
    for blob in blob_list:
        print(blob.name)

        # Process the first output file from GCS.
        # Since we specified batch_size=2, the first response contains
        # the first two pages of the input file.
    output = blob_list[0]

    json_string = output.download_as_string()
    response = json_format.Parse(json_string, vision.types.AnnotateFileResponse())

        # The actual response for the first page of the input file.
    first_page_response = response.responses[0]
    annotation = first_page_response.full_text_annotation

        # Here we print the full text from the first page.
        # The response contains more information:
        # annotation/pages/blocks/paragraphs/words/symbols
        # including confidence scores and bounding boxes
    print(u'Full text:\n{}'.format(annotation.text.encode('utf-8').decode('ascii','ignore')))
# # Trae texto reconocido de un archivo PDF alojado en GCS
# async_detect_document1(gcs_source_uri=gcs_source_uri, gcs_destination_uri=gcs_destination_uri)

## ******[MIS SECCTION START]
