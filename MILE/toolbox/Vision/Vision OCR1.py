# https://cloud.google.com/vision/docs/libraries#client-libraries-install-python
# https://cloud.google.com/vision/docs/libraries#client-libraries-install-python
# https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/vision/cloud-client/detect/detect.py
# pip install google-cloud
# pip install google-cloud-vision
# google-cloud-pubsub==0.38.0
# google-cloud-storage==1.13.0
# google-cloud-translate==1.3.1
# google-cloud-vision==0.35.0

import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./connection/apikey.JSON"

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.join(
    os.path.dirname("__file__"),
    './images/4.png')

# Loads the image into memory
with io.open(file_name,'rb') as image_file:
    content = image_file.read()
    
image = vision.types.Image(content=content)

# Performs text detection on the image file
response = client.document_text_detection(image=image)

for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                print('Paragraph confidence: {}'.format(paragraph.confidence))

                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols]).encode('utf-8').strip()
                    print('Word text: {} (confidence: {})'.format(word_text, word.confidence))
                
                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(symbol.text.encode('utf-8').strip(), symbol.confidence))
