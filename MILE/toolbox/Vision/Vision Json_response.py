
import json

with open('proyecto/images/MILE/toolbox/7-1_fulltext.json') as fp:
    document = json.load(fp)

for page in document['pages']:
    for block in page['blocks']:
        for paragraph in block['paragraphs']:
            for word in paragraph['words']:
                '''print word['boundingBox']['vertices'][0]'''