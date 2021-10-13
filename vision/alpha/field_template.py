"""
Toma un cuadro de un sector y lo convierte

https://www.pyimagesearch.com/2020/09/07/ocr-a-document-form-or-invoice-with-tesseract-opencv-and-python/
"""

#%%

import os
import cv2
from collections import namedtuple
from align import align_images
import pytesseract

# %%

OCRLocation = namedtuple("OCRLocation", ["id", "bbox"])

# [x, y, w, h]
OCR_LOCATIONS = [
    OCRLocation("prov_name", (209, 345, 415, 27)),
    OCRLocation("prov_id", (209, 443, 415, 27))
]

# %%

img = align_images(
    image=os.path.join("data", r'8_rotated1.jpg'), 
    template=os.path.join("data", r'8.jpg')
    )

#%%

results = {}

# loop over the locations of the document we are going to OCR
for loc in OCR_LOCATIONS:
    # extract the OCR ROI from the aligned image
    (x, y, w, h) = loc.bbox
    roi = img[y:y + h, x:x + w]

    # OCR the ROI using Tesseract
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(rgb)
    results[loc.id] = (text, loc._asdict())

results

# %%
