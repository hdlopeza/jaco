#%%

import os
import cv2

def mostrar(image):
    # Using cv2.imshow() method 
    # Displaying the image 
    cv2.imshow('image', image)
    
    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0) 
    
    #closing all open windows 
    cv2.destroyAllWindows() 

in_file = os.path.join("data", "test3.png")

#%%

img = cv2.imread(os.path.join(in_file))
img = img[785:1150, 70:1620]
imgx = img.copy()


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0)
img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
img1 = cv2.bitwise_not(img)
struct = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
img1 = cv2.erode(img1, struct, iterations=1)
img1 = cv2.dilate(img1, struct, anchor=(-1, -1), iterations=6)
contours, hierarchy = cv2.findContours(img1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
boxes = []
for contour in contours:
    box = cv2.boundingRect(contour)
    h = box[3]

    if 12 < h < 40:
        boxes.append(box)

boxes = list(sorted(boxes, key=lambda r: (r[1], r[0])))
rows = {}
for box in boxes:
    (x, y, w, h) = box
    row_key = y // 10
    rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

boxes = []
htemp = []
for i in rows:
    boxes.append(rows.get(i)[0])
    htemp = [rows.get(i)[0]]
    for box in rows.get(i):
        (x1, y1, w1, h1) = htemp[-1]
        (x2, y2, w2, h2) = box
        (x3, y3, w3, h3) = boxes[-1]
        if ((x2 - x1) == 0):
            pass
        elif ((x2 - (x1 + w1)) > 29):
            boxes.append(box)
        else:
            boxes[-1] = (x3, y3, w2+w3, h2)
        htemp.append(box)


boxes = list(sorted(boxes, key=lambda r: (str(r[0])[:2], r[1])))
cols = {}
for box in boxes:
    (x, y, w, h) = box
    col_key = x // 10
    cols[col_key] = [box] if col_key not in cols else cols[col_key] + [box]

boxes = []
htemp = []
for i in cols:
    boxes.append(cols.get(i)[0])
    htemp = [cols.get(i)[0]]
    for box in cols.get(i):
        (x1, y1, w1, h1) = htemp[-1]
        (x2, y2, w2, h2) = box
        (x3, y3, w3, h3) = boxes[-1]
        if (y2 - y1) == 0:
            pass
        elif ((y2 - (y1 + h1)) > 29):
            boxes.append(box)
        else:
            boxes[-1] = (x2, y1, w1, h1+h2)
        htemp.append(box)

for box in boxes:
    (x, y, w, h) = box
    cv2.rectangle(imgx, (x, y), (x + w, y + h), (0, 0, 255), 1)

mostrar(imgx)


# %%
