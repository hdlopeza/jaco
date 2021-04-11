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
img = img[760:1150, 60:1660]

# mostrar(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# mostrar(img)
img = cv2.GaussianBlur(img, (3, 3), 0)
# mostrar(img)
img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# mostrar(img)
img1 = cv2.bitwise_not(img)
struct = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
img1 = cv2.erode(img1, struct, iterations=1)
img1 = cv2.dilate(img1, struct, anchor=(-1, -1), iterations=6)
# mostrar(img1)
contours, hierarchy = cv2.findContours(img1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
boxes = []
for contour in contours:
    box = cv2.boundingRect(contour)
    h = box[3]

    if 12 < h < 40:
        boxes.append(box)

for box in boxes:
    (x, y, w, h) = box
    cv2.rectangle(img, (x, y), (x + w - 2, y + h - 2), (0, 0, 0), 1)

# mostrar(img)
rows = {}
cols = {}
for box in boxes:
    (x, y, w, h) = box
    col_key = x // 10
    row_key = y // 10
    cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
    rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

table_cells = list(filter(lambda r: len(r) >= 2, rows.values()))
table_cells = [list(sorted(tb)) for tb in table_cells]
table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]
max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

ver_lines = []
for box in table_cells[0]:
    x = box[0]
    y = box[1]
    ver_lines.append((x, y, x, max_y))

(x, y, w, h) = table_cells[0][-1]
ver_lines.append((max_x, y, max_x, max_y))

for line in ver_lines:
    [x1, y1, x2, y2] = line
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
mostrar(img)

# %%

mostrar(img)

# %%
hcol = []
htemp = []
for x, y in enumerate(table_cells[0]):
    try:
        (x1, y1, w1, h1) = htemp[-1]
        (x2, y2, w2, h2) = y
    except:
        pass

    if x == 0 :
        hcol.append(y)
    elif (x2 - (x1 + w1)) > 29:
        hcol.append(y)
    else:
        pass
    htemp.append(y)
            
hcol
# %%
