
# %%
import os
import re
import cv2
import pytesseract
import pandas as pd
import app_db as db
import app_vision
from app_misc import mostrar
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [30, 10]

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# https://github.com/UB-Mannheim/tesseract/wiki

# %%


def ocr_campos(imagen, record, height, weight):

    results = {}
    for ii in record:

        y_min = int(ii.get('y_min') * height)
        y_max = int(ii.get('y_max') * height)
        x_min = int(ii.get('x_min') * weight)
        x_max = int(ii.get('x_max') * weight)

        roi = imagen[y_min:y_max, x_min:x_max]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        results[ii.get('campo')] = re.sub(r"[\n]|[\x0c]|[,]", "",
                                          pytesseract.image_to_string(roi, config='--oem 1 --psm 6'))

    return results


def ocr_detalle(imagen, record, height, weight):

    columnas = {}  # contenedor de los limites de las columnas en el eje x
    todo = []  # contenedor de todas las filas
    roi = []  # Area de interes de la seccion de detalle

    for i in record:
        # crea la lista de columnas con ancho para la seccion de detalle
        if 'columna' in i.get('campo'):
            columnas[i.get('campo')] = [{
                "x_min": int(i.get("x_min")*weight),
                "x_max": int(i.get("x_max")*weight)
            }]

        # crea el area del detalle
        elif 'detalle' in i.get('campo'):
            y_min = int(i.get('y_min')*height)
            y_max = int(i.get('y_max')*height)
            x_min = int(0)
            x_max = int(i.get('x_max')*weight)

            # Crea fraccion de toda la factura donde es solo el detalle
            roi = imagen[y_min:y_max, x_min:x_max]
    
    if len(roi) == 0:
        print('skpd por que no hay roi')
        return None

    # Pasa a pytesseract para que reconozca todos los caracteres 1
    dataframe = pytesseract.image_to_data(
        roi, output_type=pytesseract.Output.DATAFRAME, config='--oem 1 --psm 6')

    # Eliminar los datos NaN de cualquier columna
    dataframe = dataframe.dropna()

    # Crea nueva columna donde se almacenara el nombre de la columna a la
    # que pertenece
    dataframe.loc[:, 'columna'] = None

    # A cada fila o grupo de palabras encontrada dependiendo de los limites del template
    # de las columnas del area de detalle, la clasifica
    # Itera en cada registro de columnas,
    for i in columnas:
        # Captura los limites izquierdo y derecho de cada columna
        x_min = columnas.get(i)[0].get('x_min')
        x_max = columnas.get(i)[0].get('x_max')
        columna_nombre = i

        # Itera en cada fila del dataframe y actualiza
        for ii, row in dataframe.iterrows():
            xmin = row['left']
            xmax = xmin + row['width']

            # Si el campo esta en los limites de la columna agrega el nombre
            if (xmin >= x_min) and (xmax <= x_max):
                dataframe.loc[ii, 'columna'] = columna_nombre
            else:
                pass

    dataframe = dataframe.dropna()
    if dataframe.empty:
        print('skpd porque dataframe vacio')
        return None

    # Define las filas y las ordena para tener un parametro de recorrido del
    # dataframe
    df_lineas = sorted(dataframe.loc[:, 'line_num'].unique())

    # Recorre cada una de las filas dek dataframe
    for x in df_lineas:

        # Ordena por la posicion en el eje x(left) para agrupar de izquierda a derecha
        dfxlinea = dataframe[dataframe.loc[:, 'line_num'] == x].sort_values(by=[
                                                                            'left'])

        # Extracta los nombres de las columnas identificadas
        df_columnas = sorted(dfxlinea.loc[:, 'columna'].unique())

        # Contenedor de cada fila
        registro = {}
        for i in df_columnas:

            # Filtra los elementos comunes a cada columna porque pytesseract
            # reconoce por grupos de palabras
            dfxlinea_columna = dfxlinea[dfxlinea.loc[:, 'columna'] == i]

            # Concatena los textos de la misma linea y categoria
            text_final = ''
            for ii, row in dfxlinea_columna.iterrows():
                text_final = text_final + row['text'] + ' '

            registro[i] = text_final
        todo.append(registro)

    # Quitar los caracteres basura
    dataframe = pd.DataFrame(todo).replace(
        "[\n]|[\x0c]|[,]|[|]|[)({}]|[\]\[]", "", regex=True)  # .to_dict('records')

    # Eliminar los regustros que sobran por nan
    try:
        dataframe = dataframe[dataframe['columna_valor'].notna()]
    except KeyError:
        print('skpd dataframe no tiene columna valor')

    # Resultado para adjuntar al registro resultado
    return dataframe.to_dict('records')

# %%
