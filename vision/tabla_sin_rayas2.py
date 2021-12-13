
#%%
import os
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

#%%

# lista de entrada a la funcion
li = [
    ('8001539937', 0.6190442, 'data/8001539937_new.jpg'),
    ('8001304263', 0.84001195, 'data/8001304263_new.jpg'),
    ('8110213537', 0.8046723, 'data/2-029_new.jpg'),
    ('8002126635', 0.6552293, 'data/8002126635_new.jpg'),
]

def ocr_detalle(lista):

    for j in lista:

        # Variables
        columnas = {}
        y_min = None
        y_max = None
        x_min = None
        x_max = None
        NIT = j[0]
        FILE = j[2]

        # abre la imagen y toma sus dimensiones
        img = cv2.imread(os.path.join(FILE))
        height, weight, _ = img.shape

        # Busca en la base de datos la factura en cuestion 
        # y trae los datos de todos los campos
        record = db.busca_nit(int(NIT))[0].get('campos')

        for i in record:
            # crea la lista de columnas con ancho para la seccion de detalle
            if 'columna' in i.get('campo'):
                columnas[i.get('campo')] = [{
                    "x_min": int(i.get("x_min")*weight), 
                    "x_max": int(i.get("x_max")*weight)
                    }]

            elif 'detalle' in i.get('campo'):
                y_min = int(i.get('y_min')*height)
                y_max = int(i.get('y_max')*height)
                x_min = int(0)
                x_max = int(i.get('x_max')*weight)

        # Crea fraccion de toda la factura donde es solo el detalle 
        img = img[y_min:y_max , x_min:x_max]
        # Pasa a pytesseract para que reconozca todos los caracteres
        dataframe = pytesseract.image_to_data(
            img
            , output_type=pytesseract.Output.DATAFRAME
            , config='--oem 1 --psm 6')

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
            print('skipped')
            continue

        # Define las filas y las ordena para tener un parametro de recorrido del
        # dataframe
        df_lineas = sorted(dataframe.loc[:, 'line_num'].unique())

        # contenedor de todas las filas
        todo = []

        # Recorre cada una de las filas
        for x in df_lineas:

            # Ordena por la posicion en el eje x(left) para agrupar de izquierda a derecha
            dfxlinea = dataframe[dataframe.loc[:, 'line_num']==x].sort_values(by=['left'])

            # Extracta los nombres de las columnas identificadas
            df_columnas = sorted(dfxlinea.loc[:, 'columna'].unique())

            # Contenedor de cada fila
            registro = {}
            for i in df_columnas:

                # Filtra los elementos comunes a cada columna porque pytesseract 
                # reconoce por grupos de palabras
                dfxlinea_columna = dfxlinea[dfxlinea.loc[:, 'columna']==i]

                # Concatena los textos de la misma linea y categoria
                text_final = ''
                for ii, row in dfxlinea_columna.iterrows():
                    text_final = text_final + row['text'] + ' '

                registro[i] = (
                    # 'left':  dfxlinea_columna.left.min(),
                    # 'top': dfxlinea_columna.top.max(),
                    # 'width': dfxlinea_columna.width.sum(),
                    # 'height': dfxlinea_columna.height.sum(),
                    # 'conf': dfxlinea_columna.conf.max(),
                    text_final
                    )
            todo.append(registro)

        # Quitar los caracteres basura
        dataframe = pd.DataFrame(todo).replace("[\n]|[\x0c]|[,]|[|]|[)({}]|[\]\[]", "", regex=True) #.to_dict('records')

        # Eliminar los regustros que sobran por nan
        dataframe = dataframe[dataframe['columna_valor'].notna()]

        # Resultado para adjuntar al registro resultado
        # dataframe.to_dict('records')
        dataframe
        print(i)
        # mostrar(img.copy(), [(0,0,0,0)])


ocr_detalle(lista=li)

# %%
