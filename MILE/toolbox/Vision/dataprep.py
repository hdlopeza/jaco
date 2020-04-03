import os, re, fnmatch, shutil
from pdf2image import convert_from_path

# Definicion de variables [START]
path_source_client = '/Users/hernandariolopezarchila/Downloads/Datasets/invoices/pdf1/' # OK debe ir el / al final
path_dest_client = '/Users/hernandariolopezarchila/Downloads/Datasets/invoices/images/' # OK debe ir el / al final
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
    # Si en la carpeta de origen hay archivos pdf los convierte a imagen jpg y los copia a la nueva ubicacion
    [   convert_from_path(pdf_path=path_source_client + file, 
                          output_folder=path_dest_client, 
                          output_file=file[0:len(file)-4], 
                          dpi=250, 
                          fmt='JPEG')
        for file in os.listdir(path_source_client)
        if fnmatch.fnmatch(file, '*.pdf')
    ]
    return print('Archivos entregados a M.I.L.E')

def rename_pdf():
  files = os.listdir(path_source_client)

  for index, file in enumerate(files):
      os.rename(os.path.join(path_source_client, file), os.path.join(path_source_client, str(index)+'.pdf'))
  return print('Archivos pdf cambiados de nombre: ok')

file_to_MILE()
