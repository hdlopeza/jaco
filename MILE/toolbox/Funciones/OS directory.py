# "
# https://docs.python.org/2/library/filesys.html
# https://stackoverflow.com/questions/5899497/checking-file-extension
# https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
# "


import fnmatch
import shutil
import os

path_source = './proyecto/images/'
directory = path_source + 'M.I.L.E/'

# Crea la carpeta de donde finalmente se tomaran las imagenes

if not os.path.exists(directory):
        os.makedirs(directory)
        [os.copy(path_source + file, directory + file)
        if fnmatch.fnmatch(file, '*.png') or 
                fnmatch.fnmatch(file, '*.pdf') or 
                fnmatch.fnmatch(file, '*.jpg') 
        else 0
        for file in os.listdir(path_source)]

#os.rename