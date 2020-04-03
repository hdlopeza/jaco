""" 
Usage:
    split_folders folder_with_images [--ratio]  [--seed]
Options:
    --ratio      the ratio to split. e.g. for train/val/test `.8 .1 .1` or for train/val `.8 .2`.
    --seed       set seed value for shuffling the items. defaults to 123.
Return:
    a dictionary shuffled with
    [0] label o folder that contains that file
    [1] code unique for each label

    a list shuffled with
    [0] Route of file
    [1] label o folder that contains that file
    [2] code unique for each label
    [3] if file is train/test/val set
Example:
    split_folders imgs --ratio .8 .1 .1 
author: @hdlopeza (Hernan Lopez Archila, hernand.lopeza@gmail.com)
"""

import pathlib
import random
import math
import numpy as np

def _list_dirs(directory):
    """
    Returns all directories in a given directory
    there are class_dirs
    """
    return [
        f 
        for f in pathlib.Path(directory).iterdir() 
        if f.is_dir()]

def _list_files(directory):
    """
    Returns all files in a given directory
    """
    return [
        str(f)
        for f in pathlib.Path(directory).iterdir()
        if f.is_file() and not f.name.startswith(".")
    ]

def _setup_files(class_dir, seed):
    """
    Returns shuffled files
    """
    # make sure its reproducible
    random.seed(seed)
    files = _list_files(class_dir)
    files.sort()
    random.shuffle(files)
    return files

def _split_files(files, split_test, split_val):
    """
    Splits the files along the provided indices
    """
    # Se puede usar pathlib.Path(_).parent.name
    files_test = [[_ , pathlib.Path(_).parts[-2], 'test'] for _ in files[:split_test]]
    files_val = [[_ ,  pathlib.Path(_).parts[-2],'val'] for _ in files[split_test:split_val]]
    files_train = [[_ ,  pathlib.Path(_).parts[-2],'train'] for _ in files[split_val:]]
    l=[]
    l.extend(files_test)
    l.extend(files_val)
    l.extend(files_train)

    #return files_test, files_val, files_train
    return l

def _split_class_dir_ratio(class_dir, ratio, seed):
    """
    Splits one very class folder
    """
    files = _setup_files(class_dir, seed)

    split_test = math.ceil(ratio[2] * len(files))
    split_val = math.ceil(ratio[1] * len(files)) + split_test

    return _split_files(files, split_test, split_val)

def ratio(folder, seed=123, ratio=(0.8, 0.1, 0.1)):
    """ 
    Clasifica en una lista [,2], un directorio de archivos, preferiblemente de imagenes
    entre test, validacion y entrenamiento, asegurando que como minimo cada set contenga
    una imagen """
    l = []
    for class_dir in _list_dirs(folder):
        l.extend(_split_class_dir_ratio(class_dir, ratio, seed))

    # Maestro de carpetas padre con indice en formato
    # diccionario
    y = np.array(l)
    y = np.unique(y[:,1])
    y = dict((_, idx) for idx, _ in enumerate(y))

    # Modifica la lista de files y adiciona el indice(codigo) de la
    # carpeta padre, este ultimo codigo es el que se le pasara a la NN
    l = [[
                _[0],
                _[1],
                y[pathlib.Path(_[0]).parent.name],
                _[2]
            ]
        for _ in l]

    return y, l
