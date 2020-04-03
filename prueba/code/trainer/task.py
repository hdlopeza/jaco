import argparse
from . import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--texto',
        type = str)
    
    parser.add_argument(
        '--otro',
        type = str)

    args = parser.parse_args()
    arguments = args.__dict__

    model.funcion(arguments)