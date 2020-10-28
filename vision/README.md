# vision
 
@author:  Hernán Darío López Archila
@email: hernand.lopeza@gmail.com

M.I.L.E
Machine Intelligent Learning Enterprises


@proyecto:	Agente que causa facturas en un ERP
@etapa1:	Reconocimiento de factura e identificacion de la misma
		@nameetapa: vision
		@technology: red convolucional

root:
  |__ data (almacena las imagenes)
  |__ code (codigo a implementar)
  |__ model (modelo de tf salvado)


@updated: enero 11 2020

@comandos de inicio

@track:
Contenedor
 - no se puede usar la imagen nvcr.io/nvidia/tensorflow porque la version de tf es la 1.14
 - no se puede usar la imagen nvidia/cuda porque no tiene version de python
 - se debe instalar pillow pylint jedi

 @flujo

 a. Son las librerias que se requieren para que el codigo funcione
 b. Son las librerias de preprocesamiento de datos

