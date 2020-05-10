# jaco
@author:  Hernán Darío López Archila
@email: hernand.lopeza@gmail.com

M.I.L.E
Machine Intelligent Learning Enterprises


@proyecto:	Agente que causa facturas en un erp
@etapa1:	Reconocimiento de factura e identificacion de la misma
		@nameetapa: vision
		@technology: red convolucional

@etapa2:	Lectura del documento e identifiacion de entidades
		@nameetapa: ocr
		@technology: 3rd party, google vision

@etapa3:	Asociacion de entidades a cuenta contable
		@nameetapa: nlp
		@technology: use a nlp or rnn

@directory

root:
  |__ vision
  |__ MILE (aun no se que hay)
  |__ NLP nlp (aun no se que hay)
  |__ OCR (aun no se que hay)
  |__ original (aun no se que hay)
  |__ prueba (aun no se que hay)
  |__ images (este se migrará a vision una vez este funcional en tf20)


@updated: enero 11 2020

@track:
enero11-2020 : reinicio del proyecto con documentación, y sobre la etapa1 vision, aqui hay buenas perspectivas de terminar rápido dado el GPU RTX2070, sin embargo primero es reconfigurar lo hecho anteriormente en tf20

febrero04-2020 : 
- ya durante este mes desarrolle el tfrecords haciendo mas eficiente el almacenamiento en este tipo de archivo, y el reshape de las facturas, hay que pensar si las imagenes serán jpeg o tiif, mas aun en que resolución se debe escanear para evitar perdida de información.  Definir el ratio de una imagen tamaño oficio.
- probé que una red sencilla aceptara imagenes en 224*224*3 y 900*700*3
- La etapa que sigue es crear un tf record completo, además entrenar con la base de datos.
- recordar que las categorias están desbalanceadas, probar inicialmente si así converge la red, en caso contrario aumentar datos a la máxima cantidad de muestras que se tenga por categoria.

abril 08-2020
- En marzo 16 vendi la RTX2070, que funcionó bien para desarrollo, ahora intentaré desarrollar en local con la mac y probar robusto en GCP
- Para "Vision" Lo último realizado fue pruebas de convergencia con una muestra de 10 categorias de fotos, el accuracy estaba sobre el 90%
- Sigue focalizarse en Vision, probar que funcione el código en mac bajo contenedor y estandarizar el dockerfile ante librerias que se requieran especiales.

mayo 09-2020
- Tras un mes muerto hoy reinicio operaciones con una nueva maquina