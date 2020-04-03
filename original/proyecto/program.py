import agent_MILE as K
import agent_MILE_toolbox as Kt

a = 6
if a == 1:
   # Toma los archivos, los convierte a pdf y deja las imagenes
   print(K.file_to_MILE())
elif a == 2:
   # Toma todos los archivos de la ruta y reconoce las facturas
   print(K.invoices_decode(
      K.ml_invoices_numbers_and_paths()))
elif a == 3:
   # reconoce una factura esta en la carpeta del cliente, y genera un txt y json con los poligonos
   Kt.text_and_boundy(file_source1='7-1.jpg')
elif a == 4:
   # # toma una factura, de la carpeta del cliente, y le dibuja los poligos que puede reconocer gvision en amarillo
   Kt.draw_images(file_source1='7-1.jpg')
elif a==5:
   # # toma una factura, de la carpeta del cliente, y le dibuja los poligos que va a reconocer en rojo
   Kt.draw_boxes1(file_source1='7-1.jpg', invoice_number=1088000426)
elif a==6:
   #Encuentra el poligono de una palabra en particular, no se puede palabra compuesta
   print(Kt.find_word_location(file_source1='7-1.jpg', word_to_find='LEIDY'))
elif a==7:
   # Carga datos en firestore
   # ojo para que funcione debe estar bien configurada cada factura con el campo '0' de busqueda
   Kt.invoice_field1(number=1088000426,
   field=u'LEIDY',
   level=0,
   wr='LEIDY',
   wf11='3113366',
   wf12='3113366',
   file_source1='7-1.jpg'
   )
elif a==8:
   # Carga datos en firestore
   # ojo para que funcione debe estar bien configurada cada factura con el campo '0' de busqueda
   Kt.invoice_field(number = 1088000426, 
      field = u'LEIDY', 
      level = 0, 
      p00 = [[791,42],[877,42],[877,76],[791,76]], 
      f11 = [[791,42],[877,42],[877,76],[791,76]], 
      f12 = [[791,42],[877,42],[877,76],[791,76]]
      )
