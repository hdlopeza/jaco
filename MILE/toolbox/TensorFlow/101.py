#%% [markdown]
# # Tensorflow aprendizaje
# La idea es crear un codigo unico que me permita trabajar
# con cualquier tipo de red a diseñar
# 
# Pasos:
# 1. Crear el grafico
# 2. Ejecutar el grafico
# 
# Ejemplo de creacion de un grafico en lazzy mode#

#%%
# Ejemplo de creacion de un grafico en lazzy mode#
import tensorflow as tf
a = tf.constant([5, 3, 8])
b = tf.constant([3, -1, 2])
c = tf.add(a, b)
d = a * b
e = d ** 2
print(c)

with tf.Session() as sess:
    result = sess.run(c)
    result2 = c.eval()
    result3, result4 = sess.run([d, e])
    print(result)
    print(result2)
    print(result3)
    print(result4)

#%%
# Ejemplo de creacion de un grafico en eager mode
# y visualizar el grafo#
import tensorflow as tf
x = tf.constant([2,5,7], name='x')
y = tf.constant([3,4,8], name='y')
z1 = tf.add(x, y, name='Suma')
z2 = x * y
z3 = z2 - z1

with tf.Session() as sess:
    with tf.summary.FileWriter('summaries', sess.graph) as writer:
        a1, a3 = sess.run([z1, z3])

#%%
# Usar tensorboard para ver el grafo escrito funciona desde terminal
'''%%bash
tensorboard --logdir='./summaries'''

#%%
# Ejemplo de creacion de un grafico en eager mode#
import tensorflow as tf
tf.enable_eager_execution()
print(
    tf.reduce_sum(
        tf.random_normal([1000, 1000])))

#%%
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()
x = tf.constant([2,5,7])
y = tf.constant([3,4,8])
print(x-y)

#%%  [markdown]
# Los placeholders se alimentan mientras esta la sesion ejecutandose, con feed_dict(key es placeholder, and array of values), lista o array de numpy 
# Las variables se crean con alcance y pueden cambiar de valor

#%%  [markdown]
# # Reiniciar el kernel


#%%
# Feed a session
def compute_area(sides):
  # slice the input to get the sides
  a = sides[:,0]  # 5.0, 2.3
  b = sides[:,1]  # 3.0, 4.1
  c = sides[:,2]  # 7.1, 4.8
  
  # Heron's formula
  s = (a + b + c) * 0.5   # (a + b) is a short-cut to tf.add(a, b)
  areasq = s * (s - a) * (s - b) * (s - c) # (a * b) is a short-cut to tf.multiply(a, b), not tf.matmul(a, b)
  return tf.sqrt(areasq)

with tf.Session() as sess:
  sides = tf.placeholder(tf.float32, shape=(None, 3))  # batchsize number of triangles, 3 sides
  area = compute_area(sides)
  result = sess.run(area, feed_dict = {
      sides: [
        [5.0, 3.0, 7.1],
        [2.3, 4.1, 4.8]
      ]
    })
  print(result)

#%%
#Cambiar el nivel de verbosidad 
# tf.logging.set_verbosity(tf.logging.INFO), FATAL ES EL MAS SILENCIOSO#