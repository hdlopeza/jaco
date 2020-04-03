import tensorflow as tf
import tensorflow_hub as tfhub 

module = "https://tfhub.dev/google/nnlm-en-dim128/1"
text_sentences = ["perro", "Los cachorros son agradables.", "Disfruto de dar largos paseos por la playa con mi perro."]


g = tf.Graph()
with g.as_default():
	text_input = tf.placeholder(dtype=tf.string, shape=[None])

	embedd_nnlm = tfhub.Module(module)
	embedd_text = embedd_nnlm(text_input)

	init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()	

session = tf.Session(graph=g)
session.run(init_op)

result = session.run(embedd_text, feed_dict={text_input: text_sentences})

print(result)

feature_column = tfhub.text_embedding_column(key='sentence',
	                                             module_spec=module,
	                                             trainable=False)
print(feature_column)
