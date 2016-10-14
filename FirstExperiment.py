



#	First Experiment with TensorFlow in a Python environment

#	The experiment just gets the scalar product of two matrixes using the TensorFlow workspace

import tensorflow as tf


matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
product = tf.matmul(matrix1, matrix2)


# Launch the default graph.
sess = tf.Session()

result = sess.run(product)
print(result)


sess.close()