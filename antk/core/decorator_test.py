from __future__ import print_function
import tensorflow as tf
from antk.core import node_ops
import numpy

p = node_ops.placeholder(tf.float32, [5,5], name='test')
# print(p)
print('string rep: %s' % p)
print('rep rep: %s' % p)

x = node_ops.weights('constant', [5,5], initrange=5.0, name='test')
# x = tf.Variable(tf.constant(5.0, shape=[5,5], dtype=tf.float32))
print(x)
out = tf.matmul(p,x)
cos = node_ops.cosine([x,p])
input = numpy.ones((5,5), dtype=numpy.float32)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print(sess.run(out, feed_dict={p: input}))
print(sess.run(cos, feed_dict={p: input}))
print(tf.get_collection('cosine'))
print(tf.get_collection('test_weights'))
