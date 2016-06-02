import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
from antk.core import node_ops
import tensorflow as tf
import numpy


os.environ["CUDA_VISIBLE_DEVICES"] = ''

listtensor = [[[1.0,13],[4,16], [7,19], [10,22]],
              [[2,14],[5,17],[8,20],[11,23]],
              [[3,15],[6,18],[9,21],[12,24]]]

numpytensor = numpy.array(listtensor)
print("tensor from kolda and bader in python notation (shape (3,4,2)): \n%s\n" % numpytensor)
# follows kolda and bader notation
print("row zero holding fiber fixed at zero (notation t_(0,:,0)): \n%s\n" % numpytensor[0,:,0])
print("column zero holding fiber fixed at zero (notation t_(:,0,0)): \n%s\n" % numpytensor[:,0,0])
print("fiber zero holding column fixed at zero (notation t_(0,0,:)): \n%s\n" % numpytensor[0,0,:])


t = tf.constant(numpytensor.astype(numpy.float64))

t0 = node_ops.nmode_tensor_tomatrix(t, 0)
t1 = node_ops.nmode_tensor_tomatrix(t, 1)
t2 = node_ops.nmode_tensor_tomatrix(t, 2)

# test if nmode vector multipliation works as expected
minus0 = numpy.array([[1.0, 1, 1],])
minus0t = tf.constant(minus0.astype(numpy.float64))
mult0minus = node_ops.nmode_tensor_multiply([t, minus0t], 0)

minus1 = numpy.array([[1.0, 1, 1,1],])
minus1t = tf.constant(minus1.astype(numpy.float64))
mult1minus = node_ops.nmode_tensor_multiply([t, minus1t], 1)

minus2 = numpy.array([[1.0, 1],])
minus2t = tf.constant(minus2.astype(numpy.float64))
mult2minus = node_ops.nmode_tensor_multiply([t, minus2t], 2)

# test if nmode matrix multiplication works as expected
double0 = numpy.array([[2.0, 0, 0],[0,2,0],[0,0,2]])
double0t = tf.constant(double0.astype(numpy.float64))
mult0double = node_ops.nmode_tensor_multiply([t, double0t], 0)

double1 = numpy.array([[2.0, 0, 0, 0],[0,2,0, 0],[0,0,2,0], [0,0,0,2]])
double1t = tf.constant(double1.astype(numpy.float64))
mult0double = node_ops.nmode_tensor_multiply([t, double1t], 1)

double2 = numpy.array([[2.0, 0],[0,2]])
double2t = tf.constant(double2.astype(numpy.float64))
mult0double = node_ops.nmode_tensor_multiply([t, double2t], 2)


mat1 = [[1.0,2,3],[1,2,3]]
mat2 = [[4.0,8],[9,7]]
mat1 = tf.constant(numpy.array(mat1))
mat2 = tf.constant(numpy.array(mat2))

newmat = node_ops.binary_tensor_combine([mat1,mat2], output_dim=3)
newmat2 = node_ops.binary_tensor_combine2([mat1,mat2], output_dim=3)

sess = tf.Session()
tf.set_random_seed(500)

print("binary tensor combine (outputsize 3): \nmat1 (shape = %s):\n%s\nmat2 (shape = %s)\n%s\nnewmat (shape=%s)\n%s" % (mat1.get_shape().as_list(),sess.run(mat1),mat2.get_shape().as_list(), sess.run(mat2),newmat.get_shape().as_list(), sess.run(newmat)))

print("binary tensor combine (outputsize 3): \nmat1 (shape = %s):\n%s\nmat2 (shape = %s)\n%s\nnewmat (shape=%s)\n%s" % (mat1.get_shape().as_list(),sess.run(mat1),mat2.get_shape().as_list(), sess.run(mat2),newmat2.get_shape().as_list(), sess.run(newmat)))


print("0 mode matricization (columns): \n%s\n" % sess.run(t0))
print("1 mode matricization (rows): \n%s\n" % sess.run(t1))
print("2 mode matricization (fibers): \n%s\n" % sess.run(t2))

# print vector multiply results
print("original shape: %s    orignal tensor: \n%s\n" % (numpytensor.shape, numpytensor))

result0minus = sess.run(mult0minus)
print("mode 0 vector multiply shape: %s    tensor: \n%s\n" % (result0minus.shape, result0minus))

result1minus = sess.run(mult1minus)
print("mode 1 vector multiply shape: %s    tensor: \n%s\n" % (result1minus.shape, result1minus))

result2minus = sess.run(mult2minus)
print("mode 2 vector multiply shape: %s    tensor: \n%s\n" % (result2minus.shape, result2minus))

# print matrix multipy results
print("original shape: %s    orignal tensor: \n%s\n" % (numpytensor.shape, numpytensor))

t0minus = sess.run(mult0minus)
print("mode 0 vector multiply shape: %s    tensor: \n%s\n" % (result0minus.shape, result0minus))

t1minus = sess.run(mult1minus)
print("mode 1 vector multiply shape: %s    tensor: \n%s\n" % (result1minus.shape, result1minus))

tminus = sess.run(mult2minus)
print("mode 2 vector multiply shape: %s    tensor: \n%s\n" % (result2minus.shape, result2minus))
