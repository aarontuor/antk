from antk.core import node_ops
import tensorflow as tf
import numpy as np

nptensor = np.array([[[1.0,13],[4,16], [7,19], [10,22]],
                            [[2,14],[5,17],[8,20],[11,23]],
                            [[3,15],[6,18],[9,21],[12,24]]])

t = tf.constant(nptensor.astype(np.float64))

t0 = node_ops.nmode_tensor_tomatrix(t, 0)
t1 = node_ops.nmode_tensor_tomatrix(t, 1)
t2 = node_ops.nmode_tensor_tomatrix(t, 2)

U = np.array([[1,3,5],[2,4,6]])
# test if nmode vector multipliation works as expected
minus0 = np.array([[1.0, 1, 1],])
minus0t = tf.constant(minus0.astype(np.float64))
mult0minus = node_ops.nmode_tensor_multiply([t, minus0t], 0)

minus1 = np.array([[1.0, 1, 1,1],])
minus1t = tf.constant(minus1.astype(np.float64))
mult1minus = node_ops.nmode_tensor_multiply([t, minus1t], 1)

minus2 = np.array([[1.0, 1],])
minus2t = tf.constant(minus2.astype(np.float64))
mult2minus = node_ops.nmode_tensor_multiply([t, minus2t], 2)

# test if nmode matrix multiplication works as expected
double0 = np.array([[2.0, 0, 0],[0,2,0],[0,0,2]])
double0t = tf.constant(U.astype(np.float64))
mult0double = node_ops.nmode_tensor_multiply([t, double0t], 0)

double1 = np.array([[2.0, 0, 0, 0],[0,2,0, 0],[0,0,2,0], [0,0,0,2]])
double1t = tf.constant(double1.astype(np.float64))
mult1double = node_ops.nmode_tensor_multiply([t, double1t], 1)

double2 = np.array([[2.0, 0],[0,2]])
double2t = tf.constant(double2.astype(np.float64))
mult2double = node_ops.nmode_tensor_multiply([t, double2t], 2)


mat1 = [[1.0,2,3],[1,2,3]]
mat2 = [[4.0,8],[9,7]]
mat1 = tf.constant(np.array(mat1))
mat2 = tf.constant(np.array(mat2))

newmat = node_ops.binary_tensor_combine([mat1,mat2], output_dim=3, name='newmat')
newmat2 = node_ops.binary_tensor_combine2([mat1,mat2], output_dim=3, name='newmat2')

sess = tf.Session()
tf.set_random_seed(500)

init = tf.initialize_all_variables()

sess = tf.Session()
tf.set_random_seed(500)
sess.run(init)
# def test_binary_tensor_combine_shape():
#     t = sess.run(newmat)
#     assert newmat.get_shape().as_list() == [2,3]
#
def test_binary_tensor_combine_shape():
    t = sess.run(newmat)
    assert np.array_equal(t,
                          np.array([[-3.34240468e-04,-5.96691073e-04,  -6.43865472e-05],
                            [ -5.51830158e-04,  -6.21576993e-04,  -6.23402685e-05]]))

def test_1_mode_matricize():
    t = sess.run(t1)
    assert np.array_equal(t, np.array([[  1.,   2.,   3.,  13.,  14.,  15.],
                                        [  4.,   5.,   6.,  16.,  17.,  18.],
                                        [  7.,   8.,   9.,  19.,  20.,  21.],
                                        [ 10.,  11.,  12.,  22.,  23.,  24.]]))

def test_0_mode_matricize():
    t = sess.run(t0)
    assert np.array_equal(t, np.array([[1.,   4.,   7.,  10.,  13.,  16.,  19.,  22.],
                                       [2.,   5.,   8.,  11.,  14.,  17.,  20.,  23.],
                                       [3.,   6.,   9.,  12.,  15.,  18.,  21.,  24.]]))

def test_2_mode_matricize():
    t = sess.run(t2)
    assert np.array_equal(t,
                          np.array([[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.],
                                    [ 13.,14.,15.,16.,17.,18.,19.,20.,21.,22.,23.,24.]]))
#
#
# def test_mode_0_ones_vec_multiply():
#     t = sess.run(mult0minus)
#     assert t.shape == (4,2)
#     assert np.array_equal(t,
#                           np.array([[  6.,  42.]
#                                      [ 15.,  51.],
#                                      [ 24.,  60.],
#                                      [ 33.,  69.]]))
#
# def test_mode_1_ones_vec_multiply():
#     t = sess.run(mult0minus)
#     assert t.shape == (3,2)
#     assert np.array_equal(t,
#                           np.array([[  6.,  42.]
#                                      [ 15.,  51.],
#                                      [ 24.,  60.],
#                                      [ 33.,  69.]]))
#
# def test_mode_2_ones_vec_multiply():
#     t = sess.run(mult0minus)
#     assert t.shape == (3,4)
#     assert np.array_equal(t,
#                           np.array([[ 14.,  20.,  26.,  32.],
#                                      [ 16.,  22.,  28.,  34.],
#                                      [ 18.,  24.,  30.,  36.]]))
#
# def test_mode_1_ones_mat_multiply():
#     t = sess.run(mult1double)
#     assert t.shape == (4,3,2)
#     assert np.array_equal(t,
#                           np.array([[[  2.,  26.],
#                                       [  4.,  28.],
#                                       [  6.,  30.]],
#                                      [[  8.,  32.]
#                                       [ 10.,  34.],
#                                       [ 12.,  36.]],
#                                      [[ 14.,  38.],
#                                       [ 16.,  40.],
#                                       [ 18.,  42.]],
#                                      [[ 20.,  44.],
#                                       [ 22.,  46.],
#                                       [ 24.,  48.]]]))
#
# def test_mode_0_ones_mat_multiply():
#     t = sess.run(mult0double)
#     assert t.shape == (2,4,2)
#     assert np.array_equal(t,
#                           np.array([[[  22.,  130.],
#                                       [  49.,  157.],
#                                       [  76., 184.],
#                                       [ 103.,  211.]],
#                                      [[  28.,  172.],
#                                       [  64.,  208.],
#                                       [ 100.,  244.],
#                                       [ 136.,  280.]]]))
#
# def test_mode_2_ones_mat_multiply():
#     t = sess.run(mult2double)
#     assert t.shape == (3,4,2)
#     assert np.array_equal(t,
#                           np.array([[[  2.,  26.],
#                                       [  8.,  32.],
#                                       [ 14.,  38.],
#                                       [ 20.,  44.]],
#                                      [[  4.,  28.],
#                                       [ 10.,  34.],
#                                       [ 16.,  40.],
#                                       [ 22.,  46.]],
#                                      [[  6.,  30.],
#                                       [ 12.,  36.],
#                                       [ 18.,  42.],
#                                       [ 24.,  48.]]]))













