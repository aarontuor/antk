from __future__ import division
import tensorflow as tf
import numpy
import scipy.sparse as sps
from antk.core import loader
import numbers

ACTIVATION_LAYERS = 'activation_layers'
NORMALIZED_ACTIVATIONS = 'normalized_activations'

class MissingShapeError(Exception):
    '''Raised when :any:`placeholder` can not infer shape.'''
    pass

def tanhlecun(tensor_in):
    """
    `Efficient BackProp`_
    Sigmoid with the following properties:
    (1) :math:`f(\pm 1) = \pm 1` (2) second derivative of *f* is maximum at :math:`\pm 1` (3) Effective gain is close to 1
    """
    return 1.7159*tf.nn.tanh((2.0/3.0) * tensor_in)

ACTIVATION = {'sigmoid': tf.nn.sigmoid,
              'tanh': tf.nn.tanh,
              'relu': tf.nn.relu,
              'relu6': tf.nn.relu6,
              'softplus': tf.nn.softplus,
              'tanhlecun': tanhlecun}
# ===============================================================================
# ===================NODES=======================================================
# ===============================================================================
def ident(tensor_in, name='ident'):
    """
    Identity function for grouping tensors in graph, during config parsing.

    :param tensor_in: A Tensor_ or list of tensors
    :return: tensor_in
    """
    return tensor_in

def weights(distribution, shape, datatype=tf.float32, initrange=1e-5,
            seed=None, l2=0.0, name='weights'):
    """
    Wrapper parameterizing common constructions of tf.Variables.

    :param distribution: A string identifying distribution 'tnorm' for truncated normal, 'rnorm' for random normal, 'constant' for constant, 'uniform' for uniform.
    :param shape: Shape of weight tensor.
    :param datatype: Datatype for weights
    :param initrange: Scales standard normal and trunctated normal, value of constant dist., and range of uniform dist. [-initrange, initrange].
    :param seed: For reproducible results.
    :param l2: Floating point number determining degree of of l2 regularization for these weights in gradient descent update.
    :param name: For variable scope.
    :return: A tf.Variable.
    """

    with tf.variable_scope(name):
        if distribution == 'norm':
            wghts = tf.Variable(initrange*tf.random_normal(shape, 0, 1, datatype, seed))
        elif distribution == 'tnorm':
            wghts = tf.Variable(initrange*tf.truncated_normal(shape, 0, 1, datatype, seed))
        elif distribution == 'uniform':
            wghts = tf.Variable(tf.random_uniform(shape, -initrange, initrange, datatype, seed))
        elif distribution == 'constant':
            wghts = tf.Variable(tf.constant(initrange, dtype=None, shape=None))
        else:
            raise ValueError("Function weights takes values 'norm', 'tnorm', 'uniform', 'constant', "
                             "for argument distribution. You passed %s" % distribution)
    tf.add_to_collection(name+'_weights', wghts)
    if l2 != 0.0:
        tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(wghts), l2, name=name + 'weight_loss'))
    return wghts

def cosine(operands, name='cosine'):
    """
    Takes the cosine of vectors in corresponding rows of the two matrix tensors_ in operands.

    :param operands:  A list of two tensors to take cosine of.
    :param name: An optional name for unique variable scope.
    :return: A tensor with dimensions (operands[0].shape[0], 1)
    :raises: ValueError when operands do not have matching shapes.
    """
    shape1 = operands[0].get_shape().as_list()
    shape2 = operands[1].get_shape().as_list()
    if not shape1 == shape2:
        raise ValueError("Cosine expects matching shapes for operands. Found operands[0] shape = %s, "
                         "operands[1] shape = %s" % (shape1, shape2))
    else:
        with tf.variable_scope(name):
            xlen = tf.sqrt(tf.reduce_sum(tf.mul(operands[0], operands[0]), 1, keep_dims=True))
            ylen = tf.sqrt(tf.reduce_sum(tf.mul(operands[1], operands[1]), 1, keep_dims=True))
            norm = tf.mul(xlen, ylen)
            tf.add_to_collection(name, xlen)
            tf.add_to_collection(name, ylen)
            tf.add_to_collection(name, norm)
            tensor_out = tf.div(x_dot_y(operands), norm, name=name)
            tf.add_to_collection(name, tensor_out)
            return tensor_out

def x_dot_y(operands, name='x_dot_y'):
    """
    Takes the inner product for rows of operands[1], and operands[2],
    and adds optional bias, operands[3], operands[4].
    If either operands[1] or operands[2] or both is a list of tensors
    then a list of the pairwise dot products (with bias when len(operands) > 2)
    of the lists is returned.

    :param operands: A list of 2, 3, or 4 tensors_ (the first two tensors may be replaced by lists of tensors).
    :param name: An optional identifier for unique variable_scope_.
    :return: A tensor or list of tensors with dimension (operands[1].shape[0], 1).
    :raises: Value error when operands is not a list of at least two tensors.
    """
    if type(operands) is not list or len(operands) < 2:
        raise ValueError("x_dot_y needs a list of 2-4 tensors.")
    outproducts = []
    with tf.variable_scope(name):
        if type(operands[0]) is not list:
            operands[0] = [operands[0]]
        if type(operands[1]) is not list:
            operands[1] = [operands[1]]
        for i in range(len(operands[0])):
            for j in range(len(operands[1])):
                with tf.name_scope('right%d' % i + 'left%d' % j):
                    dot = tf.reduce_sum(tf.mul(operands[0][i], operands[1][j]), 1, keep_dims=True)
                    tf.add_to_collection(name, dot)
                    if len(operands) > 2:
                        dot = dot + operands[2]
                    tf.add_to_collection(name, dot)
                    if len(operands) > 3:
                        dot = dot + operands[3]
                    tf.add_to_collection(name, dot)
                    outproducts.append(dot)
        if len(outproducts) == 1:
            return outproducts[0]
        else:
            return outproducts

def lookup(dataname=None,  data=None,  indices=None, distribution='uniform',
           initrange=None, l2=0.0, shape=None, makeplace=True, name='lookup'):
    """
    A wrapper for `tensorflow's`_ `embedding_lookup`_ which infers the shape of the
    weight matrix and placeholder value from the parameter *data*.

    :param dataname: Used exclusively by config.py
    :param data: A :any:`HotIndex` object
    :param indices: A `Placeholder`_. If indices is none the dimensions will be inferred from *data*
    :param distribution: Distribution for lookup weight initialization
    :param initrange: Initrange for weight distribution.
    :param l2: Floating point number determining degree of of l2 regularization for these weights in gradient descent update.
    :param shape: The dimensions of the output tensor_, typically [None, output-size]
    :param makeplace: A boolean to tell whether or not a placeholder has been created for this data
    :param name: A name for unique variable scope.
    :return: tf.nn.embedding_lookup(wghts, indices), wghts, indices
    """

    if type(data) is loader.HotIndex:
        if makeplace:
            indices = tf.placeholder(tf.int32, [None], name=dataname)
        initrange *= 1.0/numpy.sqrt(float(shape[1]))
        wghts = weights(distribution, [data.dim, shape[1]], initrange=initrange, l2=l2, name=name+'_wghts')
        tf.add_to_collection(name+'_weights', wghts)
        tensor_out = tf.nn.embedding_lookup(wghts, indices, name=name), wghts, indices
        tf.add_to_collection(name, tensor_out)
        return tensor_out

    else:
        raise TypeError("Type of data for lookup indices must be antk.core.loader.HotIndex")

def embedding(tensors, name='embedding'):
    """
    A wrapper for `tensorflow's`_ `embedding_lookup`_

    :param tensors: A list of two tensors_ , matrix, indices
    :param name: Unique name for variable scope
    :return: A matrix tensor_ where the i-th row = matrix[indices[i]]
    """
    matrix = tensors[0]
    indices = tensors[1]
    tensor_out = tf.nn.embedding_lookup(matrix, indices, name=name)
    tf.add_to_collection(name, tensor_out)
    return tensor_out

def placeholder(datatype, shape=None, data=None, name='placeholder'):
    """
    Wrapper to create tensorflow_ Placeholder_ which infers dimensions given data.

    :param datatype: Tensorflow datatype to initiliaze a Placeholder.
    :param shape: Dimensions of Placeholder
    :param data: Data to infer dimensions of Placeholder from.
    :param name: Unique name for variable scope.
    :return: A Tensorflow_ Placeholder.
    """
    if data is None and shape is None:
        raise MissingShapeError('Shape or data to infer the shape from must be provided')
    if data is not None and shape is None:
        if type(data) is loader.HotIndex:
            shape = [None]
        else:
            shapespec = list(data.shape)
            shape = [None]
            shape.extend(shapespec[1:len(shapespec)])
    return tf.placeholder(datatype, shape, name)


def mult_log_reg(tensor_in, numclasses=None, data=None, name='log_reg'):
    """
    Performs mulitnomial logistic regression forward pass. Weights and bias initialized to zeros.

    :param tensor_in: A tensor_ or placeholder_
    :param numclasses: For classificatio
    :param data: For shape inference.
    :param name: For `variable_scope`_
    :return: A tensor shape=(tensor_in.shape[0], numclasses)
    """
    if data is not None:
        if type(data) is loader.HotIndex:
            numclasses = data.dim
        elif loader.is_one_hot(data):
            numclasses = data.shape[1]
        else:
            raise MissingShapeError('Can not infer shape from data: %s' % data)
    elif numclasses is None:
        raise MissingShapeError('Can not infer shape. Need numclasses or data argument.')
    with tf.variable_scope(name):
        inshape = tensor_in.get_shape().as_list()
        W = tf.Variable(tf.zeros([inshape[1], numclasses]))
        b = tf.Variable(tf.zeros([numclasses]))
    tf.add_to_collection(name+'_weights', W)
    tf.add_to_collection(name+'_bias', b)
    tensor_out = tf.nn.softmax(tf.matmul(tensor_in, W) + b)
    tf.add_to_collection(name, tensor_out)
    return tensor_out

def concat(tensors, output_dim, name='concat'):
    """
    Matrix multiplies each tensor_ in *tensors* by its own weight matrix and adds together the results.

    :param tensors: A list of tensors.
    :param output_dim: Dimension of output
    :param name: An optional identifier for unique variable_scope_.
    :return: A tensor with shape [None, output_dim]
    """
    with tf.variable_scope(name):
        for i, tensor in enumerate(tensors):
            with tf.variable_scope('inTensor%d' % i):
                tensor_in = linear(tensor, output_dim, True, name=name)
                tf.add_to_collection(name, tensor_in)
                if i == 0:
                    combo = tensor_in
                    tf.add_to_collection(name, combo)
                else:
                    combo = combo + tensor_in
                    tf.add_to_collection(name, combo)
        return combo

def dnn(tensor_in, hidden_units, activation='tanh', distribution='tnorm',
        initrange=1.0, l2=0.0, bn=False, keep_prob=None, name='dnn'):
    """
    Creates fully connected deep neural network subgraph. Adapted From skflow_ `dnn_ops.py`_
        `Neural Networks and Deep Learning`_

        `Using Neural Nets to Recognize Handwritten Digits`_

    :param tensor_in: tensor_ or placeholder_ for input features.
    :param hidden_units: list of counts of hidden units in each layer.
    :param activation: activation function between layers. Can be None.
    :param distribution: Distribution for lookup weight initialization
    :param initrange: Initrange for weight distribution.
    :param l2: Floating point number determining degree of of l2 regularization for these weights in gradient descent update.
    :param bn: Whether or not to use batch normalization
    :param keep_prob: if not None, will add a dropout layer with given
                    probability.
    :param name: A name for unique variable_scope_.
    :return: A tensor_ which would be a deep neural network.
    """

    activation = ACTIVATION[activation]
    with tf.variable_scope(name):
        for i, n_units in enumerate(hidden_units):
            with tf.variable_scope('layer%d' % i):
                if activation == 'relu':
                    irange= numpy.sqrt(2.0/float(tensor_in.get_shape().as_list()[1]))
                else:
                    irange = initrange*(1.0/numpy.sqrt(float(tensor_in.get_shape().as_list()[1])))
                tensor_in = linear(tensor_in, n_units, bias=True,
                                   distribution=distribution, initrange=irange, l2=l2, name=name)
                tf.add_to_collection(name + '_preactivation', tensor_in)
                tensor_in = activation(tensor_in)
                tf.add_to_collection(ACTIVATION_LAYERS, tensor_in)
                tf.add_to_collection(name + '_activation', tensor_in)
                if bn:
                    tensor_in = batch_normalize(tensor_in)
                    tf.add_to_collection(NORMALIZED_ACTIVATIONS, tensor_in)
                    tf.add_to_collection(name + '_bn', tensor_in)
                if keep_prob:
                    tensor_in = dropout(tensor_in, keep_prob)
                    tf.add_to_collection(name + '_dropouts', tensor_in)
        return tensor_in

def convolutional_net(in_progress=None):
    """
    See: `Tensorflow Deep MNIST for Experts`_ ,
    `Tensorflow Convolutional Neural Networks`_ ,
    `ImageNet Classification with Deep Convolutional Neural Networks`_ ,
    `skflow/examples/text_classification_character_cnn.py`_ ,
    `skflow/examples/text_classification_cnn.py`_ ,
    `Character-level Convolutional Networks for Text Classification`_

    :param in_progress:
    :return:
    """

def residual_dnn(tensor_in, hidden_units,
                 activation='tanh', distribution='tnorm', initrange=1.0, l2=0.0,
                 bn=False,
                 keep_prob=None, skiplayers=3,
                 name='residual_dnn'):
    """
    Creates residual neural network with shortcut connections.
        `Deep Residual Learning for Image Recognition`_

    :param tensor_in: tensor_ or placeholder_ for input features.
    :param hidden_units: list of counts of hidden units in each layer.
    :param activation: activation function between layers. Can be None.
    :param distribution: Distribution for lookup weight initialization
    :param initrange: Initrange for weight distribution.
    :param l2: Floating point number determining degree of of l2 regularization for these weights in gradient descent update.
    :param bn: Whether or not to use batch normalization
    :param keep_prob: if not None, will add a dropout layer with given
                    probability.
    :param skiplayers: The number of layers to skip for the shortcut connection.
    :param name: A name for unique variable scope
    :return: A tensor_ which would be a residual deep neural network.
    """
    if len(hidden_units) % skiplayers != 0:
        raise ValueError('The number of layers must be a multiple of skiplayers')
    if type(activation) is str:
        activation = ACTIVATION[activation]
    with tf.variable_scope(name):
        for k in range(len(hidden_units)//skiplayers):
            shortcut = tensor_in
            start, end = k*skiplayers, k*skiplayers + skiplayers
            for i, n_units in enumerate(hidden_units[start:end]):
                with tf.variable_scope('layer%d' % i*(k+1)):
                    if activation == 'relu':
                        irange= numpy.sqrt(2.0/float(tensor_in.get_shape().as_list()[1]))
                    else:
                        irange = initrange*(1.0/numpy.sqrt(float(tensor_in.get_shape().as_list()[1])))
                    tensor_in = linear(tensor_in, n_units, bias=True, distribution=distribution,
                                       initrange=initrange*irange,
                                       l2=l2,
                                       name=name)
                    if activation:
                        tensor_in = activation(tensor_in)
                        tf.add_to_collection(ACTIVATION_LAYERS, tensor_in)
                        tf.add_to_collection(name + '_activation', tensor_in)
                    if bn:
                        tensor_in = batch_normalize(tensor_in)
                        tf.add_to_collection(NORMALIZED_ACTIVATIONS, tensor_in)
                        tf.add_to_collection(name + '_bn', tensor_in)
                    if keep_prob:
                        tensor_in = dropout(tensor_in, keep_prob)
                        tf.add_to_collection(name + '_dropouts', tensor_in)
            shp1, shp2 = shortcut.get_shape().as_list(), tensor_in.get_shape().as_list()
            if shp1[1] != shp2[1]:
                with tf.variable_scope('skip_connect%d' % k):
                    if activation == 'relu':
                        irange= numpy.sqrt(2.0/float(shp1[1]))
                    else:
                        irange = initrange*(1.0/numpy.sqrt(float(shp1[1])))
                    shortcut = linear(shortcut, shp2[1], bias=True,
                                      initrange=irange,
                                      distribution=distribution, l2=l2,
                                      name=name)
                    tf.add_to_collection(name + '_skiptransform', shortcut)
            tensor_in = tensor_in + shortcut
            tf.add_to_collection(name + '_skipconnection', tensor_in)
        return tensor_in

def highway_dnn(tensor_in, hidden_units, activation='tanh', distribution='tnorm', initrange=1.0,
                l2=0.0, bn=False,
                keep_prob=None, bias_start=-1, name='highway_dnn'):
    """
    A highway deep neural network.
        `Training Very Deep Networks`_

    :param tensor_in: A 2d matrix tensor_.
    :param hidden_units:  list of counts of hidden units in each layer.
    :param activation: Non-linearity to perform. Can be ident for no non-linearity.
    :param distribution: Distribution for lookup weight initialization
    :param initrange: Initrange for weight distribution.
    :param l2: Floating point number determining degree of of l2 regularization for these weights in gradient descent update.
    :param bn: Whether or not to use batch normalization
    :param keep_prob: Dropout rate.
    :param bias_start: initialization of transform bias weights
    :param name: A name for unique variable_scope.
    :return: A tensor_ which would be a highway deep neural network.
    """
    if type(activation) is str:
        activation = ACTIVATION[activation]
    with tf.variable_scope(name):
        for i, n_units in enumerate(hidden_units):
            with tf.variable_scope('layer%d' % i):
                with tf.variable_scope('hidden'):
                    if activation == 'relu':
                        irange= numpy.sqrt(2.0/float(tensor_in.get_shape().as_list()[1]))
                    else:
                        irange = initrange*(1.0/numpy.sqrt(float(tensor_in.get_shape().as_list()[1])))
                    hidden = linear(tensor_in, n_units, bias=True,
                                               distribution=distribution, initrange=irange, l2=l2,
                                               name=name)
                    tf.add_to_collection(name + '_preactivation', hidden)
                    hidden = activation(hidden)
                    tf.add_to_collection(ACTIVATION_LAYERS, hidden)
                    tf.add_to_collection(name + '_activation', hidden)
                with tf.variable_scope('transform'):
                    if activation == 'relu':
                        irange= numpy.sqrt(2.0/float(tensor_in.get_shape().as_list()[1]))
                    else:
                        irange = initrange*(1.0/numpy.sqrt(float(tensor_in.get_shape().as_list()[1])))
                    transform = tf.sigmoid(linear(tensor_in, n_units,
                                                  bias_start=bias_start, bias=True,
                                                  initrange=irange, l2=l2, distribution=distribution,
                                                  name=name))
                    tf.add_to_collection(ACTIVATION_LAYERS, transform)
                    tf.add_to_collection(name + '_transform', transform)
                tensor_in = hidden * transform + tensor_in * (1 - transform)
                tf.add_to_collection(name, tensor_in)
                if bn:
                    tensor_in = batch_normalize(tensor_in)
                    tf.add_to_collection(NORMALIZED_ACTIVATIONS, tensor_in)
                    tf.add_to_collection(name + '_bn', tensor_in)
                if keep_prob:
                    tensor_in = dropout(tensor_in, keep_prob, name=name)
                    tf.add_to_collection(name + '_dropouts', tensor_in)
        return tensor_in

def dropout(tensor_in, prob, name=None):
    """
    Adds dropout node. Adapted from skflow `dropout_ops.py`_ .
        `Dropout A Simple Way to Prevent Neural Networks from Overfitting`_

    :param tensor_in: Input tensor_.
    :param prob: The percent of weights to keep.
    :param name: A name for the tensor.
    :return: Tensor_ of the same shape of *tensor_in*.
    """
    with tf.variable_scope("name"):
        if isinstance(prob, float):
            keep_prob = tf.placeholder(tf.float32)
            tf.add_to_collection('dropout_prob', (keep_prob, prob))
        return tf.nn.dropout(tensor_in, keep_prob)

def linear(tensor_in, output_size, bias, bias_start=0.0,
           distribution='tnorm', initrange=1.0, l2=0.0,
           name="Linear"):
    """
    Linear map: :math:`\sum_i(args[i] * W_i)`, where :math:`W_i` is a variable.

    :param args: a 2D Tensor_
    :param output_size: int, second dimension of W[i].
    :param bias: boolean, whether to add a bias term or not.
    :param bias_start: starting value to initialize the bias; 0 by default.
    :param distribution: Distribution for lookup weight initialization
    :param initrange: Initrange for weight distribution.
    :param l2: Floating point number determining degree of of l2 regularization for these weights in gradient descent update.
    :param name: VariableScope for the created subgraph; defaults to "Linear".
    :return: A 2D Tensor with shape [batch x output_size] equal to
        :math:`\sum_i(args[i] * W_i)`, where :math:`W_i` are newly created matrices.
    :raises: ValueError: if some of the arguments has unspecified or wrong shape.
    """
    shape = tensor_in.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    with tf.variable_scope(name):
        matrix = weights(distribution, [shape[1], output_size], initrange=initrange, l2=l2, name='Matrix')
        tf.add_to_collection(name+'_weights', matrix)
        tensor_out = tf.matmul(tf.cast(tensor_in, tf.float32), matrix)
        if not bias:
            return tensor_out
        bias_term = weights('constant', [output_size], initrange=bias_start)
        tf.add_to_collection(name+'_bias', bias_term)
    return tensor_out + bias_term


def batch_normalize(tensor_in, epsilon=1e-5, name="batch_norm"):
    """
    Batch Normalization: Adapted from tensorflow `nn.py`_ and skflow `batch_norm_ops.py`_ .
        `Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift`_

    :param tensor_in: input Tensor_
    :param epsilon: A float number to avoid being divided by 0.
    :param name: For variable_scope_
    :return: Tensor with variance bounded by a unit and mean of zero according to the batch.
    """
    shape = tensor_in.get_shape().as_list()

    with tf.variable_scope(name):
        gamma = tf.get_variable("gamma", [shape[1]],
                                initializer=tf.constant_initializer(0))
        tf.add_to_collection(name, gamma)
        beta = tf.get_variable("beta", [shape[1]],
                               initializer=tf.constant_initializer(1.0))
        mean, variance = tf.nn.moments(tensor_in, [0])
        tf.add_to_collection(name, mean)
        tf.add_to_collection(name, variance)
        tf.add_to_collection(name, beta)
        inv = tf.rsqrt(epsilon + variance)
        return beta * (tensor_in - mean) * inv + gamma

def nmode_tensor_tomatrix(tensor, mode, name='nmode_matricize'):
    '''
    Nmode tensor unfolding (for order three tensor) from Kolda and Bader `Tensor Decompositions and Applications`_

    :param tensor: Order 3 tensor to unfold
    :param mode: Mode to unfold (0,1,2, columns, rows, or fibers)
    :param name: For variable scoping
    :return: A matrix (order 2 tensor) with shape dim(mode) X :math:`\Pi_{othermodes}` dim(othermodes)    '''

    tensorshape = tensor.get_shape().as_list()
    if mode == 0:
        tensor = tf.transpose(tensor, perm=[0, 2, 1])
        matricized_shape = [tensorshape[mode], 1, -1]
    if mode == 1:
        tensor = tf.transpose(tensor, perm=[1, 2, 0])
        matricized_shape = [tensorshape[mode], 1, -1]
    if mode == 2:
        tensor = tf.transpose(tensor, perm=[2, 1, 0])
        matricized_shape = [tensorshape[mode], -1, 1]
    tensor = tf.squeeze(tf.reshape(tensor, matricized_shape))
    return tensor

def nmode_tensor_multiply(tensors, mode, leave_flattened=False,
                          keep_dims=False, name='nmode_multiply'):
    '''
    Nth mode tensor multiplication (for order three tensor) from Kolda and Bader `Tensor Decompositions and Applications`_
    Works for vectors (matrix with a 1 dimension or matrices)

    :param tensors: A list of tensors the first is an order three tensor the second and order 2
    :param mode: The mode to perform multiplication against.
    :param leave_flattened: Whether or not to reshape tensor back to order 3
    :param keep_dims: Whether or not to remove 1 dimensions
    :param name: For variable scope
    :return: Either an order 3 or order 2 tensor
    '''



    tensor = tensors[0]
    matrix = tensors[1]
    tensorshape = tensor.get_shape().as_list()
    matrixshape = matrix.get_shape().as_list()
    if tensorshape[mode] != matrixshape[1]:
        raise ValueError("Number of columns of matrix must equal dimension of tensor mode")
    else:
        flattened_product = tf.matmul(matrix, nmode_tensor_tomatrix(tensor, mode))
        if not leave_flattened:
            if mode == 0:
                product = tf.transpose(tf.reshape(flattened_product, [-1,tensorshape[2],tensorshape[1]]), [0,2,1])
            elif mode == 1:
                product = tf.transpose(tf.reshape(flattened_product, [-1,tensorshape[2],tensorshape[0]]), [0,2,1])
            elif mode == 2:
                product = tf.transpose(tf.reshape(flattened_product, [-1,tensorshape[1],tensorshape[0]]), [2,1,0])
        if not keep_dims:
            product = tf.squeeze(product)
        return product


def binary_tensor_combine(tensors, output_dim=10, initrange=1e-5, name='binary_tensor_combine'):
    '''
    For performing tensor multiplications with batches of data points against an order 3
    weight tensor.

    :param tensors: A list of two matrices each with first dim batch-size
    :param output_dim: The dimension of the third mode of the weight tensor
    :param initrange: For initializing weight tensor
    :param name: For variable scope
    :return: A matrix with shape batch_size X output_dim
    '''

    mat1 = tensors[0]
    mat2 = tensors[1]
    mat1shape = mat1.get_shape().as_list()
    mat2shape = mat2.get_shape().as_list()
    if mat1shape[0] != mat2shape[0]:
        raise ValueError("Number of rows must match for matrices being combined.")
    # t = weights('tnorm', mat2.dtype, [tensors[0].get_shape().as_list()[1], tensors[1].get_shape().as_list()[1], output_dim])
    t = weights('tnorm', [mat1.get_shape().as_list()[1],
                                    mat2.get_shape().as_list()[1],
                                    output_dim],  datatype=mat1.dtype)
    tf.add_to_collection(name+'_weights', t)
    prod = nmode_tensor_multiply([t, mat1], mode=0, keep_dims=True)
    mat2 = tf.expand_dims(mat2, 1)
    return tf.squeeze(tf.batch_matmul(mat2, prod), [1])

def ternary_tensor_combine(tensors, initrange=1e-5, name='ternary_tensor_combine'):
    '''
    For performing tensor multiplications with batches of data points against an order 3
    weight tensor.

    :param tensors:
    :param output_dim:
    :param initrange:
    :param name:
    :return:
    '''
    combine_pair = [tensors[0], tensors[1]]
    combined = binary_tensor_combine2(combine_pair, output_dim = tensors[2].get_shape().as_list()[1])
    return x_dot_y([combined,tensors[2]])

def khatri_rao(tensors, name='khatrirao'):
    '''
    From `David Palzer`_

    :param tensors:
    :param name:
    :return:
    '''
    h1 = tensors[0]
    h2 = tensors[1]
    L1 = h1.get_shape().as_list()[1]
    L2 = h2.get_shape().as_list()[1]
    L = L1*L2
    h2Tiled = tf.tile(h2,[1,L1]) # how to tile h2 # L1 and L2 are the number of cols in H1 and H2 respectively
    h1Tiled = tf.reshape(tf.transpose(tf.tile(tf.reshape(h1, [1, -1]), [L2, 1])), [-1, L]) # how to tile h1
    return tf.mul(h1Tiled,h2Tiled)

def binary_tensor_combine2(tensors, output_dim=10, initrange=1e-5, name='binary_tensor_combine2'):
    with tf.variable_scope(name):
        x = khatri_rao(tensors)
        w = weights('tnorm', x.dtype, [tensors[0].get_shape().as_list()[1] * tensors[1].get_shape().as_list()[1], output_dim])
        print(x.get_shape())
        print(w.get_shape())
        return tf.matmul(x, w)

# ==================================================================================
# =============EVALUATION METRICS / LOSS FUNCTIONS==================================
# ==================================================================================

def se(predictions, targets):
    '''
    Squared Error.
    '''
    return tf.reduce_sum(tf.square(predictions - targets))

def mse(predictions, targets):
    '''
    Mean Squared Error.
    '''
    return tf.reduce_mean(tf.square(predictions - targets))

def rmse(predictions, targets):
    '''
    Root Mean Squared Error
    '''
    return tf.sqrt(tf.reduce_mean(tf.square(predictions - targets)))

def mae(predictions, targets):
    '''Mean Absolute Error'''
    return tf.reduce_mean(tf.abs(predictions - targets))

def other_cross_entropy(predictions, targets):
    '''Logistic Loss'''
    return -1*tf.reduce_sum(targets * tf.log(predictions) + (1.0 - targets) * tf.log(1.0 - predictions))

def cross_entropy(predictions, targets):
    return -tf.reduce_sum(targets*tf.log(predictions))

def perplexity(predictions, targets):
    return tf.exp(cross_entropy(predictions, targets))

def detection(predictions, threshold):
    return tf.cast(tf.greater_equal(predictions, threshold), tf.float32)

def recall(predictions, targets, threshold=0.5, detects=None):
    '''
    Percentage of actual classes predicted

    :param targets: A one hot encoding of class labels (num_points X numclasses)
    :param predictions: A real valued matrix with indices ranging between zero and 1 (num_points X numclasses)
    :param threshold: The detection threshold (between zero and 1)
    :param detects: In case detection is precomputed for efficiency when evaluating both precision and recall
    :return: A scalar value
    '''
    if not detects:
        detects = detection(predictions, threshold)
    return tf.div(tf.reduce_sum(tf.mul(detects, targets)), tf.reduce_sum(targets))

def precision(predictions, targets, threshold=0.5, detects=None):
    '''
    Percentage of classes detected which are correct.

    :param targets: A one hot encoding of class labels (num_points X numclasses)
    :param predictions: A real valued matrix with indices ranging between zero and 1 (num_points X numclasses)
    :param threshold: The detection threshold (between zero and 1)
    :param detects: In case detection is precomputed for efficiency when evaluating both precision and recall
    :return: A scalar value
    '''
    if not detects:
        detects = detection(predictions, threshold)
    return tf.reduce_sum(tf.mul(targets, detects)) / (tf.reduce_sum(detects) + 1e-8)

def fscore(predictions=None, targets=None, threshold=0.5, precisions=None, recalls=None):
    if not precisions and not recalls:
        detects = detection(predictions, threshold)
        recalls = recall(targets, threshold=threshold, detects=detects)
        precisions = precision(targets, threshold=threshold, detects=detects)
    return 2*(tf.mul(precisions, recalls) / (precisions + recalls + 1e-8))

def accuracy(predictions, targets):
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(targets, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



