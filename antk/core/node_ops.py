from __future__ import division
import tensorflow as tf
import numpy
import scipy.sparse as sps
from antk.core import loader
import numbers

from antk.lib.decorate import pholder, variable, node_op, neural_net, act, relu, loss_function


ACTIVATION_LAYERS = 'activation_layers'
NORMALIZED_ACTIVATIONS = 'normalized_activations'

class MissingShapeError(Exception):
    '''Raised when :any:`placeholder` can not infer shape.'''
    pass

def fan_scale(initrange, activation, tensor_in):
    if activation == relu:
        initrange *= numpy.sqrt(2.0/float(tensor_in.get_shape().as_list()[1]))
    else:
        initrange *= (1.0/numpy.sqrt(float(tensor_in.get_shape().as_list()[1])))




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

@pholder
def placeholder(dtype, shape=None, data=None, name='placeholder'):
    """
    Wrapper to create tensorflow_ Placeholder_ which infers dimensions given data.

    :param dtype: Tensorflow dtype to initiliaze a Placeholder.
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
    return tf.placeholder(dtype, shape, name)

@variable
def weights(distribution, shape, dtype=tf.float32, initrange=1e-5,
            seed=None, l2=0.0, name='weights'):
    """
    Wrapper parameterizing common constructions of tf.Variables.

    :param distribution: A string identifying distribution 'tnorm' for truncated normal, 'rnorm' for random normal, 'constant' for constant, 'uniform' for uniform.
    :param shape: Shape of weight tensor.
    :param dtype: dtype for weights
    :param initrange: Scales standard normal and trunctated normal, value of constant dist., and range of uniform dist. [-initrange, initrange].
    :param seed: For reproducible results.
    :param l2: Floating point number determining degree of of l2 regularization for these weights in gradient descent update.
    :param name: For variable scope.
    :return: A tf.Variable.
    """

    if distribution == 'norm':
        wghts = tf.Variable(initrange*tf.random_normal(shape, 0, 1, dtype, seed))
    elif distribution == 'tnorm':
        wghts = tf.Variable(initrange*tf.truncated_normal(shape, 0, 1, dtype, seed))
    elif distribution == 'uniform':
        wghts = tf.Variable(tf.random_uniform(shape, -initrange, initrange, dtype, seed))
    elif distribution == 'constant':
        wghts = tf.Variable(tf.constant(initrange, dtype=dtype, shape=shape))
    else:
        raise ValueError("Argument 'distribution takes values 'norm', 'tnorm', 'uniform', 'constant', "
                          "Received %s" % distribution)
    if l2 != 0.0:
        tf.add_to_collection('losses', tf.mul(tf.nn.l2_loss(wghts), l2, name=name + 'weight_loss'))
    return wghts

@node_op
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
        xlen = node_op(tf.sqrt)(tf.reduce_sum(tf.mul(operands[0], operands[0]), 1, keep_dims=True), name='cosine')
        ylen = node_op(tf.sqrt)(tf.reduce_sum(tf.mul(operands[1], operands[1]), 1, keep_dims=True))
        norm = node_op(tf.mul)(xlen, ylen)
        return tf.div(x_dot_y(operands), norm, name=name)

@node_op
def x_dot_y(operands, name='x_dot_y'):
    """
    Takes the inner product for rows of operands[1], and operands[2],
    and adds optional bias, operands[3], operands[4].
    If either operands[1] or operands[2] or both is a list of tensors
    then a list of the pairwise dot products (with bias when len(operands) > 2)
    of the lists is returned.

    :param operands: A list of 2, 3, or 4 tensors_ (the first two tensors may be replaced by lists of tensors
                                                    in which case the return value will a list of the dot products
                                                    for all members of the cross product of the two lists.).
    :param name: An optional identifier for unique variable_scope_.
    :return: A tensor or list of tensors with dimension (operands[1].shape[0], 1).
    :raises: Value error when operands is not a list of at least two tensors.
    """
    if type(operands) is not list or len(operands) < 2:
        raise ValueError("x_dot_y needs a list of 2-4 tensors.")
    outproducts = []

    if type(operands[0]) is not list:
        operands[0] = [operands[0]]
    if type(operands[1]) is not list:
        operands[1] = [operands[1]]
    for i in range(len(operands[0])):
        for j in range(len(operands[1])):
            with tf.name_scope('right%d' % i + 'left%d' % j):
                dot = node_op(tf.reduce_sum)(tf.mul(operands[0][i], operands[1][j]), 1, keep_dims=True, name=name)
                if len(operands) > 2:
                    dot = dot + operands[2]
                if len(operands) > 3:
                    dot = dot + operands[3]
                outproducts.append(dot)
    if len(outproducts) == 1:
        return outproducts[0]
    else:
        return outproducts

@node_op
def lookup(dataname=None,  data=None,  indices=None, distribution='uniform',
           initrange=0.1, l2=0.0, shape=None, makeplace=True, name='lookup'):
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
    :param makeplace: A boolean to tell whether or not a placeholder has been created for this data (Used by config.py)
    :param name: A name for unique variable scope.
    :return: tf.nn.embedding_lookup(wghts, indices), wghts, indices
    """

    if type(data) is loader.HotIndex:
        if makeplace:
            indices = tf.placeholder(tf.int32, [None], name=dataname)
        wghts = weights(distribution, [data.dim, shape[1]], initrange=initrange, l2=l2, name=name+'_weights')
        return tf.nn.embedding_lookup(wghts, indices, name=name), wghts, indices
    else:
        raise TypeError("Type of data for lookup indices must be antk.core.loader.HotIndex")


@node_op
def embedding(tensors, name='embedding'):
    """
    A wrapper for `tensorflow's`_ `embedding_lookup`_

    :param tensors: A list of two tensors_ , matrix, indices
    :param name: Unique name for variable scope
    :return: A matrix tensor_ where the i-th row = matrix[indices[i]]
    """
    matrix = tensors[0]
    indices = tensors[1]
    return tf.nn.embedding_lookup(matrix, indices, name=name)

@node_op
def mult_log_reg(tensor_in, numclasses=None, data=None, dtype=tf.float32,
                 initrange=1e-10, seed=None, l2=0.0, name='log_reg'):
    """
    Performs mulitnomial logistic regression forward pass. Weights and bias initialized to zeros.

    :param tensor_in: A tensor_ or placeholder_
    :param numclasses: For classificatio
    :param data: For shape inference.
    :param dtype: For :any:`weights` initialization.
    :param initrange: For :any:`weights` initialization.
    :param seed: For :any:`weights` initialization.
    :param l2: For :any:`weights` initialization.
    :param name: For `variable_scope`_
    :return:  A tensor shape=(tensor_in.shape[0], numclasses)
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
    inshape = tensor_in.get_shape().as_list()
    W = weights('uniform', [inshape[1], numclasses], dtype=dtype,
                initrange=initrange, seed=seed, l2=l2, name=name + '_weights')
    b = weights('uniform', [numclasses], dtype=dtype,
                initrange=initrange, seed=seed, l2=l2, name=name + '_bias')
    tensor_out = tf.nn.softmax(tf.matmul(tensor_in, W) + b)
    return tensor_out

@node_op
def concat(tensors, output_dim, name='concat'):
    """
    Matrix multiplies each tensor_ in *tensors* by its own weight matrix and adds together the results.

    :param tensors: A list of tensors.
    :param output_dim: Dimension of output
    :param name: An optional identifier for unique variable_scope_.
    :return: A tensor with shape [None, output_dim]
    """
    for i, tensor in enumerate(tensors):
        with tf.variable_scope('inTensor%d' % i):
            tensor_in = linear(tensor, output_dim, True, name=name + '_linear')
            if i == 0:
                combo = tensor_in
            else:
                combo = combo + tensor_in
    return combo

@neural_net
def dnn(tensor_in, hidden_units, activation='tanh', distribution='tnorm',
        initrange=1.0, l2=0.0, bn=False, keep_prob=None, fan_scaling=False, name='dnn'):
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
    for i, n_units in enumerate(hidden_units):
        with tf.variable_scope('layer%d' % i):
            if fan_scaling:
                initrange = fan_scale(initrange, activation, tensor_in)
            tensor_in = linear(tensor_in, n_units, bias=not bn,
                               distribution=distribution, initrange=initrange, l2=l2, name=name)
            if bn:
                tensor_in = batch_normalize(tensor_in, name=name + '_bn')
            tensor_in = activation(tensor_in, name=name + '_activation')
            if keep_prob:
                tensor_in = dropout(tensor_in, keep_prob, name=name + '_dropouts')
    return tensor_in


@neural_net
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

@neural_net
def residual_dnn(tensor_in, hidden_units, activation='tanh', distribution='tnorm',
        initrange=1.0, l2=0.0, bn=False, keep_prob=None, fan_scaling=False,
        skiplayers=3, name='residual_dnn'):
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
    if fan_scaling:
        initrange = fan_scale(initrange, activation, tensor_in)

    for k in range((len(hidden_units))//skiplayers):
        shortcut = tensor_in
        tensor_in = linear(tensor_in, hidden_units[k*skiplayers], bias=not bn, distribution=distribution,
                                   initrange=initrange,
                                   l2=l2,
                                   name=name)
        start, end = k*(skiplayers) + 1, k*(skiplayers) + skiplayers
        for i, n_units in enumerate(hidden_units[start:end]):
            with tf.variable_scope('layer%d' % i*(k+1)):
                if bn:
                    tensor_in = batch_normalize(tensor_in, name=name + '_bn')
                tensor_in = activation(tensor_in, name = name + '_activation')
                if keep_prob:
                    tensor_in = dropout(tensor_in, keep_prob, name=name + '_dropouts')
                tensor_in = linear(tensor_in, n_units, bias=not bn, distribution=distribution,
                                   initrange=initrange,
                                   l2=l2,
                                   name=name)
        shp1, shp2 = shortcut.get_shape().as_list(), tensor_in.get_shape().as_list()
        if shp1[1] != shp2[1]:
            with tf.variable_scope('skip_connect%d' % k):
                shortcut = linear(shortcut, shp2[1], bias=True,
                                  initrange=initrange,
                                  distribution=distribution, l2=l2,
                                  name=name + '_skiptransform')
        tensor_in = tensor_in + shortcut
        tensor_in = activation(tensor_in)
    return tensor_in

@neural_net
def highway_dnn(tensor_in, hidden_units, activation='tanh', distribution='tnorm',
                initrange=1.0, l2=0.0, bn=False, keep_prob=None, fan_scaling=False,
                bias_start=-1, name='highway_dnn'):
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
    if fan_scaling:
        initrange = fan_scale(initrange, activation, tensor_in)
    for i, n_units in enumerate(hidden_units):
        with tf.variable_scope('layer%d' % i):
            with tf.variable_scope('hidden'):
                hidden = linear(tensor_in, n_units, bias=not bn,
                                           distribution=distribution, initrange=initrange, l2=l2,
                                           name=name)
                # if bn:
                #     tensor_in = batch_normalize(tensor_in, name=name + '_bn')
                hidden = activation(hidden, name=name+'_activation')
            with tf.variable_scope('transform'):
                transform = linear(tensor_in, n_units,
                                   bias_start=bias_start, bias=not bn,
                                   initrange=initrange, l2=l2, distribution=distribution,
                                   name=name + '_transform')
                # if bn:
                #     transform = batch_normalize(tensor_in, name=name + '_bn')
            tensor_in = hidden * transform + tensor_in * (1 - transform)
            if bn:
                tensor_in = batch_normalize(tensor_in, name=name + '_bn')
            tf.add_to_collection(name, tensor_in)
            if keep_prob:
                tensor_in = dropout(tensor_in, keep_prob, name=name + '_dropouts')
    return tensor_in

@node_op
def dropout(tensor_in, prob, name='Dropout'):
    """
    Adds dropout node. Adapted from skflow `dropout_ops.py`_ .
        `Dropout A Simple Way to Prevent Neural Networks from Overfitting`_

    :param tensor_in: Input tensor_.
    :param prob: The percent of weights to keep.
    :param name: A name for the tensor.
    :return: Tensor_ of the same shape of *tensor_in*.
    """
    if isinstance(prob, float):
        keep_prob = tf.placeholder(tf.float32)
        tf.add_to_collection('dropout_prob', (keep_prob, prob))
    return tf.nn.dropout(tensor_in, keep_prob)

@node_op
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
    W = weights(distribution, [shape[1], output_size], initrange=initrange, l2=l2, name=name+'_weights')
    tensor_out = tf.matmul(tf.cast(tensor_in, tf.float32), W)
    if not bias:
        return tensor_out
    b = weights('uniform', [output_size], initrange=bias_start, name=name+'_bias')
    return tensor_out + b

@node_op
def batch_normalize(tensor_in, epsilon=1e-5, decay=0.999, name="batch_norm"):
    """
    Batch Normalization:
    `Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift`_

    An exponential moving average of means and variances in calculated to estimate sample mean
    and sample variance for evaluations. For testing pair placeholder is_training
    with [0] in feed_dict. For training pair placeholder is_training
    with [1] in feed_dict. Example:

    Let **train = 1** for training and **train = 0** for evaluation

    .. code-block:: python
        bn_deciders = {decider:[train] for decider in tf.get_collection('bn_deciders')}
        feed_dict.update(bn_deciders)

    :param tensor_in: input Tensor_
    :param epsilon: A float number to avoid being divided by 0.
    :param name: For variable_scope_
    :return: Tensor with variance bounded by a unit and mean of zero according to the batch.
    """

    is_training = tf.placeholder(tf.int32, shape=[None]) # [1] or [0], Using a placeholder to decide which
                                          # statistics to use for normalization allows
                                          # either the running stats or the batch stats to
                                          # be used without rebuilding the graph.
    tf.add_to_collection('bn_deciders', is_training)

    pop_mean = tf.Variable(tf.zeros([tensor_in.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([tensor_in.get_shape()[-1]]), trainable=False)

    # calculate batch mean/var and running mean/var
    batch_mean, batch_variance = node_op(tf.nn.moments)(tensor_in, [0], name=name)

    # The running mean/variance is updated when is_training == 1.
    running_mean = tf.assign(pop_mean,
                             pop_mean * (decay + (1.0 - decay)*(1.0 - tf.to_float(is_training))) +
                             batch_mean * (1.0 - decay) * tf.to_float(is_training))
    running_var = tf.assign(pop_var,
                            pop_var * (decay + (1.0 - decay)*(1.0 - tf.to_float(is_training))) +
                            batch_variance * (1.0 - decay) * tf.to_float(is_training))

    # Choose statistic
    mean = tf.nn.embedding_lookup(tf.pack([running_mean, batch_mean]), is_training)
    variance = tf.nn.embedding_lookup(tf.pack([running_var, batch_variance]), is_training)

    shape = tensor_in.get_shape().as_list()
    gamma = weights('constant', [shape[1]], initrange=0.0, name=name + '_gamma')
    beta = weights('constant', [shape[1]], initrange=1.0, name=name + '_beta')

    # Batch Norm Transform
    inv = node_op(tf.rsqrt)(epsilon + variance, name=name)
    tensor_in = beta * (tensor_in - mean) * inv + gamma

    tf.add_to_collection(NORMALIZED_ACTIVATIONS, tensor_in)

    return tensor_in


@node_op
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

@node_op
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

@node_op
def binary_tensor_combine(tensors, output_dim=10, initrange=1e-5, l2=0.0,
                          distribution='tnorm', name='binary_tensor_combine'):
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
    t = weights(distribution, [mat1.get_shape().as_list()[1],
                                    mat2.get_shape().as_list()[1],
                                    output_dim],  dtype=mat1.dtype, l2=l2)
    tf.add_to_collection(name+'_weights', t)
    prod = nmode_tensor_multiply([t, mat1], mode=0, keep_dims=True)
    mat2 = tf.expand_dims(mat2, 1)
    return tf.squeeze(tf.batch_matmul(mat2, prod), [1])

@node_op
def ternary_tensor_combine(tensors, initrange=1e-5, distribution='tnorm',
                           l2=0.0, name='ternary_tensor_combine'):
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
    combined = binary_tensor_combine(combine_pair, output_dim=tensors[2].get_shape().as_list()[1], l2=l2)
    return x_dot_y([combined,tensors[2]])

@node_op
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

@node_op
def binary_tensor_combine2(tensors, output_dim=10, initrange=1e-5, name='binary_tensor_combine2'):
    with tf.variable_scope(name):
        x = khatri_rao(tensors)
        w = weights('tnorm',
                    [tensors[0].get_shape().as_list()[1] * tensors[1].get_shape().as_list()[1],
                     output_dim],
                    dtype=x.dtype)
        return tf.matmul(x, w)

# ==================================================================================
# =============EVALUATION METRICS / LOSS FUNCTIONS==================================
# ==================================================================================

@loss_function
def se(predictions, targets, name='squared_error'):
    '''
    Squared Error.
    '''
    return tf.reduce_sum(tf.square(predictions - targets))

@loss_function
def mse(predictions, targets, name='mse'):
    '''
    Mean Squared Error.
    '''
    return tf.reduce_mean(tf.square(predictions - targets))

@loss_function
def rmse(predictions, targets, name='rmse'):
    '''
    Root Mean Squared Error
    '''
    return tf.sqrt(tf.reduce_mean(tf.square(predictions - targets)))

@loss_function
def mae(predictions, targets, name='mae'):
    '''Mean Absolute Error'''
    return tf.reduce_mean(tf.abs(predictions - targets))


@loss_function
def other_cross_entropy(predictions, targets, name='logistic_loss'):
    '''Logistic Loss'''
    return -1*tf.reduce_sum(targets * tf.log(predictions) + (1.0 - targets) * tf.log(1.0 - predictions))

@loss_function
def cross_entropy(predictions, targets, name='cross_entropy'):
    return -tf.reduce_sum(targets*tf.log(predictions + 1e-8))

@loss_function
def perplexity(predictions, targets, name='perplexity'):
    return tf.exp(cross_entropy(predictions, targets))

@loss_function
def detection(predictions, threshold, name='detection'):
    return tf.cast(tf.greater_equal(predictions, threshold), tf.float32)

@loss_function
def recall(predictions, targets, threshold=0.5, detects=None, name='recall'):
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

@loss_function
def precision(predictions, targets, threshold=0.5, detects=None, name='precision'):
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

@loss_function
def fscore(predictions=None, targets=None, threshold=0.5, precisions=None, recalls=None, name='fscore'):
    if not precisions and not recalls:
        detects = detection(predictions, threshold)
        recalls = recall(targets, threshold=threshold, detects=detects)
        precisions = precision(targets, threshold=threshold, detects=detects)
    return 2*(tf.mul(precisions, recalls) / (precisions + recalls + 1e-8))

@loss_function
def accuracy(predictions, targets, name='accuracy'):
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(targets, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
