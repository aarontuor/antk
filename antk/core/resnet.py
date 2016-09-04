from __future__ import division
import tensorflow as tf
import numpy

def fan_scale(initrange, activation, tensor_in):
    if activation == tf.nn.relu:
        initrange *= numpy.sqrt(2.0/float(tensor_in.get_shape().as_list()[1]))
    else:
        initrange *= (1.0/numpy.sqrt(float(tensor_in.get_shape().as_list()[1])))

def dropout(tensor_in, prob, name=None):
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

def batch_normalize(tensor_in, epsilon=1e-5, decay=0.999, name="batch_norm"):
    """
    Batch Normalization:
    `Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift`_

    An exponential moving average of means and variances in calculated to estimate sample mean
    and sample variance for evaluations. For testing pair placeholder is_training
    with 0 in feed_dict. For training pair placeholder is_training
    with 1 in feed_dict. Example:

    Let **train = 1** for training and **train = 0** for evaluation

    .. code-block:: python
        bn_deciders = {decider:train for decider in tf.get_collection('bn_deciders')}
        feed_dict.update(bn_deciders)

    :param tensor_in: input Tensor_
    :param epsilon: A float number to avoid being divided by 0.
    :param name: For variable_scope_
    :return: Tensor with variance bounded by a unit and mean of zero according to the batch.
    """

    is_training = tf.placeholder(tf.int32, shape=[]) # 1 or 0, Using a placeholder to decide which
                                          # statistics to use for normalization allows
                                          # either the running stats or the batch stats to
                                          # be used without rebuilding the graph.
    tf.add_to_collection('bn_deciders', is_training)

    pop_mean = tf.Variable(tf.zeros([tensor_in.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([tensor_in.get_shape()[-1]]), trainable=False)

    # calculate batch mean/var and running mean/var
    batch_mean, batch_variance = tf.nn.moments(tensor_in, [0], name=name)

    # The running mean/variance is updated when is_training == 1.
    running_mean = tf.assign(pop_mean,
                             pop_mean * (decay + (1 - decay)*(1 - is_training)) +
                             batch_mean * (1 - decay) * is_training)
    running_var = tf.assign(pop_var,
                            pop_var * (decay + (1 - decay)*(1 - is_training)) +
                            batch_variance * (1 - decay) * is_training)

    # Choose statistic
    mean = tf.nn.embedding(tf.concat(0, [running_mean, batch_mean]), tf.reshape(is_training, [1]))
    variance = tf.nn.embedding(tf.concat(0, [running_var, batch_variance]), tf.reshape(is_training, [1]))

    shape = tensor_in.get_shape().as_list()
    gamma = weights('constant', [shape[1]], initrange=0.0, name=name + '_gamma')
    beta = weights('constant', [shape[1]], initrange=1.0, name=name + '_beta')

    # Batch Norm Transform
    inv = tf.rsqrt(epsilon + variance, name=name)
    tensor_in = beta * (tensor_in - mean) * inv + gamma

    return tensor_in


def residual_dnn(tensor_in, hidden_units, activation='tanh', distribution='tnorm',
        initrange=1.0, l2=0.0, bn=False, keep_prob=None, fan_scaling=False,
        skiplayers=3, name='dnn'):
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