import functools
import tensorflow as tf
import inspect
import collections
import numpy

def node_op(func):
    @functools.wraps(func)
    def new_function(*args, **kwargs):
        defaults = get_default_args(func)
        if 'name' not in defaults:
            defaults['name'] = 'unnamed_tensor'
        if 'name' not in kwargs:
            kwargs['name'] = defaults['name']
        with tf.variable_scope(kwargs['name']):
            tensor_out = func(*args, **kwargs)
        def node_repr(tensor_node):
            return 'Tensor("%s", shape=%s, dtype=%r)' % (tensor_node.name,
                                                         tensor_node.get_shape().as_list(),
                                                         tensor_node.dtype)
        tensor_out.__class__.__repr__ = node_repr
        tensor_out.__class__.__str__ = node_repr
        if isinstance(tensor_out, collections.Iterable):
            for t in tensor_out:
                tf.add_to_collection(kwargs['name'], t)
        else:
            tf.add_to_collection(kwargs['name'], tensor_out)
        return tensor_out
    return new_function

def act(func):
    func = node_op(func)
    @functools.wraps(func)
    def non_linearitied(*args, **kwargs):
        tensor_out = func(*args, **kwargs)
        tf.add_to_collection('activation_layers', tensor_out)
        return tensor_out
    return non_linearitied

@act
def tanhlecun(tensor_in):
    """
    `Efficient BackProp`_
    Sigmoid with the following properties:
    (1) :math:`f(\pm 1) = \pm 1` (2) second derivative of *f* is maximum at :math:`\pm 1` (3) Effective gain is close to 1
    """
    return 1.7159*tf.nn.tanh((2.0/3.0) * tensor_in)


sigmoid = act(tf.nn.sigmoid)
tanh = act(tf.nn.tanh)
relu = act(tf.nn.relu)
relu6 = act(tf.nn.relu6)
softplus = act(tf.nn.softplus)

ACTIVATION = {'sigmoid': sigmoid,
              'tanh': tanh,
              'relu': relu,
              'relu6': relu6,
              'softplus': softplus,
              'tanhlecun': tanhlecun}


def ph_rep(ph):
    """
    Convenience function for representing a tensorflow placeholder.

    :param ph: A `tensorflow`_ `placeholder`_.
    :return: A string representing the placeholder.
    """
    return 'Placeholder("%s", shape=%s, dtype=%r)' % (ph.name, ph.get_shape().as_list(), ph.dtype)

def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(reversed(args), reversed(defaults)))




def pholder(func):
    func = node_op(func)
    @functools.wraps(func)
    def new_function(*args, **kwargs):
        tensor_out = func(*args, **kwargs)
        def ph_rep(ph):
            return 'Placeholder("%s", shape=%s, dtype=%r)' % (ph.name, ph.get_shape().as_list(), ph.dtype)
        tensor_out.__class__.__repr__ = ph_rep
        tensor_out.__class__.__str__ = ph_rep
        return tensor_out
    return new_function

def variable(func):
    func = node_op(func)
    @functools.wraps(func)
    def new_function(*args, **kwargs):
        tensor_out = func(*args, **kwargs)
        def vr_rep(vr):
            return 'Variable("%s", shape=%s, dtype=%r)' % (vr.name, vr.get_shape().as_list(), vr.dtype)
        tensor_out.__class__.__repr__ = vr_rep
        tensor_out.__class__.__str__ = vr_rep
        return tensor_out
    return new_function


def loss_function(func):
    func = node_op(func)
    @functools.wraps(func)
    def loss_functioned(*args, **kwargs):
        tensor_out = func(*args, **kwargs)
        def loss_repr(loss):
            return 'Loss_Tensor("%s", shape=%s, dtype=%r)' % (loss.name, loss.get_shape().as_list(), loss.dtype)
        tensor_out.__class__.__repr__ = loss_repr
        tensor_out.__class__.__str__ = loss_repr
        tf.add_to_collection(kwargs['name'] + '_loss', tensor_out)
        return tensor_out
    return loss_functioned


def neural_net(func):

    func = node_op(func)
    @functools.wraps(func)
    def nnetted(*args, **kwargs):
        defaults = get_default_args(func)
        if 'activation' not in defaults:
            defaults['activation'] = 'tanh'
        if 'activation' not in kwargs:
            kwargs['activation'] = defaults['activation']
        if type(kwargs['activation']) is str:
            if kwargs['activation'] not in ACTIVATION:
                raise ValueError('Unrecognized activation parameter "%s": \n'
                                 'Accepted activations %s' % (kwargs['activation'], ACTIVATION.keys()))
            else:
                kwargs['activation'] = ACTIVATION[kwargs['activation']]
        elif not callable(kwargs['activation']):
            raise TypeError('activation parameter must be a string, or function. Received %s of type %s' %
                            (kwargs['activation'], type(kwargs)))
        else:
            kwargs['activation'] = act(kwargs['activation'])
        return nnetted



