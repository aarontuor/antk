import functools
import tensorflow as tf
import inspect

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
    @functools.wraps(func)
    def pholdered(*args, **kwargs):
        defaults = get_default_args(func)
        if 'name' not in defaults:
            defaults['name'] = 'unnamed_tensor'
        if 'name' not in kwargs:
            kwargs['name'] = defaults['name']
        tensor_out = func(*args, **kwargs)
        def ph_rep(ph):
            return 'Placeholder("%s", shape=%s, dtype=%r)' % (ph.name, ph.get_shape().as_list(), ph.dtype)
        tensor_out.__class__.__repr__ = ph_rep
        tensor_out.__class__.__str__ = ph_rep
        return tensor_out
    return pholdered

def variable(func):
    @functools.wraps(func)
    def variabled(*args, **kwargs):
        defaults = get_default_args(func)
        if 'name' not in defaults:
            defaults['name'] = 'unnamed_tensor'
        if 'name' not in kwargs:
            kwargs['name'] = defaults['name']
        tensor_out = func(*args, **kwargs)
        def vr_rep(vr):
            return 'Variable("%s", shape=%s, dtype=%r)' % (vr.name, vr.get_shape().as_list(), vr.dtype)
        tensor_out.__class__.__repr__ = vr_rep
        tensor_out.__class__.__str__ = vr_rep
        tf.add_to_collection(kwargs['name'] + '_weights', tensor_out)
        return tensor_out
    return variabled

def node_op(func):
    @functools.wraps(func)
    def node_opped(*args, **kwargs):
        defaults = get_default_args(func)
        if 'name' not in defaults:
            defaults['name'] = 'unnamed_tensor'
        if 'name' not in kwargs:
            kwargs['name'] = defaults['name']
        tensor_out = func(*args, **kwargs)
        def no_rep(no):
            return 'Tensor("%s", shape=%s, dtype=%r)' % (no.name, no.get_shape().as_list(), no.dtype)

        tensor_out.__class__.__repr__ = no_rep
        tensor_out.__class__.__str__ = no_rep
        tf.add_to_collection(kwargs['name'], tensor_out)
        return tensor_out
    return node_opped
