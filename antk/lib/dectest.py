import functools
import inspect
def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(reversed(args), reversed(defaults)))

def funcwrap(func):
    def newfunc(*args, **kwargs):
        defaults = get_default_args(func)
        if 'grape' not in defaults:
            defaults['grape'] = 'grape'
        if 'grape' not in kwargs:
            kwargs['grape'] = defaults['grape']
        kwargs['grape'] = 'sour' + kwargs['grape']
        return func(*args, **kwargs)
    return newfunc

@funcwrap
def funck(apple, orange='orange', grape='grape'):
    print(apple.split(','), orange, grape)

funck('pizza', orange='orange', grape='tomato')