from antk.core import loader
from loader import import_data, export_data
import numpy as np

def test_import_dense():
    """
    Test the saving and loading of .dense format.
    """
    x = np.random((20, 50))
    assert x == import_data(export_data('/tmp/test.dense', x))
