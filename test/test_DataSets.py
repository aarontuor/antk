from antk.core.loader import DataSets
from antk.core.loader import DataSet
import numpy as np

def test_init1():
    d = DataSets({'train': DataSet({'id': np.eye(5), 'one': np.ones((5,6))}),
                  'dev': DataSet({'id': 5*np.eye(5), 'one': 5*np.ones((5,6))})})
    assert np.array_equal(d.dev.features['id'] - d.train.features['id'], 4*np.eye(5))

def test_init2():
    d = DataSets({})




