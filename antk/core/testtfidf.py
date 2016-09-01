import numpy
import scipy.sparse as sps
from antk.core import loader

def zero_rows(X):
    numpy.where(~X.any(axis=1))[0]

tf = [[1, 5, 0, 0, 0], [2, 0, 4, 3, 2]]
tf2 = [[1,1,1,1], [1,1,1,1]]