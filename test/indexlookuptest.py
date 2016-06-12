import tensorflow as tf

from antk.core import loader
import numpy
# data = loader.read_data_sets('/home/aarontuor/data/ml100k', folders=['dev', 'user', 'item'])
data = loader.read_data_sets('/home/aarontuor/data/ml100k', folders=['dev'], hashlist=['user', 'item', 'rating'])
data = data.dev

tmat = numpy.hstack((numpy.reshape(data.features['user'].vec, (-1, 1)), numpy.reshape(data.features['item'].vec, (-1, 1))))
print(tmat.shape)

b = tmat[tmat[:, 0] == range(data.features['user'].dim)]
print(b)
#data.showmore()

