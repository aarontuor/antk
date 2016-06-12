from antk.core import loader
import numpy
import random
indices = numpy.array(range(20))
a = loader.HotIndex(indices, 19)
print(a)
print('%r' % a)
print('%s' % a)

b = range(20)
random.shuffle(b)
print(b)
print(b)
print(a[1:5])
print(a[b])
print(a[1,1])
print(a[0,1])

