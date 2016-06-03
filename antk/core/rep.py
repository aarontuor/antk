from antk.core import loader
import numpy

a = loader.read_data_sets('/home/aarontuor/data/ml100k')

print(a)

test = numpy.random.random((3,3))
test2 = numpy.random.random((3,4))
test3 = numpy.random.random((3,5))
datadict = {'feature1': test, 'feature2': test2, 'feature3': test3}
data = loader.DataSet(datadict)
print(data)

x = loader.read_data_sets('/home/aarontuor/data/ml100k')
x.show()


x = loader.read_data_sets('/home/aarontuor/data/ml100k', folders=['user', 'item'])
x.show()

loader.read_data_sets('/home/aarontuor/data/ml100k', folders=['user', 'item'], hashlist=['zip', 'sex', 'year']).show()