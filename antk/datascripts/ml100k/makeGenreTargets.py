import numpy
from antk.core import loader
import scipy.sparse as sps
from numpy.linalg import norm
genres = loader.import_data('/home/aarontuor/data/ml100k/item/features_genre.mat')
dev_item = loader.import_data('/home/aarontuor/data/ml100k/dev/features_item.index')
train_item = loader.import_data('/home/aarontuor/data/ml100k/train/features_item.index')
test_item = loader.import_data('/home/aarontuor/data/ml100k/test/features_item.index')
words = loader.import_data('/home/aarontuor/data/ml100k/item/features_bin_doc_term.mat')



genre_dist = genres/(norm(genres, axis=1, keepdims=True) * norm(genres, axis=1, keepdims=True))
devgenre = genres[dev_item.vec]
testgenre = genres[test_item.vec]
traingenre = genres[train_item.vec]
devwords = words[dev_item.vec]
testwords = words[test_item.vec]
trainwords = words[train_item.vec]
print(devgenre.shape)
print(testgenre.shape)
print(traingenre.shape)

print(devwords.shape)
print(testwords.shape)
print(trainwords.shape)

loader.export_data('/home/aarontuor/data/ml100k/dev/labels_genre.mat', devgenre)
loader.export_data('/home/aarontuor/data/ml100k/train/labels_genre.mat', traingenre)
loader.export_data('/home/aarontuor/data/ml100k/test/labels_genre.mat', testgenre)
loader.export_data('/home/aarontuor/data/ml100k/dev/features_words.mat', devwords)
loader.export_data('/home/aarontuor/data/ml100k/train/features_words.mat', trainwords)
loader.export_data('/home/aarontuor/data/ml100k/test/features_words.mat', testwords)