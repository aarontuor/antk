"""
Implements a general purpose data loader for python non-sequential machine learning tasks. Several common data transformations are provided in this module, e.g., tfidf, whitening, etc.

Loading, Saving, and Testing
----------------------------
:any:`save`
:any:`export_data`
:any:`load`
:any:`import_data`
:any:`is_one_hot`
:any:`read_data_sets`

Classes
-------
:any:`DataSet`
:any:`DataSets`
:any:`HotIndex`

Data Transforms
---------------
:any:`center`
:any:`l1normalize`
:any:`l2normalize`
:any:`pca_whiten`
:any:`tfidf`
:any:`toOnehot`
:any:`toIndex`
:any:`unit_variance`

Exceptions
----------
:any:`BadDirectoryStructureError`
:any:`MatFormatError`
:any:`SparseFormatError`
:any:`UnsupportedFormatError`

Proposed Extensions
-------------------
DataSet.split(scheme={devtraintest, crossvalidate, traintest} returns DataSets
DataSets.join() returns DataSet (combines train or cross validation)
DataSet + DataSet returns DataSet
DataSets + DataSets returns DataSets
DataSets constructor from list of DataSet objects
"""

import struct
import numpy as np
import scipy.io
import os
import scipy.sparse as sps
from antk.lib import termcolor as tc
import tarfile
import os.path
import urllib
import numbers
import repr

slash = '/'
if os.name == 'nt':
    slash = '\\'  # so this works in Windows


class BadDirectoryStructureError(Exception):
    """Raised when a data directory specified, does not contain
    a subfolder specified in the *folders* argument to
    :any:`read_data_sets`."""
    pass


class UnsupportedFormatError(Exception):
    """Raised when a file is requested to be loaded or saved without one
    of the supported file extensions."""
    pass


class MatFormatError(Exception):
    """Raised if the .mat file being read does not contain a
    variable named *data*."""
    pass


class SparseFormatError(Exception):
    '''Raised when reading a plain text file with .sparsetxt
    extension and there are not three entries per line.'''
    pass


# ==============================================================================================
# =============================DATA STRUCTURES==================================================
# ==============================================================================================

class DataSet(object):
    """
    General data structure for mini-batch gradient descent training involving non-sequential data.

    :param features: A dictionary of string label names to data matrices. Matrices may be :any:`HotIndex`, scipy sparse csr_matrix, or numpy arrays.
    :param labels: A dictionary of string label names to data matrices. Matrices may be :any:`HotIndex`, scipy sparse csr_matrix, or numpy arrays.
    :param num_examples: How many data points.
    :param mix: Whether or not to shuffle per epoch.

    Attributes
    ----------
    features
    index_in_epoch
    labels
    num_examples
    """

    def __init__(self, features, labels=None, num_examples=None, mix=False):
        self._features = features  # hashmap of feature matrices
        if num_examples:
            self._num_examples = num_examples
        else:
            if labels:
                self._num_examples = labels[labels.keys()[0]].shape[0]
            else:
                self._num_examples = features[features.keys()[0]].shape[0]
        if labels:
            self._labels = labels # hashmap of label matrices
        else:
            self._labels = {}
        self._index_in_epoch = 0
        self._mix_after_epoch = mix
        self._last_batch_size = self._num_examples

    def __repr__(self):
        attrs = vars(self)
        return 'antk.core.DataSet object with fields:\n' + '\n'.join("\t%r: %r" % item for item in attrs.items())

    # ======================================================================================
    # =============================PROPERTIES===============================================
    # ======================================================================================
    @property
    def features(self):
        """A dictionary of feature matrices."""
        return self._features

    @property
    def index_in_epoch(self):
        '''The number of datapoints that have been trained on in a particular epoch.'''
        return self._index_in_epoch

    @property
    def labels(self):
        '''A dictionary of label matrices'''
        return self._labels

    @property
    def num_examples(self):
        '''Number of rows (data points) of the matrices in this :any:`DataSet`.'''
        return self._num_examples

    # ======================================================================================
    # =============================PUBLIC METHODS===========================================
    # ======================================================================================
    def reset_index_to_zero(self):
        """Sets **index_in_epoch** to 0."""
        self._index_in_epoch = 0

    def mix_after_epoch(self, mix):
        """
        Whether or not to shuffle after training for an epoch.

        :param mix: True or False
        """
        self._mix_after_epoch = mix

    def next_batch(self, batch_size):
        '''
        Return a sub DataSet of next batch-size examples.
            If no shuffling (mix=False):
                If `batch_size`
                is greater than the number of examples left in the epoch then a batch size DataSet wrapping back to
                past beginning will be returned.
            If shuffling enabled (mix=True):
                If `batch_size` is greater than the number of examples left in the epoch, points will be shuffled and
                batch_size DataSet is returned starting from index 0.

        :param batch_size: int
        :return: A :any:`DataSet` object with the next `batch_size` examples.
        '''
        if batch_size != self._last_batch_size and self._index_in_epoch != 0:
            self.reset_index_to_zero()
        self._last_batch_size = batch_size
        assert batch_size <= self._num_examples
        start = self._index_in_epoch
        if self._index_in_epoch + batch_size > self._num_examples:

            if not self._mix_after_epoch:
                self._index_in_epoch = (self._index_in_epoch + batch_size) % self._num_examples
                end = self._index_in_epoch
                newbatch = DataSet(self._next_batch_(self._features, start, end),
                               self._next_batch_(self._labels, start, end),
                               batch_size)
            else:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._shuffle_(perm, self._features)
                self._shuffle_(perm, self._labels)
                start = 0
                end = batch_size
                newbatch = DataSet(self._next_batch_(self._features, start, end),
                               self._next_batch_(self._labels, start, end),
                               batch_size)
                self._index_in_epoch = batch_size
            return newbatch
        else:
            end = self._index_in_epoch + batch_size
            self._index_in_epoch = (batch_size + self._index_in_epoch) % self._num_examples
            if self._index_in_epoch == 0 and self._mix_after_epoch:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._shuffle_(perm, self._features)
                self._shuffle_(perm, self._labels)
            return DataSet(self._next_batch_(self._features, start, end),
                           self._next_batch_(self._labels, start, end),
                           batch_size)

    def show(self):
        '''
        Pretty printing of all the data (dimensions, keys, type) in the :any:`DataSet` object
        '''

        print('features:')
        for name, feature, in self.features.iteritems():
            if type(feature) is HotIndex:
                print('\t %s: vec.shape: %s dim: %s %s' % (name, feature.vec.shape,
                                                           feature.dim, type(feature)))
            else:
                print('\t %s: %s %s' % (name, feature.shape, type(feature)))
        print('labels:')
        for name, label in self.labels.iteritems():
            if type(label) is HotIndex:
                print('\t %s: vec.shape: %s dim: %s %s' % (name, (label.vec.shape),
                                                           label.dim, type(label)))
            else:
                print('\t %s: %s %s' % (name, label.shape, type(label)))

    def showmore(self):
        '''
        Print a sample of the first up to twenty rows of matrices in DataSet
        '''

        print('features:')
        for name, feature in self.features.iteritems():
            print('\t %s: \nFirst twenty rows:\n%s\n' % (name, feature[1:min(20, feature.shape[0])]))
        print('labels:')
        for name, label in self.labels.iteritems():
            print('\t %s: \nFirst twenty rows:\n%s\n' % (name, feature[1:min(20, feature.shape[0])]))

    # ======================================================================================
    # =============================PRIVATE METHODS===========================================
    # ======================================================================================
    def _shuffle_(self, order, datamap):
        '''
        :param order: A list of the indices for the row permutation
        :param datamap:
        :return: void
        Shuffles the rows an individual matrix in the :any:`DataSet` object.'
        '''
        for matrix in datamap:
            if type(datamap[matrix]) is HotIndex:
                datamap[matrix] = HotIndex(datamap[matrix].vec[order], datamap[matrix].dim)
            else:
                datamap[matrix] = datamap[matrix][order]

    def _next_batch_(self, datamap, start, end=None):
        '''
        :param datamap: A hash map of matrices
        :param start: starting row
        :param end: ending row
        :return: A hash map of slices of matrices from row start to row end
        '''
        if end is None:
            end = self._num_examples
        batch_data_map = {}
        if end <= start:
            start2 = 0
            end2 = end
            end = self._num_examples
            wrapdata = {}
            for matrix in datamap:
                if type(datamap[matrix]) is HotIndex:
                    wrapdata[matrix] = datamap[matrix].vec[start2:end2]
                    batch_data_map[matrix] = datamap[matrix].vec[start:end]
                else:
                    wrapdata[matrix] = datamap[matrix][start2:end2]
                    batch_data_map[matrix] = datamap[matrix][start:end]
                if sps.issparse(batch_data_map[matrix]):
                        batch_data_map[matrix] = sps.vstack([batch_data_map[matrix], wrapdata[matrix]])
                else:
                    batch_data_map[matrix] = np.concatenate([batch_data_map[matrix], wrapdata[matrix]], axis=0)
        else:
            for matrix in datamap:
                if type(datamap[matrix]) is HotIndex:
                    batch_data_map[matrix] = datamap[matrix].vec[start:end]
                else:
                    batch_data_map[matrix] = datamap[matrix][start:end]
        return batch_data_map

class DataSets(object):
    '''
    A record of DataSet objects with a display function.
    '''

    def __init__(self, datasets_map, mix=False):
        for k, v in datasets_map.items():
            setattr(self, k, DataSet(v['features'], v['labels'], v['num_examples'], mix=mix))

    def __repr__(self):
        attrs = vars(self)
        return 'antk.core.DataSets object with fields:\n' + '\n'.join("\t%s: %s" % item for item in attrs.items())

    def show(self):
        """
        Pretty print data attributes.
        """
        datasets = [s for s in dir(self) if not s.startswith('__') and not s == 'show' and not s == 'showmore']
        for dataset in datasets:
            print tc.colored(dataset + ':', 'yellow')
            getattr(self, dataset).show()

    def showmore(self):
        """
        Pretty print data attributes, and data.
        """
        datasets = [s for s in dir(self) if not s.startswith('__') and not s == 'show' and not s == 'showmore']
        for dataset in datasets:
            print tc.colored(dataset + ':', 'yellow')
            getattr(self, dataset).showmore()



class IndexVector(object):
    """
    Index vector representation of one hot matrix.

    Parameters
    ----------
    matrix: scipy.sparse.csr_matrix or numpy array
            A one hot matrix or vector of *on* indices of a one hot matrix.
            If **matrix** is a vector of indices and no dimension argument is supplied
            then dimension is set to the maximum index value + 1.

    Notes
    -----
    IndexVector objects follow the python sequence protocol, so slicing,
    indexing and iteration behave as you might expect.
    Slices of an IndexVector return another IndexVector.
    Indexing returns an integer. Iteration will loop over all the elements
    in the **vec** attribute.

    Examples
    --------
    >>> import numpy as np
    >>> from antk.core import loader
    >>> xhot = np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]])
    >>> xindex = loader.IndexVector(xhot)
    >>> xindex.vec
    array([0, 0, 1, 2])
    >>> xindex.dim
    3
    >>> xindex.hot() #doctest: +NORMALIZE_WHITESPACE
    <4x3 sparse matrix of type '<type 'numpy.float64'>'
        with 4 stored elements in Compressed Sparse Row format>
	>>> xindex.hot().toarray() #doctest: +NORMALIZE_WHITESPACE
	array([[ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> xindex.shape
    (4, 3)
    >>> xindex
    <class 'antk.core.loader.IndexVector'>(shape=(4, 3))
    vec=[0, 0, 1, 2]
    dim=3
    >>> xindex[0]
    0
    >>> xindex[1:3]
    <class 'antk.core.loader.IndexVector'>(shape=(2, 3))
    vec=[0, 1]
    dim=3
    >>> [index+2 for index in xindex]
    [2, 2, 3, 4]
    """
    def __init__(self, matrix, dimension=None):
        if is_one_hot(matrix):
            self._dim = matrix.shape[1]
            self._vec = toIndex(matrix).flatten()
        else:
            if matrix.dtype != np.int64:
                raise ValueError('Indices must be integers.')
            self._dim = dimension
            if self.dim is None:
                self._dim = np.amax(matrix) + 1
            self._vec = np.array(matrix).flatten()

    def __repr__(self):
        vector = repr.repr(self._vec.tolist())
        return '%s(shape=%s)\nvec=%s\ndim=%s' % (type(self), self.shape, vector, self._dim)


    def __str__(self):
        return '%s(shape=%s)\nvec=%s\ndim=%s' % (type(self), self.shape, self._vec, self._dim)

    def __iter__(self):
        return iter(self._vec)

    def __eq__(self, other):
        return np.array_equal(self._vec, other._vec) and self._dim == other._dim

    def __len__(self):
        return self._vec.shape[0]

    def __getitem__(self, index):
        cls = type(self)
        if isinstance(index, numbers.Integral):
            return self._vec[index]
        elif isinstance(index, slice):
            return cls(self._vec[index], self.dim)
        else:
            raise TypeError('Indices must be integers.')

    @property
    def dim(self):
        '''The feature dimension (number of columns) of the one hot matrix.'''
        return self._dim

    @property
    def vec(self):
        '''The vector of hot indices.'''
        return self._vec

    @property
    def shape(self):
        '''The shape of the one hot matrix encoded.'''
        return (self.vec.shape[0], self.dim)

    def hot(self):
        """
        :return: A one hot scipy sparse csr_matrix
        """
        return toOnehot(self)

class HotIndex(IndexVector):
    """Same class as IndexVector. This is the legacy name of the class."""
    pass
# ===================================================================================
# ====================== I/0 ========================================================
# ===================================================================================

def load(filename):
    '''
    Calls :any:`import_data`.
    Decides how to load data into python matrices by file extension.
    Raises :any:`UnsupportedFormatError` if extension is not one of the supported
    extensions (mat, sparse, binary, dense, sparsetxt, densetxt, index).

    :param filename: A file of an accepted format representing a matrix.
    :return: A numpy matrix, scipy sparse csr_matrix, or any:`HotIndex`.
    '''
    return import_data(filename)

def import_data(filename):
    '''
    Decides how to load data into python matrices by file extension.
    Raises :any:`UnsupportedFormatError` if extension is not one of the supported
    extensions (mat, sparse, binary, dense, sparsetxt, densetxt, index).

    :param filename: A file of an accepted format representing a matrix.
    :return: A numpy matrix, scipy sparse csr_matrix, or any:`HotIndex`.
    '''
    extension = filename.split(slash)[-1].split('.')[-1].strip()
    if extension == 'mat':
        mat_file_map = scipy.io.loadmat(filename)
        if 'data' not in mat_file_map:
            raise MatFormatError('Matrix in .mat file ' +
                                   filename + ' must be named "data"')
        if mat_file_map['data'].shape[0] == 1:
            return np.transpose(mat_file_map['data'])
        else:
            return mat_file_map['data']
    elif extension == 'index':
        return _imatload(filename)
    elif extension == 'sparse':
        return _smatload(filename)
    elif extension == 'binary' or extension == 'dense':
        return _matload(filename)
    elif extension == 'sparsetxt':
        X = np.loadtxt(filename)
        if X.shape[1] != 3:
            raise SparseFormatError('Sparse Format: row col val')
        return sps.csr_matrix((X[:, 2], (X[:, 0], X[:, 1])))
    elif extension == 'densetxt':
        return np.loadtxt(filename)
    else:
        raise UnsupportedFormatError('Supported extensions: '
                                       'mat, sparse, binary, sparsetxt, densetxt, index')

def save(filename, data):
    '''
    Calls :any`export_data`.
    Decides how to save data by file extension.
    Raises :any:`UnsupportedFormatError` if extension is not one of the supported
    extensions (mat, sparse, binary, dense, index).
    Data contained in .mat files should be saved in a matrix named *data*.

    :param filename: A file of an accepted format representing a matrix.
    :param data: A numpy array, scipy sparse matrix, or :any:`HotIndex` object.
    '''
    export_data(filename, data)

def export_data(filename, data):
    '''
    Decides how to save data by file extension.
    Raises :any:`UnsupportedFormatError` if extension is not one of the supported
    extensions (mat, sparse, binary, dense, index).
    Data contained in .mat files should be saved in a matrix named *data*.

    :param filename: A file of an accepted format representing a matrix.
    :param data: A numpy array, scipy sparse matrix, or :any:`HotIndex` object.
    '''
    extension = filename.split(slash)[-1].split('.')[-1].strip()
    if extension == 'mat':
        scipy.io.savemat(filename, {'data': data})
    elif extension == 'index':
        if not isinstance(data, HotIndex):
            raise UnsupportedFormatError('Only HotIndex objects may be saved in .index format.')
        _imatsave(filename, data)
    elif extension == 'sparse':
        if not sps.issparse(data):
            raise UnsupportedFormatError('Only scipy sparse matrices may be saved in .sparse format.')
        _smatsave(filename, data)
    elif extension == 'binary' or extension == 'dense':
        if sps.issparse(data):
            raise UnsupportedFormatError('Only numpy 2d arrays may be saved in .binary or .dense format.')
        _matsave(filename, data)
    elif extension == 'densetxt':
        if sps.issparse(data):
            raise UnsupportedFormatError('Only numpy 2d arrays may be saved in .densetxt format')
        np.savetxt(filename, data)
    elif extension == 'sparsetxt':
        if not sps.issparse(data):
            raise UnsupportedFormatError('Only scipy sparse matrices may be saved in .sparsetxt format.')
        indices = list(data.nonzero())
        indices.append(data.data)
        data = [m.reshape((-1,1)) for m in indices]
        data = np.concatenate(data, axis=1)
        np.savetxt(filename, data)
    else:
        raise UnsupportedFormatError('Supported extensions: '
                                       'mat, sparse, binary, dense, index, sparsetxt, densetxt')

def _write_int64(file_obj, num):
    """
    Writes an 8 byte integer to a file in binary format. From David Palzer.

    :param file_obj: the open file object to write to
    :param num: the integer to write, will be converted to a long int
    """
    file_obj.write(struct.pack('q', long(num)))


def _read_int64(file_obj):
    """
    Reads an 8 byte binary integer from a file. From David Palzer.

    :param file_obj: The open file object from which to read.
    :return: The eight bytes read from the file object interpreted as a long int.
    """
    return struct.unpack('q', file_obj.read(8))[0]


def _matload(filename):
    """
    Reads in a dense matrix from binary (dense) file filename. From `David Palzer`_.

    :param filename: file from which to read.
    :return: the matrix which has been read.
    """
    f = open(filename, 'r')
    m = _read_int64(f)
    n = _read_int64(f)
    x = np.fromfile(f, np.dtype(np.float64), -1, "")
    x = x.reshape((m, n), order="FORTRAN")
    f.close()
    return np.array(x)


def _matsave(filename, x):
    """
    Saves the input matrix to the input file in dense format. From `David Palzer`_.

    :param filename: file to write to
    :param x: matrix to write
    """
    if len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0], 1))
    f = open(filename, 'wb')
    _write_int64(f, x.shape[0])
    _write_int64(f, x.shape[1])
    x.astype(np.float64, copy=False).ravel('FORTRAN').tofile(f)
    f.close()


def _smatload(filename):
    """
    Reads in a sparse matrix from file. From `David Palzer`_.

    :param filename: the file from which to read
    :return: a sparse matrix created from the sparse data
    """
    f = open(filename, 'r')
    row = _read_int64(f)
    col = _read_int64(f)
    nnz = _read_int64(f)
    S = np.fromfile(f, 'd', 3 * nnz)
    f.close()
    S = S.reshape((nnz, 3))
    rows = S[:, 0].astype(int) - 1
    cols = S[:, 1].astype(int) - 1
    vals = S[:, 2]
    return sps.csr_matrix((vals, (rows, cols)), shape=(row, col))


def _smatsave(filename, t):
    """
    Saves the input matrix to the input file in sparse format. From `David Palzer`_.
    :param filename:
    :param t:
    """
    t = sps.csr_matrix(t,copy=False)
    f = open(filename, 'wb')
    _write_int64(f, t.shape[0])
    _write_int64(f, t.shape[1])
    indices = t.nonzero()
    idxs = np.vstack((indices[0], indices[1]))
    _write_int64(f, len(indices[1]))
    ti = np.mat(t[indices])
    indices = np.concatenate((idxs, ti))
    indices[0:2, :] += 1
    indices.astype(float, copy=False).ravel('F').tofile(f)
    f.close()



def _imatload(filename):
    """
    Reads in a :any:`HotIndex` matrix from file
    :param filename: the file from which to read where a :any:`HotIndex` object was stored.
    :return: A :any:`HotIndex` object.
    """
    f = open(filename, 'r')
    vec_length = _read_int64(f)
    dim = _read_int64(f)
    vec = np.fromfile(f, 'd', vec_length)
    f.close()
    vec = vec.astype(int) - 1
    return HotIndex(vec, dim)


def _imatsave(filename, index_vec):
    """
    Saves the input matrix to the input file in sparse format

    :param filename: Filename to save to.
    :param index_vec: A :any:`HotIndex` object.
    """
    f = open(filename, 'wb')
    vector = index_vec.vec
    vector = vector + 1
    _write_int64(f, vector.shape[0])
    _write_int64(f, index_vec.dim)
    vector.astype(float, copy=False).tofile(f)
    f.close()


def makedirs(datadirectory, sub_directory_list=('train', 'dev', 'test')):
    '''
    :param datadirectory: Name of the directory you want to create containing the subdirectory folders.
     If the directory already exists it will be populated with the subdirectory folders.
    :param sub_directory_list: The list of subdirectories you want to create
    :return: void
    '''

    if not datadirectory.endswith(slash):
        datadirectory += slash
    os.system('mkdir ' + datadirectory)
    for sub in sub_directory_list:
        os.system('mkdir ' + datadirectory + sub)


def read_data_sets(directory, folders=('train', 'dev', 'test'), hashlist=(), mix=False):
    '''
    :param directory: Root directory containing data to load.
    :param folders: The subfolders of *directory* to read data from by default there are train, dev, and test folders. If you want others you have to make an explicit list.
    :param hashlist: If you provide a hashlist these files and only these files will be added to your :any:`DataSet` objects.
        It you do not provide a hashlist then anything with
        the privileged prefixes labels_ or features_ will be loaded.
    :return: A :any:`DataSets` object.
    '''

    if not directory.endswith(slash):
        directory += slash
    dir_files = os.listdir(directory)

    datasets_map = {}
    for folder in folders:  # iterates over keys
        dataset_map = {'features': {}, 'labels': {}, 'num_examples': 0}
        print('reading ' + folder + '...')
        if folder not in dir_files:
            raise BadDirectoryStructureError('Need ' + folder + ' folder in ' + directory + ' directory.')
        file_list = os.listdir(directory + folder)
        for filename in file_list:
            prefix = filename.split('_')[0]
            if prefix == 'features' or prefix == 'labels':
                prefix_ = prefix + '_'
                descriptor = (filename.split('.')[0]).split(prefix_)[-1]
                if (not hashlist) or (descriptor in hashlist):
                    dataset_map[prefix][descriptor] = import_data(directory + folder + slash + filename)
                    if prefix == 'labels':
                        dataset_map['num_examples'] = dataset_map[prefix][descriptor].shape[0]
        datasets_map[folder] = dataset_map
    return DataSets(datasets_map, mix=mix)


# ===================================================================================
# ====================== DATA MANIPULATION===========================================
# ===================================================================================
def toOnehot(X, dim=None):
    '''
    :param X: Vector of indices or :any:`HotIndex` object
    :param dim: Dimension of indexing
    :return: A sparse csr_matrix of one hots.

        Examples
        --------
        >>> import np
        >>> from antk.core import loader
        >>> x = np.array([0, 1, 2, 3])
        >>> loader.toOnehot(x) #doctest: +ELLIPSIS
        <4x4 sparse matrix of type '<type 'np.float64'>'...
        >>> loader.toOnehot(x).toarray()
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])
        >>> x = loader.HotIndex(x, dimension=8)
        >>> loader.toOnehot(x).toarray()
        array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.]])


    '''
    if isinstance(X, IndexVector):
        dim = X.dim
        X = X.vec
    else:
        if dim is None:
            dim = np.amax(X) + 1
    return sps.csr_matrix(([1.0]*X.shape[0], (range(X.shape[0]), X.astype(int))), shape=(X.shape[0], dim))


def is_one_hot(A):
    '''
    :param A: A 2-d numpy array or scipy sparse matrix
    :return: True if matrix is a sparse matrix of one hot vectors, False otherwise

        Examples
        --------
        >>> import numpy as np
        >>> from antk.core import loader
        >>> x = np.eye(3)
        >>> loader.is_one_hot(x)
        True
        >>> x *= 5
        >>> loader.is_one_hot(x)
        False
        >>> x = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        >>> loader.is_one_hot(x)
        True
        >>> x[0,1] = 2
        >>> loader.is_one_hot(x)
        False

    '''
    B = sps.csr_matrix(A)
    (i, j, v) = sps.find(B)
    return (np.sum(v) == B.shape[0] and
            np.unique(i).shape[0] == B.shape[0] and
            np.unique(v).shape[0] == 1 and
            np.unique(v)[0] == 1 and
            len(A.shape) == 2)


def toIndex(A):
    '''
    :param A: A matrix of one hot row vectors.
    :return: The hot indices.

        Examples
        --------

        >>> import np
        >>> from antk.core import loader
        >>> x = np.array([[1,0,0], [0,0,1], [1,0,0]])
        >>> loader.toIndex(x)
        array([0, 2, 0])
    '''
    if is_one_hot(A):
        if sps.issparse(A):
            return sps.find(A)[1]
        else:
            return np.nonzero(A)[1]
    else:
        raise ValueError('Argument to function must be a one hot matrix.')


def center(X, axis=None):
    """

    :param X: A matrix to center about the mean(over columns axis=0, over rows axis=1, over all entries axis=None)
    :return: A matrix with entries centered along the specified axis.
    """
    if sps.issparse(X):
        X = X.todense()
        return sps.csr_matrix(X - np.mean(X, axis=axis))
    else:
        return X - np.mean(X, axis=axis)


def unit_variance(X, axis=None):
    """

    :param X: A matrix to transfrom to have unit variance (over columns axis=0, over rows axis=1, over all entries axis=None)
    :return: A matrix with unit variance along the specified axis.
    """
    if sps.isspparse(X):
        X = X.todense()
        return sps.csr_matrix(X / np.std(X, axis=axis))
    else:
        return X / np.std(X, axis=axis)

def pca_whiten(X):
    """
    Returns matrix with PCA whitening transform applied.
    This transform assumes that data points are rows of matrix.

    :param X: Numpy array, scipy sparse matrix
    :param axis: Axis to whiten over.
    :return:
    """
    if sps.issparse(X):
        return sps.csr_matrix(pca_whiten(X.todense()))
    else:
        X -= np.mean(X, axis=0)
        cov = np.dot(X.T, X)/X.shape[0]
        U, S, V = np.linalg.svd(cov)
        Xrot = np.dot(X, U)
        return Xrot/np.sqrt(S + 1e-5)

# ===================================================
# Normalizations for tfidf or whatever
# ====================================================
def l2normalize(X, axis=1):
    """
    axis=1 normalizes each row of X by norm of said row. :math:`l2normalize(X)_{ij} = \\frac{X_{ij}}{\sqrt{\sum_k X_{
    ik}^2}}`

    axis=0 normalizes each column of X by norm of said column. :math:`l2normalize(X)_{ij} = \\frac{X_{ij}}{\sqrt{\sum_k
    X_{kj}^2}}`

    axis=None normalizes entries of X  by norm of X. :math:`l2normalize(X)_{ij} = \\frac{X_{ij}}{\sqrt{\sum_k \sum_p
    X_{kp}^2}}`

    :param X: A scipy sparse csr_matrix or numpy array.
    :param axis: The dimension to normalize over.
    :return: A normalized matrix.
    """
    if sps.issparse(X):
        X = X.toarray()
    normalized_matrix = X/np.linalg.norm(X, ord=2, axis=axis, keepdims=True)
    if sps.issparse(X):
        normalized_matrix = sps.csr_matrix(normalized_matrix)
    return normalized_matrix

def l1normalize(X, axis=1):
    """
    axis=1 normalizes each row of X by norm of said row. :math:`l1normalize(X)_{ij} = \\frac{X_{ij}}{\sum_k |X_{ik}|}`

    axis=0 normalizes each column of X by norm of said column. :math:`l1normalize(X)_{ij} = \\frac{X_{ij}}{\sum_k
    |X_{kj}|}`

    axis=None normalizes entries of X  by norm of X. :math:`l1normalize(X)_{ij} = \\frac{X_{ij}}{\sum_k \sum_p
    |X_{kp}|}`


    :param X: A scipy sparse csr_matrix or numpy array.
    :param axis: The dimension to normalize over.
    :return: A normalized matrix.
    """
    if sps.issparse(X):
        X = X.toarray()
    normalized_matrix = X/np.linalg.norm(X, ord=1, axis=axis, keepdims=True)
    if sps.issparse(X):
        normalized_matrix = sps.csr_matrix(normalized_matrix)
    return normalized_matrix

def maxnormalize(X, axis=1):
    """
    axis=1 normalizes each row of X by norm of said row. :math:`maxnormalize(X)_{ij} = \\frac{X_{ij}}{max(X_{i:})}`

    axis=0 normalizes each column of X by norm of said column. :math:`maxnormalize(X)_{ij} = \\frac{X_{ij}}{max(X_{
    :j})}`

    axis=None normalizes entries of X  norm of X. :math:`maxnormalize(X)_{ij} = \\frac{X_{ij}}{max(X)}`


    :param X: A scipy sparse csr_matrix or numpy array.
    :param axis: The dimension to normalize over.
    :return: A normalized matrix.
    """
    if sps.issparse(X):
        X = X.toarray()
    normalized_matrix = X/np.linalg.norm(X, ord=np.inf, axis=axis, keepdims=True)
    if sps.issparse(X):
        normalized_matrix = sps.csr_matrix(normalized_matrix)
    return normalized_matrix

NORM = {'l2': l2normalize,
         'count': l1normalize,
         'max': maxnormalize}


def tfidf(X, norm='l2'):
    """
    :param X: A document-term matrix.
    :param norm: Normalization strategy: 'l2row': normalizes the scores of rows by length of rows after basic tfidf (each document vector is a unit vector), 'count': normalizes the scores of rows by the the total word count of a document. 'max' normalizes the scores of rows by the maximum count for a single word in a document.
    :return: Returns tfidf of document-term matrix X with optional normalization.
    """
    X = sps.csr_matrix(X)
    idf = np.log(X.shape[0]/X.sign().sum(0))
    # make a diagonal matrix of idf values to matrix multiply with tf.
    IDF = sps.csr_matrix((idf.tolist()[0], (range(X.shape[1]), range(X.shape[1]))))
    if norm == 'count' or norm == 'max':
        # Only normalizing tf
        return sps.csr_matrix(sps.csr_matrix(NORM[norm](X)).dot(IDF))
    elif norm == 'l2':
        # normalizing tfidf
        return sps.csr_matrix(NORM[norm](X.dot(IDF)))
    else:
        # no normalization
        return sps.csr_matrix(X.dot(IDF))

# ========================================================
# ==================MISC==================================
# ========================================================
def untar(fname):
    """
    Untar and ungzip a file in the current directory.
    :param fname: Name of the .tar.gz file
    """
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname)
        tar.extractall()
        tar.close()
        print("Extracted " + fname + " in Current Directory")
    else:
        print("Not a tar.gz file: '%s '" % fname)

def maybe_download(filename, directory, source_url):
    """
    Download the data from source url, unless it's already here. From https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/base.py

    :param filename: string, name of the file in the directory.
    :param directory: string, path to working directory.
    :param source_url: url to download from if file doesn't exist.
    :return: Path to resulting file.
    """

    filepath = os.path.join(directory, filename)
    if not os.path.isfile(filepath):
        urlopen = urllib.URLopener()
        urlopen.retrieve(source_url, filepath)
    return filepath

