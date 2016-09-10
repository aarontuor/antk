import antk.core.loader as loader
import numpy as np
import scipy.sparse as sps
"""
:any:`center`

:any:`l1normalize`

:any:`l2normalize`

:any:`pca_whiten`

:any:`tfidf`

:any:`unit_variance`
"""
x = np.array([[0.0,0.0,6.0],
              [2.0,4.0,2.0],
              [2.0,6.0,0.0]])
y = sps.csr_matrix(x)
# numpy.testing.assert_array_almost_equal

def test_l1_dense_test_axis0():
    assert np.array_equal(loader.l1normalize(x, axis=0),
                          np.array([[0.0, 0.0, .75],
                                    [.5, .4, .25],
                                    [.5, .6, 0.0]]))

def test_l1_sparse_test_axis0():
    assert np.array_equal(loader.l1normalize(y, axis=0),
                          np.array([[0.0, 0.0, .75],
                                    [.5, .4, .25],
                                    [.5, .6, 0.0]]))


def test_l1_dense_test_axis1():
    assert np.array_equal(loader.l1normalize(x, axis=1),
                          np.array([[0.0, 0.0, 1.0],
                                    [.25, .5, .25],
                                    [.25, .75, 0.0]]))


def test_l1_sparse_test_axis1():
    assert np.array_equal(loader.l1normalize(y, axis=1),
                          np.array([[0.0, 0.0, 1.0],
                                    [.25, .5, .25],
                                    [.25, .75, 0.0]]))


# def test_l2_dense_test_axis0():
#     assert np.testing.assert_array_almost_equal(loader.l2normalize(x, axis=0),
#                           np.array([[0.0, 0.0, 3.0 / np.sqrt(10.0)],
#                            [1.0 / np.sqrt(2.0), 2.0 / np.sqrt(13.0), 1.0 / np.sqrt(10.0)],
#                            [1.0 / np.sqrt(2.0), 3.0 / np.sqrt(13.0), 0.0]]), decimal=5)
#
#
# def test_l2_sparse_test_axis0():
#     assert np.testing.assert_array_almost_equal(loader.l2normalize(x, axis=0),
#                                                 np.array([[0.0, 0.0, 3.0 / np.sqrt(10.0)],
#                                                           [1.0 / np.sqrt(2.0), 2.0 / np.sqrt(13.0),
#                                                            1.0 / np.sqrt(10.0)],
#                                                           [1.0 / np.sqrt(2.0), 3.0 / np.sqrt(13.0), 0.0]]), decimal=5)


def test_max_dense_test_axis0():
    assert np.array_equal(loader.maxnormalize(x, axis=0),
                          [[0.0, 0.0, 1.0],
                           [1.0, 2.0/3.0, 1.0/3.0],
                           [1.0, 1.0, 0.0]])



def test_max_sparse_test_axis0():
    assert np.array_equal(loader.maxnormalize(y, axis=0),
                          [[0.0, 0.0, 1.0],
                           [1.0, 2.0 / 3.0, 1.0 / 3.0],
                           [1.0, 1.0, 0.0]])


def test_max_dense_test_axis1():
    assert np.array_equal(loader.maxnormalize(x, axis=1),
                          [[0.0, 0.0, 1.0],
                           [.5, 1, .5],
                           [1.0/3.0, 1.0, 0.0]])


def test_max_sparse_test_axis1():
    assert np.array_equal(loader.maxnormalize(y, axis=1),
                          [[0.0, 0.0, 1.0],
                           [.5, 1, .5],
                           [1.0 / 3.0, 1.0, 0.0]])


def test_center_dense_test():
    np.testing.assert_array_almost_equal(loader.center(x, axis=None).mean(axis=None), 0.0)


def test_center_dense_test_axis0():
    np.testing.assert_array_almost_equal(np.sum(loader.center(x, axis=0).mean(axis=0)), 0.0)


def test_center_dense_test_axis1():
    np.testing.assert_array_almost_equal(np.sum(loader.center(x, axis=1).mean(axis=1)), 0.0)

def test_center_sparse_test():
    np.testing.assert_array_almost_equal(np.sum(loader.center(y, axis=None).mean(axis=None)), 0.0)


def test_center_sparse_test_axis0():
    np.testing.assert_array_almost_equal(np.sum(loader.center(y, axis=0).mean(axis=0)), 0.0)


def test_center_sparse_test_axis1():
    np.testing.assert_array_almost_equal(np.sum(loader.center(y, axis=1).mean(axis=1)), 0.0)


def test_unit_variance_dense_test():
    np.testing.assert_array_almost_equal(loader.unit_variance(x, axis=None).std(axis=None), 1.0)


def test_unit_variance_sparse_test_axis0():
    np.testing.assert_array_almost_equal(np.sum(loader.unit_variance(y, axis=0).std(axis=0)), 3.0)

def test_unit_variance_sparse_test_axis1():
    np.testing.assert_array_almost_equal(np.sum(loader.unit_variance(y, axis=1).std(axis=1)), 3.0)





