from antk.core import loader
import numpy as np
import scipy.sparse as sps

def test_is_one_hot_true_sparse():
    w = sps.csr_matrix(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    assert loader.is_one_hot(w)

def test_is_one_hot_true_dense():
    w = np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]])
    assert loader.is_one_hot(w)

def test_is_one_hot_false1_sparse():
    w = sps.csr_matrix(np.array([[5,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    assert not loader.is_one_hot(w)

def test_is_one_hot_false1_dense():
    w = np.array([[5,0,0], [1,0,0], [0,1,0], [0,0,1]])
    assert not loader.is_one_hot(w)

def test_is_one_hot_false2_sparse():
    w = sps.csr_matrix(np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    assert not loader.is_one_hot(w)

def test_is_one_hot_false2_dense():
    w = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
    assert not loader.is_one_hot(w)

def test_is_one_hot_false3_dense():
    w = np.array([0,0,1])
    assert not loader.is_one_hot(w)

def test_to_index_to_hot_sparse():
    w = sps.csr_matrix(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    assert np.array_equal(w.toarray(), (loader.toOnehot(loader.toIndex(w), dim=3).toarray()))

def test_to_index_to_hot_dense():
    w = np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]])
    assert np.array_equal(w, (loader.toOnehot(loader.toIndex(w), dim=3).toarray()))

def test_to_hot_to_index():
    w = np.array([1,2,3,0,3,10])
    assert np.array_equal(w, loader.toIndex(loader.toOnehot(w, dim=11)))
def test_vec():
    x = loader.HotIndex(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    assert np.array_equal(x.vec, np.array([0,0,1,2]))

def test_dim():
    x = loader.HotIndex(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    assert x.dim == 3

def test_eq():
    x = loader.HotIndex(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    y = loader.HotIndex(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    assert x == y

def test_HotIndex_slicing_left_index():
    x = loader.HotIndex(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    assert x[0:] == x

def test_HotIndex_slicing_right_index():
    x = loader.HotIndex(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    assert x[:4] == x

def test_HotIndex_slicing_two_indices():
    x = loader.HotIndex(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    assert x[1:3] == loader.HotIndex(np.array([[1,0,0], [0,1,0]]))

def test_HotIndex_step_slicing():
    x = loader.HotIndex(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    assert x[0:4:2] == loader.HotIndex(np.array([[1,0,0], [0,1,0]]))

def test_HotIndex_indexing():
    x = loader.HotIndex(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    assert x[3] == 2

def test_HotIndex_to_one_hot():
    w = np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]])
    x = loader.HotIndex(sps.csr_matrix(w))
    assert np.array_equal(w, x.hot().toarray())

def test_init_from_vec():
    w = loader.HotIndex(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    x = loader.HotIndex(np.array([0,0,1,2]))
    assert x == w

def test_HotIndex_iteration():
    w = loader.HotIndex(np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1]]))
    indices = [0,0,1,2]
    for ind, item in enumerate(w):
        assert indices[ind] == item


