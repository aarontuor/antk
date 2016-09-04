from antk.core.loader import import_data, export_data
import numpy as np
import scipy.sparse as sps

def test_import_values_dense():
    """
    Test values after saving and loading of .dense format.
    """
    x = np.random.rand(7, 11)
    export_data('/tmp/test.dense', x)
    assert np.array_equal(x, import_data('/tmp/test.dense'))

def test_import_type_dense():
    """
    Test the type after saving and loading of .dense format.
    """
    x = np.random.rand(7, 11)
    export_data('/tmp/test.dense', x)
    assert x.dtype == import_data('/tmp/test.dense').dtype

def test_import_values_sparse():
    """
    Test values after saving and loading of .sparse format.
    """
    x = sps.csr_matrix(np.random.rand(7, 11))
    export_data('/tmp/test.sparse', x)
    assert np.array_equal(x.toarray(), import_data('/tmp/test.sparse').toarray())

def test_import_type_sparse():
    """4
    Test the type after saving and loading of .sparset1 format.
    """
    x = sps.csr_matrix(np.random.rand(7, 11))
    export_data('/tmp/test.sparse', x)
    assert x.dtype == import_data('/tmp/test.sparse').dtype

def test_import_values_densetxt():
    """
    Test values after saving and loading of .dense format.
    """
    x = np.random.rand(7, 11)
    export_data('/tmp/test.densetxt', x)
    assert np.array_equal(x, import_data('/tmp/test.densetxt'))

def test_import_type_densetxt():
    """
    Test the type after saving and loading of .dense format.
    """
    x = np.random.rand(7, 11)
    export_data('/tmp/test.densetxt', x)
    assert x.dtype == import_data('/tmp/test.densetxt').dtype

def test_import_values_sparsetxt():
    """
    Test values after saving and loading of .sparse format.
    """
    x = sps.csr_matrix(np.random.rand(3, 2))
    export_data('/tmp/test.sparsetxt', x)
    assert np.array_equal(x.toarray(), import_data('/tmp/test.sparsetxt').toarray())

def test_import_type_sparsetxt():
    """4
    Test the type after saving and loading of .sparset1 format.
    """
    x = sps.csr_matrix(np.random.rand(3, 2))
    export_data('/tmp/test.sparsetxt', x)
    assert x.dtype == import_data('/tmp/test.sparsetxt').dtype

def test_import_sparse_values_mat():
    """
    Test values after saving and loading of .sparse format.
    """
    x = sps.csr_matrix(np.random.rand(3, 2))
    export_data('/tmp/test.mat', x)
    assert np.array_equal(x.toarray(), import_data('/tmp/test.mat').toarray())

def test_import_sparse_type_mat():
    """4
    Test the type after saving and loading of .sparset1 format.
    """
    x = sps.csr_matrix(np.random.rand(3, 2))
    export_data('/tmp/test.mat', x)
    assert x.dtype == import_data('/tmp/test.mat').dtype

def test_import_dense_values_mat():
    """
    Test values after saving and loading of .sparse format.
    """
    x = np.random.rand(3, 2)
    export_data('/tmp/test.mat', x)
    assert np.array_equal(x, import_data('/tmp/test.mat'))

def test_import_dense_type_mat():
    """4
    Test the type after saving and loading of .sparset1 format.
    """
    x = np.random.rand(3, 2)
    export_data('/tmp/test.mat', x)
    assert x.dtype == import_data('/tmp/test.mat').dtype



