from antk.core.loader import DataSet
import numpy as np
import scipy.linalg

def get_train_x():
    return DataSet({'toe': scipy.linalg.toeplitz([1, 0, 0, 0], [1, 2, 3, 0, 0, 0]), #4X6
                 'id': np.eye(4)}) #4X4

def test_init1():
    x = get_train_x()
    assert np.array_equal(x.features['toe'], scipy.linalg.toeplitz([1, 0, 0, 0], [1, 2, 3, 0, 0, 0]) )
    assert np.array_equal(x.features['id'], np.eye(4))

def test_num_examples1():
    x = get_train_x()
    assert x.num_examples == 4

def test_index_in_epoch_start():
    x = get_train_x()
    assert x.index_in_epoch == 0

def get_x():
    return DataSet({'toe': scipy.linalg.toeplitz([1, 0, 0, 0], [1, 2, 3, 0, 0, 0]), #4X6
             'id': np.eye(4)},
            labels={'toe': scipy.linalg.toeplitz([1, 0, 0, 0], [1, 2, 3, 0, 0, 0]), #4X6
                    'id': np.eye(4)}, num_examples=4) #4X4

def get_y():
    return DataSet({'toe': scipy.linalg.toeplitz([1, 0, 0, 0], [1, 2, 3, 0, 0, 0]), #4X6
             'id': np.eye(4)},
            labels={'toe': scipy.linalg.toeplitz([1, 0, 0, 0], [1, 2, 3, 0, 0, 0]), #4X6
                    'id': np.eye(4)}, num_examples=4, mix=True) #4X4

def test_init1():
    x = get_x()
    assert np.array_equal(x.features['toe'], x.labels['toe'])
    assert np.array_equal(x.features['id'], x.labels['id'])

def test_num_examples1():
    x = get_x()
    assert x.num_examples == 4

def test_next_mixed_mini_batch_shape1():
    y = get_y()
    n = y.next_batch(3)
    assert n.features['toe'].shape == (3,6)
    assert n.labels['id'].shape == (3,4)

def test_next_mixed_mini_batch1():
    y = get_y()
    x = get_x()
    n = y.next_batch(3)
    assert np.array_equal(n.features['toe'], x.labels['toe'][0:3])
    assert np.array_equal(n.labels['id'], x.features['id'][0:3])

def test_next_mixed_mini_batch_index1():
    y = get_y()
    y.next_batch(3)
    assert y.index_in_epoch == 3

def test_next_mixed_mini_batch_shape2():
    y = get_y()
    y.next_batch(3)
    n = y.next_batch(3)
    assert n.features['toe'].shape == (3,6)
    assert n.labels['id'].shape == (3,4)

def test_next_mixed_mini_batch2():
    toe_equals = True
    id_equals = True
    for i in xrange(20):
        y = get_y()
        y.next_batch(3)
        n = y.next_batch(3)
        x = get_x()
        if not np.array_equal(n.features['toe'], x.labels['toe'][0:3]):
            toe_equals = False
        if not np.array_equal(n.labels['id'], x.features['id'][0:3]):
            id_equals = False
    assert not toe_equals
    assert not id_equals
    assert np.array_equal(n.features['toe'], y.labels['toe'][0:3])
    assert np.array_equal(n.labels['id'], y.features['id'][0:3])

def test_next_mixed_mini_batch_index2():
    y = get_y()
    y.next_batch(3)
    y.next_batch(3)
    assert y.index_in_epoch == 3

def test_next_unmixed_mini_batch_shape1():
    x = get_x()
    n = x.next_batch(3)
    assert n.features['toe'].shape == (3,6)
    assert n.labels['id'].shape == (3,4)

def test_next_unmixed_mini_batch1():
    x = get_x()
    n = x.next_batch(3)
    assert np.array_equal(n.features['toe'], x.labels['toe'][0:3])
    assert np.array_equal(n.labels['id'], x.features['id'][0:3])

def test_next_unmixed_mini_batch_index1():
    x = get_x()
    x.next_batch(3)
    assert x.index_in_epoch == 3

def test_next_unmixed_mini_batch_shape2():
    x = get_x()
    x.next_batch(3)
    n = x.next_batch(3)
    assert n.features['toe'].shape == (3,6)
    assert n.labels['id'].shape == (3,4)

def test_next_unmixed_mini_batch2():
    x = get_x()
    x.next_batch(3)
    n = x.next_batch(3)
    assert np.array_equal(n.features['toe'], x.labels['toe'][[3,0,1]])
    assert np.array_equal(n.labels['id'], x.features['id'][[3,0,1]])

def test_next_unmixed_mini_batch_index2():
    x = get_x()
    x.next_batch(3)
    x.next_batch(3)
    assert x.index_in_epoch == 2

def test_next_batch_unmixed_whole_batch():
    x = get_x()
    n = x.next_batch(4)
    assert np.array_equal(n.features['toe'], x.labels['toe'])
    assert np.array_equal(n.labels['id'], x.features['id'])
    assert x.index_in_epoch == 0

def test_next_batch_unmixed_whole_batch2():
    x = get_x()
    n = x.next_batch(4)
    m = x.next_batch(4)
    assert np.array_equal(n.features['toe'], m.labels['toe'])
    assert np.array_equal(n.labels['id'], m.features['id'])
    assert x.index_in_epoch == 0

def test_next_batch_mixed_whole_batch():
    x = get_x()
    n = x.next_batch(4)
    assert np.array_equal(n.features['toe'], x.labels['toe'])
    assert np.array_equal(n.labels['id'], x.features['id'])
    assert x.index_in_epoch == 0

def test_next_batch_mixed_whole_batch2():
    y = get_y()
    n = y.next_batch(4)
    m = y.next_batch(4)
    assert not np.array_equal(n.features['toe'], m.labels['toe'])
    assert not np.array_equal(n.labels['id'], m.features['id'])
    assert y.index_in_epoch == 0

def test_next_mixed_mini_batch_shape2_divides_batch():
    y = get_y()
    y.next_batch(2)
    n = y.next_batch(2)
    assert n.features['toe'].shape == (2,6)
    assert n.labels['id'].shape == (2,4)

def test_next_mixed_mini_batch2_divides_batch():
    toe_equals = True
    id_equals = True
    for i in range(20):
        y = get_y()
        x = get_x()
        y.next_batch(2)
        m = y.next_batch(2)
        n = y.next_batch(2)
        if not np.array_equal(y.features['toe'][0:2], x.labels['toe'][0:2]):
            toe_equals = False
        if not np.array_equal(y.labels['id'][0:2], x.features['id'][0:2]):
            id_equals = False
    assert not toe_equals
    assert not id_equals
    assert np.array_equal(m.features['toe'], y.labels['toe'][2:4])
    assert np.array_equal(m.labels['id'], y.features['id'][2:4])

def test_next_mixed_mini_batch_index2_divides_batch():
    y = get_y()
    y.next_batch(2)
    y.next_batch(2)
    assert y.index_in_epoch == 0

def test_next_unmixed_mini_batch_shape2_divides_batch():
    y = get_x()
    y.next_batch(2)
    n = y.next_batch(2)
    assert n.features['toe'].shape == (2,6)
    assert n.labels['id'].shape == (2,4)

def test_next_unmixed_mini_batch2_divides_batch():
    y = get_x()
    y.next_batch(2)
    m = y.next_batch(2)
    n = y.next_batch(2)
    assert np.array_equal(m.features['toe'], y.labels['toe'][2:4])
    assert np.array_equal(m.labels['id'], y.features['id'][2:4])
    assert np.array_equal(n.features['toe'], y.labels['toe'][0:2])
    assert np.array_equal(n.labels['id'], y.features['id'][0:2])

def test_next_unmixed_mini_batch_index2_divides_batch():
    y = get_x()
    y.next_batch(2)
    y.next_batch(2)
    assert y.index_in_epoch == 0









