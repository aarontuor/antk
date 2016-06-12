from __future__ import print_function
import tensorflow as tf
from antk.core import config
from antk.core import generic_model
from antk.core import node_ops
from antk.core import loader
import scipy.sparse as sps
import numpy
def svdplus(data, lamb_bias=0.005, lambfactor=0.015,
            kfactors=20,
            learnrate=0.01,
            verbose=True,
            epochs=1000,
            maxbadcount=20,
            mb=500,
            initrange=1,
            eval_rate=500,
            random_seed=None,
            develop=False):


        data = loader.read_data_sets(data, folders=['train', 'dev', 'item'], hashlist=['user', 'item', 'ratings'])
        data.train.labels['ratings'] = loader.center(data.train.labels['ratings'])
        data.dev.labels['ratings'] = loader.center(data.dev.labels['ratings'])
        utility_matrix = sps.csr_matrix((numpy.ones(data.train.features['user'].vec.shape[0]),
                                         (data.train.features['user'].vec, data.train.features['item'].vec)),
                                        shape=(data.train.features['user'].dim, data.train.features['item'].dim))
        data.item.features['util'] = utility_matrix

        xuser = tf.placeholder(tf.int32, [None])
        xitem = tf.placeholder(tf.int32, [None])

        xutil = tf.placeholder(tf.float32, [None, None])

        wuser = initrange*tf.Variable(tf.truncated_normal([data.dev.features['user'].dim, kfactors]))
        witem = initrange*tf.Variable(tf.truncated_normal([data.dev.features['item'].dim, kfactors]))
        wplus = initrange*tf.Variable(tf.truncated_normal([data.dev.features['item'].dim, kfactors]))

        ubias = initrange*tf.Variable(tf.truncated_normal([data.dev.features['user'].dim]))
        ibias = initrange*tf.Variable(tf.truncated_normal([data.dev.features['item'].dim]))

        i_bias = tf.nn.embedding_lookup(ibias, xitem)
        u_bias = tf.nn.embedding_lookup(ubias, xuser)

        huser = tf.nn.embedding_lookup(wuser, xuser)
        hitem = tf.nn.embedding_lookup(witem, xitem)
        hplus = tf.nn.embedding_lookup(xutil, xuser)

        plus = tf.mul(tf.matmul(hplus, wplus, a_is_sparse=True), tf.rsqrt(tf.reduce_sum(hplus, reduction_indices=1, keep_dims=True)))
        huserplus = huser + plus

        y = node_ops.x_dot_y([huserplus, hitem, i_bias, u_bias])
        y_ = tf.placeholder("float", [None, None], name='Target')

        with tf.name_scope('objective'):
            objective = (tf.reduce_sum(tf.square(y_ - y)) +
                         lambfactor*tf.reduce_sum(tf.square(huser)) +
                         lambfactor*tf.reduce_sum(tf.square(hitem)) +
                         lambfactor*tf.reduce_sum(tf.square(wplus)) +
                         lamb_bias*tf.reduce_sum(tf.square(i_bias)) +
                         lamb_bias*tf.reduce_sum(tf.square(u_bias)))

        placeholderdict = {'ratings': y_, 'util': xutil, 'user': xuser, 'item': xitem}

        with tf.name_scope('dev_rmse'):
            dev_rmse = node_ops.rmse(y_, y)
        model = generic_model.Model(objective, placeholderdict,
                                    mb=mb,
                                    learnrate=learnrate,
                                    verbose=verbose,
                                    maxbadcount=maxbadcount,
                                    epochs=epochs,
                                    evaluate=dev_rmse,
                                    predictions=y,
                                    model_name='svdplus',
                                    random_seed=random_seed,
                                    decay=(500, 0.999))
        model.train(data.train, dev=data.dev, supplement=data.item.features, eval_schedule=eval_rate)

        return model
