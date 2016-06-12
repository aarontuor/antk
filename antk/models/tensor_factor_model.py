from __future__ import print_function
import tensorflow as tf
from antk.core import config
from antk.core import generic_model
from antk.core import node_ops
from antk.core import loader


def tensorfactor(data,  lamb=0.01,
            learnrate=0.0001,
            verbose=True,
            epochs=100,
            maxbadcount=20,
            mb=500,
            initrange=0.0001,
            eval_rate=10000,
            random_seed=None,
            uembed=50,
            iembed=50,
            gembed=50):

        data = loader.read_data_sets(data, folders=('train', 'dev', 'item'),
                                     hashlist=('user', 'item', 'genres', 'ratings'))
        data.train.labels['ratings'] = loader.center(data.train.labels['ratings'])
        data.dev.labels['ratings'] = loader.center(data.dev.labels['ratings'])

        data.train.features['genre'] = data.item.features['genres'][data.train.features['item'].vec, :]
        data.dev.features['genre'] = data.item.features['genres'][data.dev.features['item'].vec, :]

        data.show()


        item = tf.placeholder(tf.int32, [None])
        user = tf.placeholder(tf.int32, [None])
        genre = tf.placeholder(tf.float32, [None, data.dev.features['genre'].shape[1]])

        wuser = initrange*tf.Variable(tf.truncated_normal([data.dev.features['user'].shape[1], uembed]))
        witem = initrange*tf.Variable(tf.truncated_normal([data.dev.features['item'].shape[1], iembed]))
        wgenre = initrange*tf.Variable(tf.truncated_normal([data.dev.features['genre'].shape[1], gembed]))

        xuser = tf.nn.embedding_lookup(wuser, user)
        xitem = tf.nn.embedding_lookup(witem, item)
        xgenre = tf.matmul(genre, wgenre, a_is_sparse=True)

        ibias = tf.Variable(tf.truncated_normal([data.dev.features['item'].shape[1]]))
        ubias = tf.Variable(tf.truncated_normal([data.dev.features['user'].shape[1]]))
        gbias = tf.Variable(tf.truncated_normal([data.dev.features['genre'].shape[1], 1]))

        i_bias = tf.nn.embedding_lookup(ibias, item)
        u_bias = tf.nn.embedding_lookup(ubias, user)
        g_bias = tf.matmul(genre, gbias, a_is_sparse=True)

        y = node_ops.ternary_tensor_combine([xuser, xitem, xgenre],
                                            initrange=initrange,
                                            l2=lamb) + i_bias + u_bias
        y_ = tf.placeholder("float", [None, None], name='Target')

        placeholderdict = {'user': user, 'item': item, 'genre': genre, 'ratings': y_}
        with tf.name_scope('objective'):
            objective = (tf.reduce_sum(tf.square(y_ - y)) +
                         lamb*tf.reduce_sum(tf.square(wgenre)) +
                         lamb*tf.reduce_sum(tf.square(xuser)) +
                         lamb*tf.reduce_sum(tf.square(xitem)) +
                         lamb*tf.reduce_sum(tf.square(i_bias)) +
                         lamb*tf.reduce_sum(tf.square(u_bias)))
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
                                    model_name='tensorfactor',
                                    random_seed=random_seed)
        model.train(data.train, dev=data.dev, eval_schedule=eval_rate)

        return model