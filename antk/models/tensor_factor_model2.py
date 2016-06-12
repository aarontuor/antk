from __future__ import print_function
import tensorflow as tf
from antk.core import config
from antk.core import generic_model
from antk.core import node_ops
from antk.core import loader


def tensorfactor(data,  context_key='occ', lamb=0.01,
            learnrate=0.0001,
            verbose=True,
            epochs=5,
            maxbadcount=20,
            mb=500,
            initrange=0.0001,
            eval_rate=10000,
            random_seed=None,
            uembed=50,
            iembed=50,
            cembed=50):

        data = loader.read_data_sets(data, folders=('train', 'dev', 'item', 'user'),
                                     hashlist=('user', 'item', context_key, 'ratings'))
        data.train.labels['ratings'] = loader.center(data.train.labels['ratings'])
        data.dev.labels['ratings'] = loader.center(data.dev.labels['ratings'])

        if context_key in data.item.features:
            data.train.features[context_key] = data.item.features[context_key][data.train.features['item']]
            data.dev.features[context_key] = data.item.features[context_key][data.dev.features['item']]
            del data.item.features[context_key]
        elif context_key in data.user.features:
            data.train.features[context_key] = data.user.features[context_key][data.train.features['user']]
            data.dev.features[context_key] = data.user.features[context_key][data.dev.features['user']]
            del data.user.features[context_key]
        data.show()


        item = tf.placeholder(tf.int32, [None])
        user = tf.placeholder(tf.int32, [None])
        context = tf.placeholder(tf.int32, [None])

        wuser = initrange*tf.Variable(tf.truncated_normal([data.dev.features['user'].shape[1], uembed]))
        witem = initrange*tf.Variable(tf.truncated_normal([data.dev.features['item'].shape[1], iembed]))
        wcontext = initrange*tf.Variable(tf.truncated_normal([data.dev.features[context_key].shape[1], cembed]))

        xuser = tf.nn.embedding_lookup(wuser, user)
        xitem = tf.nn.embedding_lookup(witem, item)
        xcontext = tf.nn.embedding_lookup(wcontext, context)

        ibias = tf.Variable(tf.truncated_normal([data.dev.features['item'].shape[1]]))
        ubias = tf.Variable(tf.truncated_normal([data.dev.features['user'].shape[1]]))
        cbias = tf.Variable(tf.truncated_normal([data.dev.features[context_key].shape[1]]))

        i_bias = tf.nn.embedding_lookup(ibias, item)
        u_bias = tf.nn.embedding_lookup(ubias, user)
        c_bias = tf.nn.embedding_lookup(cbias, context)

        y = node_ops.ternary_tensor_combine([xuser, xitem, xcontext],
                                            initrange=initrange,
                                            l2=lamb) + i_bias + u_bias
        y_ = tf.placeholder("float", [None, None], name='Target')

        placeholderdict = {'user': user, 'item': item, context_key: context, 'ratings': y_}
        with tf.name_scope('objective'):
            objective = (tf.reduce_sum(tf.square(y_ - y)) +
                         lamb*tf.reduce_sum(tf.square(wcontext)) +
                         lamb*tf.reduce_sum(tf.square(xuser)) +
                         lamb*tf.reduce_sum(tf.square(xitem)) +
                         lamb*tf.reduce_sum(tf.square(i_bias)) +
                         lamb*tf.reduce_sum(tf.square(u_bias)) +
                         lamb*tf.reduce_sum(tf.square(c_bias)))
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