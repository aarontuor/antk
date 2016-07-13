from __future__ import print_function
import tensorflow as tf
from antk.core import config
from antk.core import generic_model
from antk.core import node_ops

def tree(data, configfile, lamb=0.001,
            kfactors=50,
            learnrate=0.00001,
            verbose=True,
            maxbadcount=20,
            mb=500,
            initrange=0.00001,
            epochs=10,
            random_seed=None,
            eval_rate=500,
            keep_prob=0.95,
            act='tanh'):

    datadict = data.user.features.copy()
    datadict.update(data.item.features)

    configdatadict = data.dev.features.copy()
    configdatadict.update(datadict)

    with tf.name_scope('ant_graph'):
        ant = config.AntGraph(configfile,
                                data=configdatadict,
                                marker='-',
                                variable_bindings = {'kfactors': kfactors, 'initrange': initrange, 'keep_prob':
                                    keep_prob, 'act': act},
                                graph_name='tree')
    # ant.display_graph()
    alpha = node_ops.weights('tnorm', [1,1], l2=1.0)
    beta = node_ops.weights('tnorm', [1,1], l2=1.0)
    print(ant.tensor_out)
    ubias = initrange*tf.Variable(tf.truncated_normal([data.dev.features['user'].dim]))
    ibias = initrange*tf.Variable(tf.truncated_normal([data.dev.features['item'].dim]))

    i_bias = tf.nn.embedding_lookup(ibias, ant.placeholderdict['item'])
    u_bias = tf.nn.embedding_lookup(ubias, ant.placeholderdict['user'])
    y = alpha*ant.tensor_out[0] + beta*ant.tensor_out[1] + u_bias + i_bias
    y_ = tf.placeholder("float", [None, None], name='Target')
    ant.placeholderdict['ratings'] = y_  # put the new placeholder in the graph for training
    with tf.name_scope('objective'):

        objective = (tf.reduce_sum(tf.square(y_ - y)) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['huser'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['hitem'])) +
                    lamb*tf.reduce_sum(tf.square(ant.tensordict['huser2'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['hitem2'])) +
                     lamb*tf.reduce_sum(tf.square(u_bias)) +
                     lamb*tf.reduce_sum(tf.square(i_bias)))
    with tf.name_scope('dev_rmse'):
        dev_rmse = node_ops.rmse(y_, y)

    with tf.name_scope('training'):
        model = generic_model.Model(objective, ant.placeholderdict,
                                    mb=mb,
                                    learnrate=learnrate,
                                    verbose=verbose,
                                    maxbadcount=maxbadcount,
                                    epochs=epochs,
                                    evaluate=dev_rmse,
                                    predictions=y,
                                    model_name='tree',
                                    train_evaluate=dev_rmse)
        model.train(data.train, dev=data.dev, supplement=datadict, eval_schedule=eval_rate, train_dev_eval_factor=5)
    return model