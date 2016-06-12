from __future__ import print_function
import tensorflow as tf
from antk.core import config
from antk.core import generic_model

def tree(data, configfile, lamb=0.001,
            kfactors=20,
            learnrate=0.0001,
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

    y = ant.tensor_out
    y_ = tf.placeholder("float", [None, None], name='Target')
    ant.placeholderdict['ratings'] = y_  # put the new placeholder in the graph for training
    with tf.name_scope('objective'):

        objective = (tf.reduce_sum(tf.square(y_ - y)) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['huser'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['hitem'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['ubias'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['ibias'])))
    with tf.name_scope('dev_rmse'):
        dev_rmse = tf.sqrt(tf.div(tf.reduce_sum(tf.square(y - y_)), data.dev.num_examples))

    with tf.name_scope('training'):
        model = generic_model.Model(objective, ant.placeholderdict,
                                    mb=mb,
                                    learnrate=learnrate,
                                    verbose=verbose,
                                    maxbadcount=maxbadcount,
                                    epochs=epochs,
                                    evaluate=dev_rmse,
                                    predictions=y,
                                    model_name='tree')
        model.train(data.train, dev=data.dev, supplement=datadict, eval_schedule=eval_rate)

    return model