from __future__ import print_function
import tensorflow as tf
from antk.core import config
from antk.core import generic_model
from antk.core import node_ops
from antk.core import loader


def mf(data, configfile, lamb=0.001,
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


    with tf.name_scope('ant_graph'):
        ant = config.AntGraph(configfile,
                              data=data.dev.features,
                              marker='-',
                              graph_name='basic_mf',
                              develop=develop,
                              variable_bindings={'kfactors': kfactors,
                                                 'initrange': initrange})
        y = ant.tensor_out
        y_ = tf.placeholder("float", [None, None], name='Target')
        ant.placeholderdict['ratings'] = y_
        with tf.name_scope('objective'):
            objective = (tf.reduce_sum(tf.square(y_ - y)))
        objective += (lamb*tf.reduce_sum(tf.square(ant.tensordict['huser'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['hitem'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['ubias'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['ibias'])))
        with tf.name_scope('dev_rmse'):
            dev_rmse = node_ops.rmse(y_, y)
        model = generic_model.Model(objective, ant.placeholderdict,
                                    mb=mb,
                                    learnrate=learnrate,
                                    verbose=verbose,
                                    maxbadcount=maxbadcount,
                                    epochs=epochs,
                                    evaluate=dev_rmse,
                                    predictions=y,
                                    model_name='mf',
                                    random_seed=random_seed)
        model.train(data.train, dev=data.dev, eval_schedule=eval_rate)

        return model
