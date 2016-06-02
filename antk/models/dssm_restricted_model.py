from __future__ import print_function
import tensorflow as tf
from antk.core import config
from antk.core import generic_model
from antk.core import node_ops

def dssm(data, configfile,
         layers=[10,10,10],
         bn=True,
         keep_prob=.95,
         act='tanhlecun',
         initrange=1,
         kfactors=10,
         lamb=.1,
         mb=500,
         learnrate=0.0001,
         verbose=True,
         maxbadcount=10,
         epochs=100,
         model_name='dssm',
         random_seed=500,
         eval_rate=500):

    datadict = data.user.features.copy()
    datadict.update(data.item.features)

    configdatadict = data.dev.features.copy()
    configdatadict.update(datadict)
    with tf.name_scope('ant_graph'):
        ant = config.AntGraph(configfile,
                              data=configdatadict,
                              marker='-',
                              graph_name='basic_mf',
                              variable_bindings={'layers': layers,
                                                 'bn': bn,
                                                 'keep_prob': keep_prob,
                                                 'act': act,
                                                 'initrange': initrange,
                                                 'kfactors': kfactors,
                                                 })
    y_ = tf.placeholder("float", [None, None], name='Target')
    ant.placeholderdict['ratings'] = y_
    dotproducts = node_ops.x_dot_y(ant.tensor_out)
    dotproducts = []
    with tf.variable_scope('dotproducts'):
        for j in range(len(ant.tensor_out[1])):
            with tf.name_scope('item%d' % j):
                dot = node_ops.x_dot_y([ant.tensor_out[0][0], ant.tensor_out[1][j], ant.tensor_out[2], ant.tensor_out[3]])
                dotproducts.append(dot)
        for i in range(len(ant.tensor_out[0])):
            with tf.name_scope('user%d' % i):
                dot = node_ops.x_dot_y([ant.tensor_out[0][i], ant.tensor_out[1][0], ant.tensor_out[2], ant.tensor_out[3]])
                dotproducts.append(dot)
    with tf.name_scope('objective'):
        if type(dotproducts) is list:
            objective = tf.reduce_sum(tf.square(y_ - dotproducts[0]))
        for i in range(1, len(dotproducts)):
            objective += tf.reduce_sum(tf.square(y_ - dotproducts[i]))
        objective += (lamb*tf.reduce_sum(tf.square(ant.tensordict['huser'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['hitem'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['ubias'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['ibias'])))
    with tf.name_scope('dev_rmse'):
        dev_rmse = node_ops.rmse(dotproducts[0], y_)

    model = generic_model.Model(objective, ant.placeholderdict,
                                mb=mb,
                                learnrate=learnrate,
                                verbose=verbose,
                                maxbadcount=maxbadcount,
                                epochs=100,
                                evaluate=dev_rmse,
                                predictions=dotproducts[0],
                                model_name='dssm',
                                random_seed=500)

    model.train(data.train, dev=data.dev, supplement=datadict, eval_schedule=eval_rate)
    return model