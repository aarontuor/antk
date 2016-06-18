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
                              variable_bindings={'initrange': initrange, 'kfactors': kfactors})
    y_ = tf.placeholder("float", [None, None], name='Target')
    ant.placeholderdict['ratings'] = y_
    with tf.name_scope('objective'):
        if type(ant.tensor_out) is list:
            objective = tf.reduce_sum(tf.square(y_ - ant.tensor_out[0]))
        for i in range(1, len(ant.tensor_out)):
            objective += tf.reduce_sum(tf.square(y_ - ant.tensor_out[i]))
        objective += (lamb*tf.reduce_sum(tf.square(ant.tensordict['huser'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['hitem'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['ubias'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['ibias'])))
    with tf.name_scope('dev_rmse'):
        dev_rmse = node_ops.rmse(ant.tensor_out[0], y_)

    model = generic_model.Model(objective, ant.placeholderdict,
                                mb=mb,
                                learnrate=learnrate,
                                verbose=verbose,
                                maxbadcount=maxbadcount,
                                epochs=epochs,
                                evaluate=dev_rmse,
                                predictions=ant.tensor_out[0],
                                model_name='dssm',
                                random_seed=random_seed)

    model.train(data.train, dev=data.dev, supplement=datadict, eval_schedule=eval_rate)
    return model