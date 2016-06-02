from __future__ import print_function
import tensorflow as tf
from antk.core import config
from antk.core import generic_model

def dsadd(data, configfile,
         initrange=0.1,
         kfactors=20,
         lamb = .01,
         mb=500,
         learnrate=0.003,
         verbose=True,
         maxbadcount=10,
         epochs=100,
         model_name='dssm',
         random_seed=500,
         eval_rate = 500):

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
            scalars = tf.Variable(0.001*tf.truncated_normal([len(ant.tensor_out), 1]))
            prediction = tf.mul(ant.tensor_out[0], tf.slice(scalars, [0,0], [1, 1]))
            for i in range(1, len(ant.tensor_out)):
                with tf.variable_scope('predictor%d' %i):
                    prediction = prediction + tf.mul(ant.tensor_out[i], tf.slice(scalars, [i, 0], [1, 1]))
            prediction = tf.square(y_ - prediction)

        objective = (tf.reduce_sum(prediction) + lamb*tf.reduce_sum(tf.square(ant.tensordict['huser'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['hitem'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['ubias'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['ibias'])) +
                     lamb*tf.reduce_sum(tf.square(tf.concat(1, ant.tensor_out))))
    with tf.name_scope('dev_rmse'):
        dev_rmse = tf.sqrt(tf.div(tf.reduce_sum(prediction), data.dev.num_examples))

    model = generic_model.Model(objective, ant.placeholderdict,
                                mb=500,
                                learnrate=0.000001,
                                verbose=True,
                                maxbadcount=10,
                                epochs=100,
                                evaluate=dev_rmse,
                                predictions=ant.tensor_out[0],
                                model_name='dssm',
                                random_seed=500)

    model.train(data.train, dev=data.dev, supplement=datadict, eval_schedule = eval_rate)
    return model