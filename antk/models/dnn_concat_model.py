import tensorflow as tf
from antk.core import config
from antk.core import generic_model

def dnn_concat(data, configfile,
            layers=[16, 8, 8],
            activation='tanhlecun',
            initrange=1e-3,
            bn=True,
            keep_prob=.95,
            concat_size=24,
            uembed=32,
            iembed=32,
            learnrate=.00001,
            verbose=True,
            epochs=10,
            maxbadcount=20,
            mb=2000,
            eval_rate=500):

    with tf.name_scope('ant_graph'):
        ant = config.AntGraph(configfile,
                                data=data.dev.features,
                                marker='-',
                                graph_name='dnn_concat',
                                variable_bindings={'layers': layers,
                                                   'activation': activation,
                                                   'initrange': initrange,
                                                   'bn': bn,
                                                   'keep_prob': keep_prob,
                                                   'concat_size': concat_size,
                                                   'uembed': uembed,
                                                   'iembed': iembed,
                                                   })

    y = ant.tensor_out
    y_ = tf.placeholder("float", [None, None], name='Target')
    ant.placeholderdict['ratings'] = y_  # put the new placeholder in the graph for training
    with tf.name_scope('objective'):
        objective = tf.reduce_sum(tf.square(y_ - y))
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
                                    model_name='dnn_concat',
                                    make_histograms=False,
                                    save=False,
                                    tensorboard=False)
        model.train(data.train, dev=data.dev, eval_schedule=eval_rate)
    return model