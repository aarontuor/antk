from antk.core import node_ops, generic_model, loader, config
import tensorflow as tf

# -train_feat TRAIN_FEAT_FN
# -train_target TRAIN_TARGET_FN
# -dev_feat DEV_FEAT_FN
# -dev_target DEV_TARGET_FN
# -epochs EPOCHS
# -learnrate LEARNRATE
# -nunits NUM_HIDDEN_UNITS
# -type PROBLEM_MODE
# -hidden_act HIDDEN_UNIT_ACTIVATION
# -optimizer OPTIMIZER
# [-mb MINIBATCH_SIZE]
# [-nlayers NUM_HIDDEN_LAYERS]

def deep(data, configfile,
        epochs=10,
        learnrate=0.001
        layers=[10, 10, 10],
        act='tanh',
        opt='grad',
        mb=500,
        type='r')


    with tf.name_scope('ant_graph'):
        ant = config.AntGraph(configfile,
                              data=data.dev.features,
                              marker='-',
                              graph_name='deep',
                              variable_bindings = {'layers': layers,
                                                   'act': act})
    y = ant.tensor_out
    y_ = tf.placeholder("float", [None, None], name='Target')
    ant.placeholderdict['targets'] = y_
    if type == 'r':
        objective = node_ops.mse(y, y_)
    if type == 'c':
        objective = node_ops.accuracy(y, y_)

    with tf.name_scope('objective'):
        objective = (tf.reduce_sum(tf.square(y_ - y)) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['huser'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['hitem'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['ubias'])) +
                     lamb*tf.reduce_sum(tf.square(ant.tensordict['ibias'])))
    with tf.name_scope('dev_rmse'):
        dev_rmse = node_ops.rmse(y_, y)
        #dev_rmse = tf.sqrt(tf.div(tf.reduce_sum(tf.square(y - y_)), data.dev.num_examples))

    model = generic_model.Model(objective, ant.placeholderdict,
                                mb=mb,
                                learnrate=learnrate,
                                verbose=verbose,
                                maxbadcount=maxbadcount,
                                epochs=epochs,
                                evaluate=dev_rmse,
                                predictions=y,
                                model_name='mf',
                                random_seed=500)
    model.train(data.train, dev=data.dev, eval_schedule=200)

    return model
