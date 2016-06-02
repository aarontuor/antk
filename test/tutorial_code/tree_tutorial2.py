import tensorflow as tf
from antk.core import config
from antk.core import generic_model
from antk.core import loader
from antk.core import node_ops

data = loader.read_data_sets('ml100k', folders=['dev', 'train', 'item', 'user'])
data.show()

data.train.labels['ratings'] = loader.center(data.train.labels['ratings'], axis=None)
data.dev.labels['ratings'] = loader.center(data.dev.labels['ratings'], axis=None)
data.user.features['age'] = loader.center(data.user.features['age'], axis=None)
data.item.features['year'] = loader.center(data.item.features['year'], axis=None)
data.user.features['age'] = loader.maxnormalize(data.user.features['age'])
data.item.features['year'] = loader.maxnormalize(data.item.features['year'])

datadict = data.user.features.copy()
datadict.update(data.item.features)
configdatadict = data.dev.features.copy()
configdatadict.update(datadict)

with tf.variable_scope('mfgraph'):
    ant = config.AntGraph('tree.config',
                            data=configdatadict,
                            marker='-',
                            variable_bindings = {'kfactors': 100, 'initrange':0.0001},
                            develop=False)

y = ant.tensor_out
y_ = tf.placeholder("float", [None, None], name='Target')
ant.placeholderdict['ratings'] = y_  # put the new placeholder in the graph for training
objective = tf.reduce_sum(tf.square(y_ - y))
dev_rmse =  node_ops.rmse(y, y_)

model = generic_model.Model(objective, ant.placeholderdict,
                            mb=500,
                            learnrate=0.00001,
                            verbose=True,
                            maxbadcount=20,
                            epochs=100,
                            evaluate=dev_rmse,
                            predictions=y)
model.train(data.train, dev=data.dev, supplement=datadict, eval_schedule=2000)
