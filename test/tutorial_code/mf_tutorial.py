import tensorflow as tf
from antk.core import config
from antk.core import generic_model
from antk.core import loader
from antk.core import node_ops

loader.maybe_download('ml100k.tar.gz', '.',
                      'http://sw.cs.wwu.edu/~tuora/aarontuor/ml100k.tar.gz')
loader.untar('ml100k.tar.gz')
data = loader.read_data_sets('ml100k', folders=['dev', 'train'],
                              hashlist=['item', 'user', 'ratings'])
data.show()
data.train.labels['ratings'] = loader.center(data.train.labels['ratings'])
data.dev.labels['ratings'] = loader.center(data.dev.labels['ratings'])

with tf.variable_scope('mfgraph'):
        ant = config.AntGraph('mf2.config',
                                data=data.dev.features,
                                marker='-',
                                variable_bindings = {'kfactors': 100, 'initrange':0.001, 'l2':0.1})


y = ant.tensor_out
y_ = tf.placeholder("float", [None, None], name='Target')
ant.placeholderdict['ratings'] = y_ # put the new placeholder in the graph for training
objective = tf.reduce_sum(tf.square(y_ - y))
dev_rmse =  node_ops.rmse(y, y_)
dev_mae = node_ops.mae(y, y_)

model = generic_model.Model(objective, ant.placeholderdict,
          mb=500,
          learnrate=0.01,
          verbose=True,
          maxbadcount=10,
          epochs=100,
          evaluate=dev_rmse,
          predictions=y,
          save_tensors={'dev_mae': dev_mae})
model.train(data.train, dev=data.dev)
