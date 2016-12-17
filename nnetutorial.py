import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from antk.core.loader import DataSet

class SimpleModel():
    """
    A class for gradient descent training arbitrary models.
    
    :param loss: loss_tensor defined in graph
    :param learnrate: step_size for gradient descent
    :param mb: Size of minibatch, 
               how many datapoints to run through graph at a time
    """
    
    def __init__(self, loss, eval_tensor, ph_dict, learnrate=0.01, debug=False):
        self.loss = loss
        self.ph_dict = ph_dict
        self.eval_tensor = eval_tensor
        self.debug=debug
        self.train_step = tf.train.GradientDescentOptimizer(learnrate).minimize(loss)
        self.init = tf.initialize_all_variables()
        self.epoch = 0.0
        self.sess = tf.Session()
        self.sess.run(self.init)
        
    def train(self, train_data, dev_data, mb=100):
        """
        :param train_data: A list of train data
        :param dev_data
        """
        for i in range(1000):
            self.epoch += float(mb)/float(train.num_examples)
            new_batch = train.next_batch(mb)
            self.sess.run(self.train_step, feed_dict=self.get_feed_dict(new_batch, self.ph_dict))
            sys.stdout.write('epoch %.2f\tdev eval: %.4f' % (self.epoch, self.evaluate(dev_data)))
            sys.stdout.write('\r')
        
    def evaluate(self, data):
        return self.sess.run(self.eval_tensor, feed_dict=self.get_feed_dict(data, self.ph_dict))
                                 
    def get_feed_dict(self, batch, ph_dict):

        """
        :param batch: A dataset object.
        :param ph_dict: A dictionary where the keys match keys in batch, and the values are placeholder tensors
        :return: A feed dictionary with keys of placeholder tensors and values of numpy matrices
        """
        
        datadict = batch.features.copy()
        datadict.update(batch.labels)
                                 
        if self.debug:
            for desc in ph_dict:
                print('%s\n\tph: %s\t%s\tdt: %s\t%s' % (desc,
                                                        ph_dict[desc].get_shape().as_list(), ph_dict[desc].dtype, 
                                                        datadict[desc].shape, datadict[desc].dtype))
        return {ph_dict[key]:datadict[key] for key in ph_dict}

# pass 2 at nnet classifier
def nnet_classifier2(x, layers=[50,10], act=tf.nn.relu, name='nnet'):
    """
    Second pass at a classifier, eliminate repeated code. Bonus: An arbitrarilly deep neural network.
    
    :param x: Input to the network
    :param layers: Sizes of network layers
    :param act: Activation function to produce hidden layers of neural network.
    :param name: An identifier for retrieving tensors made by dnn
    """
    
    for ind, hidden_size in enumerate(layers):
        with tf.variable_scope('layer_%s' % ind):
            fan_in = x.get_shape().as_list()[1]
            scale = 1.0/np.sqrt(fan_in)
            W = tf.Variable(scale*tf.truncated_normal([fan_in, hidden_size], 
                                                     mean=0.0, stddev=1.0, 
                                                     dtype=tf.float32, seed=None, name='W'))
            tf.add_to_collection(name + '_weights', W)
            b = tf.Variable(tf.zeros([hidden_size])) 
            tf.add_to_collection(name + '_bias', b)
            x = tf.matmul(x,W) + b
            if ind != len(layers) - 1:
                x = act(x, name='h' + str(ind)) # The hidden layer
            tf.add_to_collection(name + '_activation', x)
    return tf.nn.softmax(x)

# Data prep ================================================================
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train = DataSet({'images': mnist.train.images}, labels={'digits': mnist.train.labels}, mix=False)
dev = DataSet({'images': mnist.test.images}, labels={'digits': mnist.test.labels}, mix=False)

# Make graph ============================================================
ph_dict = {'images': tf.placeholder(tf.float32, shape=[None, 784]),
           'digits': tf.placeholder(tf.float32, shape=[None, 10])}

prediction = nnet_classifier2(ph_dict['images'], 
                             layers = [50, 10], 
                             act = tf.nn.relu, 
                             name='nnet')

# Loss function
cross_entropy = -tf.reduce_sum(ph_dict['digits']*tf.log(prediction))

# Evaluate
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ph_dict['digits'],1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Make model
model = SimpleModel(cross_entropy, accuracy, ph_dict, learnrate=0.01)

# Train ================================================================
model.train(train, dev, mb=100)

