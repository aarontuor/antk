.. _Aaron Tuor: http://sw.cs.wwu.edu/~tuora/aarontuor/
.. _Brian Hutchinson: http://fw.cs.wwu.edu/~hutchib2/
.. _David Palzer: https://cse.wwu.edu/computer-science/palzerd

.. paper links

.. _Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift: http://arxiv.org/pdf/1502.03167v3.pdf
.. _Neural Networks and Deep Learning: http://natureofcode.com/book/chapter-10-neural-networks/
.. _Using Neural Nets to Recognize Handwritten Digits: http://neuralnetworksanddeeplearning.com/chap1.html
.. _Dropout A Simple Way to Prevent Neural Networks from Overfitting: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
.. _Training Very Deep Networks: http://arxiv.org/pdf/1507.06228v2.pdf
.. _Deep Residual Learning for Image Recognition: http://arxiv.org/pdf/1512.03385v1.pdf

.. convolutional nets

.. _Tensorflow Deep MNIST for Experts: https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html
.. _Tensorflow Convolutional Neural Networks: https://www.tensorflow.org/versions/r0.7/tutorials/deep_cnn/index.html#convolutional-neural-networks
.. _ImageNet Classification with Deep Convolutional Neural Networks: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
.. _skflow/examples/text_classification_character_cnn.py: https://github.com/tensorflow/skflow/blob/master/examples/text_classification_character_cnn.py
.. _skflow/examples/text_classification_cnn.py: https://github.com/tensorflow/skflow/blob/master/examples/text_classification_cnn.py
.. _Character-level Convolutional Networks for Text Classification: http://arxiv.org/pdf/1509.01626v2.pdf

.. tensorflow/skflow links for use in docs

.. _Tensorflow: https://www.tensorflow.org/
.. _tensorflow: https://www.tensorflow.org/
.. _tensorflow's: https://www.tensorflow.org/
.. _variable_scope: https://www.tensorflow.org/versions/r0.7/how_tos/variable_scope/index.html
.. _Tensor: https://www.tensorflow.org/versions/r0.7/api_docs/python/framework.html#Tensor
.. _tensor: https://www.tensorflow.org/versions/r0.7/api_docs/python/framework.html#Tensor
.. _tensors: https://www.tensorflow.org/versions/r0.7/api_docs/python/framework.html#Tensor
.. _skflow: https://github.com/tensorflow/skflow
.. _placeholder: https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
.. _Placeholder: https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
.. _embedding_lookup: https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#embeddings
.. _dnn_ops.py: https://github.com/tensorflow/skflow/blob/master/skflow/ops/dnn_ops.py
.. _rnn_cell.py: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py
.. _nn.py: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn.py
.. _batch_norm_ops.py: https://github.com/tensorflow/skflow/blob/master/skflow/ops/batch_norm_ops.py
.. _dropout_ops.py: https://github.com/tensorflow/skflow/blob/master/skflow/ops/dropout_ops.py
.. _Efficient BackProp: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
.. _Tensor Decompositions and Applications: http://dl.acm.org/citation.cfm?id=1655230

*******************
node_ops
*******************



The :any:`node_ops` module consists of a collection of mid to high level functions which take a `tensor`_ or structured list of tensors, perform a sequence of tensorflow operations, and return a tensor or structured list of tensors. All node_ops functions conform to
the following specifications.

- All tensor input (if it has tensor input) is received by the function's first argument, which may be a single tensor, a list of tensors, or a structured list of tensors, e.g., a list of lists of tensors.

- The return is a tensor, list of tensors or structured list of tensors.

- The final argument is an optional *name* argument for `variable_scope`_.

Use Cases
=========

:any:`node_ops` functions may be used in a `tensorflow`_ script wherever you might use an equivalent sequence of tensorflow
 ops during the graph building portion of a script.

:any:`node_ops` functions may be called in a .config file following the .config file syntax which is explained in :any:`config_tutorial`.

Making Custom ops For use With :any:`config` module
===================================================
The :any:`AntGraph` constructor in the *config* module will add tensor operations to the tensorflow graph which are specified
in a config file and fit the node_ops spec but not defined in the *node_ops* module. This leaves the user free to define new
node_ops for use with the config module, and to use many pre-existing tensorflow and third party defined ops with the config
module as well.

The :any:`AntGraph` constructor has two arguments *function_map* and *imports* which may be used to incorporate custom node_ops.

- **function_map** is a hashmap of function_handle:function, key value pairs

- **imports** is a hashmap of module_name:path_to_module pairs for importing an entire module of custom node_ops.

Accessing Tensors Created in a node_ops Function
================================================

Tensors which are created by a node_ops function but not returned to the caller are kept track of in an intuitive fashion
by calls to **tf.add_to_collection**. Tensors can be accessed later by calling **tf.get_collection** by the following convention:

For a node_ops function which was handed the argument **name='some_name'**:

- The **nth weight tensor** created may be accessed as

.. code-block:: python

   tf.get_collection('some_name_weights')[n]

- The **nth bias tensor** created may be accessed as

.. code-block:: python

   tf.get_collection('some_name_bias')[n]

- The **nth preactivation tensor** created may be accessed as

.. code-block:: python

   tf.get_collection('some_name_preactivation')[n]

- The **nth activation tensor** created may be accessed as

.. code-block:: python

   tf.get_collection('some_name_activations')[n]

- The **nth post dropout** tensor created may be accessed as

.. code-block:: python

   tf.get_collection('some_name_dropouts')[n]

- The **nth post batch normalization tensor** created may be accessed as

.. code-block:: python

   tf.get_collection('some_name_bn')[n]

- The **nth tensor created not listed above** may be accessed as

.. code-block:: python

   tf.get_collection('some_name')[n],

- The **nth hidden layer size skip transform** (for :any:`residual_dnn`):

.. code-block:: python

   tf.get_collection('some_name_skiptransform')[n]

- The **nth skip connection** (for :any:`residual_dnn`):

.. code-block:: python

   tf.get_collection('some_name_skipconnection')[n]

- The **nth transform layer** (for :any:`highway_dnn`):

.. code-block:: python

   tf.get_collection('some_name_transform')[n]

Weights
=======
Here is a simple wrapper for common initializations of tensorflow `Variables`_. There is a option for
l2 regularization which is automatically added to the objective function when using the :any:`generic_model` module.

:any:`weights`

Placeholders
============
Here is a simple wrapper for a tensorflow placeholder constructor that when used in conjunction with
the :any:`config` module, infers the correct dimensions of the `placeholder`_ from a string hashed set
of numpy matrices.

:any:`placeholder`

Neural Networks
===============
.. warning::

   The output of a neural network node_ops function is the output after activation of the last hidden layer.
   For regression an additional call to :any:`linear` must be made and for classification and additional call to
   :any:`mult_log_reg` must be made.

Initialization
--------------

Neural network weights are initialized with the following scheme where the range is dependent on the second
dimension of the input layer:

.. code-block:: python

   if activation == 'relu':
      irange= initrange*numpy.sqrt(2.0/float(tensor_in.get_shape().as_list()[1]))
   else:
      irange = initrange*(1.0/numpy.sqrt(float(tensor_in.get_shape().as_list()[1])))

*initrange* above is defaulted to 1. The user has the choice of several distributions,

- 'norm', 'tnorm': *irange* scales distribution with mean zero and standard deviation 1.

- 'uniform': *irange* scales uniform distribution with range [-1, 1].

- 'constant': *irange* equals the initial scalar entries of the matrix.

Dropout
-------
Dropout with the specified *keep_prob* is performed post activation.

Batch Normalization
-------------------
If requested batch normalization is performed after dropout.

Networks
--------

:any:`dnn`

:any:`residual_dnn`

:any:`highway_dnn`

:any:`convolutional_net`

Loss Functions and Evaluation Metrics
=====================================
:any:`se`

:any:`mse`

:any:`rmse`

:any:`mae`

:any:`cross_entropy`

:any:`other_cross_entropy`

:any:`perplexity`

:any:`detection`

:any:`recall`

:any:`precision`

:any:`accuracy`

:any:`fscore`

Custom Activations
==================
:any:`ident`

:any:`tanhlecun`

:any:`mult_log_reg`

Matrix Operations
=================
:any:`concat`

:any:`x_dot_y`

:any:`cosine`

:any:`linear`

:any:`embedding`

:any:`lookup`

:any:`khatri_rao`


Tensor Operations
=================

Some tensor operations from Kolda and Bader's `Tensor Decompositions and Applications` are provided here. For now these
operations only work on up to order 3 tensors.

:any:`nmode_tensor_tomatrix`

:any:`nmode_tensor_multiply`

:any:`binary_tensor_combine`

:any:`ternary_tensor_combine`

Batch Normalization
===================

:any:`batch_normalize`


Dropout
========
Dropout is automatically 'turned' off during evaluation when used in conjuction with the :any:`generic_model` module.

:any:`dropout`

API
====

.. automodule:: node_ops
   :members:
   :undoc-members:

   .. autofunction:: placeholder(dtype, shape=None, data=None, name='placeholder')
   .. autofunction:: weights(distribution, shape, dtype=tf.float32, initrange=1e-5,
            seed=None, l2=0.0, name='weights')
   .. autofunction:: cosine(operands, name='cosine')
   .. autofunction:: x_dot_y(operands, name='x_dot_y')
   .. autofunction:: lookup(dataname=None,  data=None,  indices=None, distribution='uniform', initrange=0.1, l2=0.0, shape=None, makeplace=True, name='lookup')
   .. autofunction:: embedding(tensors, name='embedding')
   .. autofunction:: mult_log_reg(tensor_in, numclasses=None, data=None, dtype=tf.float32, initrange=1e-10, seed=None, l2=0.0, name='log_reg')
   .. autofunction:: concat(tensors, output_dim, name='concat')
   .. autofunction:: dnn(tensor_in, hidden_units, activation='tanh', distribution='tnorm',initrange=1.0, l2=0.0, bn=False, keep_prob=None, fan_scaling=False, name='dnn')
   .. autofunction:: residual_dnn(tensor_in, hidden_units, activation='tanh', distribution='tnorm', initrange=1.0, l2=0.0, bn=False, keep_prob=None, fan_scaling=False, skiplayers=3, name='residual_dnn')
   .. autofunction:: highway_dnn(tensor_in, hidden_units, activation='tanh', distribution='tnorm', initrange=1.0, l2=0.0, bn=False, keep_prob=None, fan_scaling=False, bias_start=-1, name='highway_dnn')
   .. autofunction:: dropout(tensor_in, prob, name='Dropout'):
   .. autofunction:: linear(tensor_in, output_size, bias, bias_start=0.0, distribution='tnorm', initrange=1.0, l2=0.0, name="Linear")
   .. autofunction:: batch_normalize(tensor_in, epsilon=1e-5, decay=0.999, name="batch_norm")
   .. autofunction:: nmode_tensor_tomatrix(tensor, mode, name='nmode_matricize'):
   .. autofunction:: nmode_tensor_multiply(tensors, mode, leave_flattened=False, keep_dims=False, name='nmode_multiply')
   .. autofunction:: ternary_tensor_combine(tensors, initrange=1e-5, distribution='tnorm', l2=0.0, name='ternary_tensor_combine')
   .. autofunction:: khatri_rao(tensors, name='khatrirao')
   .. autofunction:: binary_tensor_combine2(tensors, output_dim=10, initrange=1e-5, name='binary_tensor_combine2')
   .. autofunction:: se(predictions, targets, name='squared_error')
   .. autofunction:: mse(predictions, targets, name='mse')
   .. autofunction:: rmse(predictions, targets, name='rmse')
   .. autofunction:: mae(predictions, targets, name='mae')
   .. autofunction:: other_cross_entropy(predictions, targets, name='logistic_loss')
   .. autofunction:: cross_entropy(predictions, targets, name='cross_entropy')
   .. autofunction:: perplexity(predictions, targets, name='perplexity')
   .. autofunction:: detection(predictions, threshold, name='detection')
   .. autofunction:: recall(predictions, targets, threshold=0.5, detects=None, name='recall')
   .. autofunction:: precision(predictions, targets, threshold=0.5, detects=None, name='precision')
   .. autofunction:: fscore(predictions=None, targets=None, threshold=0.5, precisions=None, recalls=None, name='fscore')
   .. autofunction:: accuracy(predictions, targets, name='accuracy')
