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

.. _Distributed Representations of Words and Phrases and their Compositionality: http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
======
Models
======

The models below are available in ANTk. If the model takes a config file then a sample config is provided.

Skipgram
---------
.. automodule:: skipgram
   :members:
   :undoc-members:

Matrix Factorization
--------------------
.. automodule:: mfmodel
   :members:
   :undoc-members:

Sample Config
*************
.. code-block:: python

    dotproduct x_dot_y()
        -huser lookup(dataname='user', initrange=0.001, shape=[None, 20])
        -hitem lookup(dataname='item', initrange=0.001, shape=[None, 20])
        -ibias lookup(dataname='item', initrange=0.001, shape=[None, 1])
        -ubias lookup(dataname='user', initrange=0.001, shape=[None, 1])

Low Rank Matrix Factorization is a popular machine learning technique used to produce recommendations
given a set of ratings a user has given an item. The known ratings are collected in a user-item utility matrix
and the missing entries are predicted by optimizing a low rank factorization of the utility matrix given the known
entries. The basic idea behind matrix factorization models is that the information encoded for items
in the columns of the utility matrix, and for users in the rows of the utility matrix is not
exactly independent. We optimize the objective function :math:`\sum_{(u,i)} (R_{ui} - P_i^T U_u)^2` over the observed
ratings for user *u* and item *i* using gradient descent.

.. image:: _static/factormodel.png
   :align: center

We can express the same optimization in the form of a computational graph that will play nicely with tensorflow:

.. image:: _static/graphmf.png
   :align: center

Here :math:`xitem_i`, and :math:`xuser_j` are some representation of the indices for the user and item vectors in the utility matrix.
These could be one hot vectors, which can then be matrix multiplied by the *P* and *U* matrices to select the corresponding
user and item vectors. In practice it is much faster to let :math:`xitem_i`, and :math:`xuser_j` be vectors of indices
which can be used by tensorflow's **gather** or **embedding_lookup** functions to select the corresponding vector from
the *P* and *U* matrices.

DSSM (Deep Structured Semantic Model) Variant
---------------------------------------------
.. automodule:: dssm_model
   :members:
   :undoc-members:

.. image:: _static/dssm.png
    :align: center

Sample Config
*************
.. code-block:: python

    dotproduct x_dot_y()
    -user_vecs ident()
    --huser lookup(dataname='user', initrange=$initrange, shape=[None, $kfactors])
    --hage dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=.8)
    ---agelookup embedding()
    ----age placeholder(tf.float32)
    ----user placeholder(tf.int32)
    --hsex dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    ---sexlookup embedding()
    ----sex_weights weights('tnorm', tf.float32, [2, $kfactors])
    ----sexes embedding()
    -----sex placeholder(tf.int32)
    -----user placeholder(tf.int32)
    --hocc dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    ---occlookup embedding()
    ----occ_weights weights('tnorm', tf.float32, [21, $kfactors])
    ----occs embedding()
    -----occ placeholder(tf.int32)
    -----user placeholder(tf.int32)
    --hzip dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    ---ziplookup embedding()
    ----zip_weights weights('tnorm', tf.float32, [1000, $kfactors])
    ----zips embedding()
    -----zip placeholder(tf.int32)
    -----user placeholder(tf.int32)
    --husertime dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    ---time placeholder(tf.float32)
    -item_vecs ident()
    --hitem lookup(dataname='item', initrange=$initrange, shape=[None, $kfactors])
    --hgenre dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    ---genrelookup embedding()
    ----genres placeholder(tf.float32)
    ----item placeholder(tf.int32)
    --hmonth dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    ---monthlookup embedding()
    ----month_weights weights('tnorm', tf.float32, [12, $kfactors])
    ----months embedding()
    -----month placeholder(tf.int32)
    -----item placeholder(tf.int32)
    --hyear dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    ---yearlookup embedding()
    ----year placeholder(tf.float32)
    ----item placeholder(tf.int32)
    --htfidf dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    ---tfidflookup embedding()
    ----tfidf_doc_term placeholder(tf.float32)
    ----item placeholder(tf.int32)
    --hitemtime dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    ---time placeholder(tf.float32)
    -ibias lookup(dataname='item', shape=[None, 1], initrange=$initr
Weighted DSSM variant
---------------------
.. automodule:: dsaddmodel
   :members:
   :undoc-members:

This model is the same architecture as the variant of DSSM above but with a different loss:

.. image:: _static/weightedloss.png
    :align: center

Binary Tree of Deep Neural Networks for Multiple Inputs
-------------------------------------------------------
.. automodule:: tree_model
   :members:
   :undoc-members:

.. image:: _static/tree1.png
    :align: center

Sample Config
*************
.. code-block:: python

    dotproduct x_dot_y()
    -all_user dnn([$kfactors,$kfactors,$kfactors], activation='tanh',bn=True,keep_prob=None)
    --tanh_user tf.nn.tanh()
    ---merge_user concat($kfactors)
    ----huser lookup(dataname='user', initrange=$initrange, shape=[None, $kfactors])
    ----hage dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    -----agelookup embedding()
    ------age placeholder(tf.float32)
    ------user placeholder(tf.int32)
    ----hsex dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    -----sexlookup embedding()
    ------sex_weights weights('tnorm', tf.float32, [2, $kfactors])
    ------sexes embedding()
    -------sex placeholder(tf.int32)
    -------user placeholder(tf.int32)
    ----hocc dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    -----occlookup embedding()
    ------occ_weights weights('tnorm', tf.float32, [21, $kfactors])
    ------occs embedding()
    -------occ placeholder(tf.int32)
    -------user placeholder(tf.int32)
    ----hzip dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    -----ziplookup embedding()
    ------zip_weights weights('tnorm', tf.float32, [1000, $kfactors])
    ------zips embedding()
    -------zip placeholder(tf.int32)
    -------user placeholder(tf.int32)
    ----husertime dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    -----time placeholder(tf.float32)
    -all_item dnn([$kfactors,$kfactors,$kfactors], activation='tanh',bn=True,keep_prob=None)
    --tanh_item tf.nn.tanh()
    ---merge_item concat($kfactors)
    ----hitem lookup(dataname='item', initrange=$initrange, shape=[None, $kfactors])
    ----hgenre dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    -----genrelookup embedding()
    ------genres placeholder(tf.float32)
    ------item placeholder(tf.int32)
    ----hmonth dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    -----monthlookup embedding()
    ------month_weights weights('tnorm', tf.float32, [12, $kfactors])
    ------months embedding()
    -------month placeholder(tf.int32)
    -------item placeholder(tf.int32)
    ----hyear dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    -----yearlookup embedding()
    ------year placeholder(tf.float32)
    ------item placeholder(tf.int32)
    ----htfidf dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    -----tfidflookup embedding()
    ------tfidf_doc_term placeholder(tf.float32)
    ------item placeholder(tf.int32)
    ----hitemtime dnn([$kfactors,$kfactors,$kfactors],activation='tanh',bn=True,keep_prob=None)
    -----time placeholder(tf.float32)
    -ibias lookup(dataname='item', shape=[None, 1], initrange=$initrange)
    -ubias lookup(dataname='user', shape=[None, 1], initrange=$initrange)



A Deep Neural Network with Concatenated Input Streams
-----------------------------------------------------
.. automodule:: dnn_concat_model
   :members:
   :undoc-members:

.. image:: _static/dnn_concat.png
    :align: center

Sample Config
*************
.. code-block:: python

    out linear(1, True)
    -h1 dnn([16, 8], activation='tanhlecun', bn=True, keep_prob=.95)
    --x concat(24)
    ---huser lookup(dataname='user', initrange=.001, shape=[None, $embed])
    ---hitem lookup(dataname='item', initrange=.001, shape=[None, $embed])

Multiplicative Interaction between Text, User, and Item
--------------------------------------------------------

.. image:: _static/multoutputs.png
    :align: center




