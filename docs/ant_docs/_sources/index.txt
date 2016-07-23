.. Automated Neural-graph toolkit documentation master file, created by
   sphinx-quickstart on Tue Aug 11 05:04:40 2009.

.. Author links

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


.. tensorflow links for use in docs

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
.. _Install tensorflow: https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html
.. _Install graphviz: http://www.graphviz.org/

*Principal Authors* `Aaron Tuor`_ , `Brian Hutchinson`_

About ANTk
==========

The Automated Neural-graph toolkit is a high level machine learning toolkit built on top of Google's Tensorflow_ to
facilitate rapid prototyping of Neural Network models which may consist of multiple models chained together. This includes
models which have multiple input and/or multiple output streams.

ANTk will be most useful to people who have gone through some of the basic tensorflow tutorials, have some machine learning
background, and wish to take advantage
of some of tensorflow's more advanced features. The code itself is consistent, well-formatted, well-documented, and abstracted
only to a point necessary for code reuse, and complex model development. The toolkit code contains tensorflow usage developed and discovered over six
months of machine learning research conducted in tensorflow, by Hutch Research based out of Western Washington University's Computer Science Department.

The kernel of the toolkit is comprised of 4 independent, but
complementary modules:

   **loader:**
      Implements a general purpose data loader for python non-sequential machine learning tasks.
      Contains functions for common data pre-processing tasks.

   **config:**
      Facilitates the generation of complex tensorflow models, built from
      compositions of tensorflow functions.

   **node_ops:**
      Contains functions taking a tensor or structured list of tensors and returning a tensor or structured list of tensors.
      The functions are commonly used compositions of tensorflow functions which operate on tensors.

   **generic_model:**
      A general purpose model builder equipped with generic train, and predict functions which takes parameters for
      optimization strategy, mini-batch, etc...

**Motivation:**

   Working at a high level of abstraction is important for the rapid
   development of machine learning models. Many successful state of the art models chain together or create an ensemble of several
   complex models. To facilitate the need for building models whose components are models we have
   developed a highly modularized set of utilities.

   While this high level of abstraction is often attractive for development, when working with a highly abstracted machine learning toolkit it
   is often difficult to assess details of implementation and the underlying math behind a packaged model. To address this concern
   we have made the toolkit implementation and underlying math as transparent as possible. There are links to source code, and relevant scientific papers
   in the API and we have taken pains to illuminate the workings of complex code with well formatted mathematical equations. Also,
   we have taken care to allow easy access to tensor objects created by high level operations such as deep neural networks.

**Design methodology:**

   ANTK was designed to be highly modular, and allow for a high level of abstraction with a great degree of transparency to
   the underlying implementation. We hope that this design can eliminate the reproduction of coding efforts without sacrificing important
   knowledge about implementation that may effect the overall performance of a model.

Dependencies
===============
Tensorflow, scipy, numpy, sklearn, graphviz.


`Install tensorflow`_

`Install graphviz`_

Installation
=============
A virtual environment is recommended for installation. Make sure that tensorflow is installed in your virtual environment
and graphviz is installed on your system.

In a terminal:

.. code-block:: bash

    (venv)$ mkdir antk
    (venv)$ cd antk
    (venv)$ git init
    Initialized empty Git repository in /home/tuora/garbage/.git/
    (venv)$ git remote add origin https://github.com/aarontuor/antk.git
    (venv)$ git pull origin master
    ...
    (venv)$ python setup.py develop

Documentation
=============


.. toctree::
   :maxdepth: 2

   api.rst
   tutorials.rst
   command_line.rst
   models.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



