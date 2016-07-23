=========================
Generic Model Tutorial
=========================

The generic_model module abstracts away from many common training scenarios for a reusable model training interface.

Here is sample code in straight tensorflow for the simply Mnist tutorial.

.. code-block:: python
    :linenos:

    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    accuracy_summary = tf.scalar_summary('Accuracy', accuracy)
    session = tf.Session()
    summary_writer = tf.train.SummaryWriter('log/logistic_regression', session.graph.as_graph_def())
    session.run(tf.initialize_all_variables())

    for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
      acc, accuracy_summary_str = session.run([accuracy, accuracy_summary], feed_dict={x: mnist.test.images,
                                                                                y_: mnist.test.labels})
      summary_writer.add_summary(accuracy_summary_str, i)
      print('Accuracy: %f' % acc)

In the case of this simple Mnist example lines 1-14 process data and define the computational graph, whereas lines
16-28 involve choices about how to train the model, and actions to take during training. An ANTK :any:`Model` object
parameterizes these choices for a wide variety of use cases to allow for reusable code to train a model. To achieve the
same result as our simple Mnist example we can replace lines 17-29 above as follows:

.. code-block:: python
    :linenos:

    import tensorflow as tf
    from antk.core import generic_model
    from tensorflow.examples.tutorials.mnist import input_data
    from antk.core import loader
    import os
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    predictions = tf.argmax(y, 1)
    correct_prediction = tf.equal(predictions, tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    trainset = loader.DataSet({'images': mnist.train.images}, {'labels': mnist.train.labels})
    print(type(mnist.train.labels[0,0]))
    devset = loader.DataSet({'images': mnist.test.images},{'labels': mnist.test.labels})
    pholders = {'images': x, 'labels': y_}
    model = generic_model.Model(cross_entropy, pholders,
                                mb=100,
                                maxbadcount=500,
                                learnrate=0.001,
                                verbose=True,
                                epochs=100,
                                evaluate=1 - accuracy,
                                model_name='simple_mnist',
                                tensorboard=False)

    dev = loader.DataSet({'images': mnist.test.images, 'labels': mnist.test.labels})
    dev.show()
    train = loader.DataSet({'images': mnist.train.images, 'labels': mnist.train.labels})
    train.show()
    model.train(train, dev=dev, eval_schedule=100)

Notice that we had to change the evaluation function to take advantage of early stopping so that when the model does
better the evaluation function is less. So we evaluate on 1 - accuracy = error. Using :any:`generic_model` now allows us to easily test out
different training scenarios by changing some of the default settings.

We can go through all the options and see what is available. Replace your call to the :any:`Model` constructor with
the following call that makes all default parameters explicit.

.. code-block:: python

    model = generic_model.Model(cross_entropy, pholders,
                                maxbadcount=20,
                                momentum=None,
                                mb=1000,
                                verbose=True,
                                epochs=50,
                                learnrate=0.01,
                                save=False,
                                opt='grad',
                                decay=[1, 1.0],
                                evaluate=1-accuracy,
                                predictions=predictions,
                                logdir='log/simple_mnist',
                                random_seed=None,
                                model_name='simple_mnist',
                                clip_gradients=0.0,
                                make_histograms=False,
                                best_model_path='/tmp/model.ckpt',
                                save_tensors={},
                                tensorboard=False):

Suppose we want to save our best set of weights, and bias for this logistic regression model, and make a
tensorboard histogram plot of how the weights change over time. Also, we want to be able to make predictions with our trained model as well.

We just need to set a few arguments in the call to the :any:`Model` constructor:

.. code-block:: python

    save_tensors=[W, b]
    make_histograms=True

You can view the graph with histograms with the usual tensorboard call from the terminal.

.. code-block:: bash

    $ tensorboard --logdir log/simple_mnist

Also, to be able to make predictions with our trained model we need to set the predictions argument in the call to
the constructor as below:

.. code-block:: python

    predictions=tf.argmax(y,1)

Now we can get predictions from the trained model using:

.. code-block:: python

    dev_classes = model.predict(devset)

