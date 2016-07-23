Part 3: Crafting a new model
************************************

So far we have been using functions in our config files that are defined in the :any:`node_ops` module. In the model we
have defined, the **concat** nodes are passed directly to **dnn** nodes with no intervening non-linearity. We can introduce
non-linear transformations at these points by directly calling tensorflow non-linear transforms in the config file.

.. todo::

    Introduce tensorflow non-linear transforms in tree.config

We also have the ablility to introduce new node functions which are not defined by tensorflow or :any:`node_ops`.
One thing we might try is to replace our **dnn** nodes with **highway_dnn** nodes that we need to define. Here is a first
approximation of defining a new **highway_dnn** node:

.. code-block:: python

    import node_ops

    def highway_dnn(tensor_in, hidden_units, activation='tanh', keep_prob=None, name='highway_dnn'):
        activation = node_ops.ACTIVATION[activation]
        with tf.variable_scope(name):
            for i, n_units in enumerate(hidden_units):
                with tf.variable_scope('layer%d' % i):
                    with tf.variable_scope('hidden'):
                        hidden = activation(node_ops.linear(tensor_in, n_units, True))
                    with tf.variable_scope('transform'):
                        transform = tf.sigmoid(node_ops.linear(tensor_in, n_units, True))
                    tensor_in = tf.mul(hidden, transform) + tf.mul(tensor_in, 1 - transform)
                    if keep_prob:
                        tensor_in = node_ops.dropout(tensor_in, keep_prob)
            return tensor_in

This looks okay for a first approximation. However we would like to initialize the bias for the transform gate to some
negative value, so we have to set the optional *bias_start=0* parameter in the call to :any:`linear` for the transform gate to
a value we would like.

.. todo::

    Alter the **highway_dnn** code to take an optional parameter for transform bias initialization. Replace **dnn** nodes
    in tree.config with **highway_dnn** nodes.

Now we need to let the :any:`AntGraph` constructor know about our new node by using the factory constructor :any:`graph_setup` with
one of the two optional arguments introducing new node functions. If
your new node definition is in it's own module possibly full of other node definitions you have created you can use the
*imports* argument which is a dictionary with module names as keys and the paths to modules as values. In our case since we
have only made one new node function we will use the *function_map* parameter which takes a dictionary of function_name keys
and function values. So we replace our :any:`AntGraph` constructor call as follows:

.. code-block:: python

    with tf.variable_scope('mfgraph'):
        ant = config.graph_setup('tree.config',
                                  data=data-dev,
                                  marker='.',
                                  function_map = {'highway_dnn': highway_dnn},
                                  variable_bindings = {'kfactors': 20, 'initrange':0.001})

Here is an example config file and corresponding graphviz dot image adding non-linearity and a **highway_dnn** node.
Notice that since highway networks have a fixed dimension size for hidden layers we have used an intermediate **dnn** node
to map the input to a dimension we want.

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


.. image:: _static/highway_nonlinear.png
    :align: center

Now that we are familiar with the tricks for crafting new models we can engage in creative enterprise. We have been using
the scalar ratings targets for objectives, but there are also one hot targets for classification included in the dataset.

.. todo::

    Using the data provided, craft your own model, and do some preliminary testing to see how it performs against
    the other models you have explored. If you run into a bug, contact me and I will try to address it but feel free
    to just take note of the bug, abandon the effort, and try something else.
    Some ideas:
        Classification

        Multiple targets of regression and classification

        A different combine function which has a multiplicative interaction between data streams

        Combining data streams in a binary fashion (for item streams don't combine all three at once)

        Scale the regression targets and use the built in :any:`cosine` node function instead of :any:`x_dot_y`

        Make the branches dealing with indices linear, as they are in basic MF, and combine them later in the tree

        Create an LSTM node that takes a sequence as input and outputs the last hidden state vector.

    For this task results are not paramount. Think of a way to address the data that makes sense and explores the kind of
    functionality you may want to employ the toolkit for.

