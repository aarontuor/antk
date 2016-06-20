import scipy.sparse as sps
import tensorflow as tf
import numpy as np
import time
from antk.core import loader
import os
import datetime
import matplotlib.pyplot as plt

# ============================================================================================
# ============================CONVENIENCE DICTIONARY==========================================
# ============================================================================================
OPT = {'adam': tf.train.AdamOptimizer,
       'ada': tf.train.AdagradOptimizer,
       'grad': tf.train.GradientDescentOptimizer,
       'mom': tf.train.MomentumOptimizer}

# ============================================================================================
# ============================GLOBAL MODULE FUNCTIONS=========================================
# ============================================================================================
def get_feed_list(batch, placeholderdict, supplement=None, dropouts=None, dropout_flag='train'):

    """
    :param batch: A dataset object.
    :param placeholderdict: A dictionary where the keys match keys in batch, and the values are placeholder tensors
    :param supplement: A dictionary of numpy input matrices with keys corresponding to placeholders in placeholderdict, where the row size of the matrices do not correspond to the number of datapoints. For use with input data intended for `embedding_lookup`_.
    :param dropouts: Dropout tensors in graph.
    :param dropout_flag: Whether to use Dropout probabilities for feed forward.
    :return: A feed dictionary with keys of placeholder tensors and values of numpy matrices, paired by key
    """
    ph, dt = [], []
    datadict = batch.features.copy()
    datadict.update(batch.labels)
    if supplement:
        datadict.update(supplement)

    for desc in placeholderdict:
        ph.append(placeholderdict[desc])
        if sps.issparse(datadict[desc]):
            dt.append(datadict[desc].todense().astype(float, copy=False))
        elif type(datadict[desc]) is loader.HotIndex:
            dt.append(datadict[desc].vec)
        else:
            dt.append(datadict[desc])
    if dropouts:
        for prob in dropouts:
            ph.append(prob[0])
            if dropout_flag == 'train':
                dt.append(prob[1])
            elif dropout_flag == 'eval':
                dt.append(1.0)
            else:
                raise ValueError('dropout_flag must be "train" or "eval". Found %s' % dropout_flag)
    return {i: d for i, d in zip(ph, dt)}

def parse_summary_val(summary_str):
    """
    Helper function to parse numeric value from tf.scalar_summary

    :param summary_str: Return value from running session on tf.scalar_summary
    :return: A dictionary containing the numeric values.
    """
    summary_proto = tf.Summary()
    summary_proto.ParseFromString(summary_str)
    summaries = {}
    for val in summary_proto.value:
        summaries[val.tag] = val.simple_value
    return summaries

# ============================================================================================
# ============================GENERIC MODEL CLASS=============================================
# ============================================================================================
class Model(object):
    """
    Generic model builder for training and predictions.

    :param objective: Loss function
    :param placeholderdict: A dictionary of placeholders
    :param maxbadcount: For early stopping
    :param momentum: The momentum for tf.MomentumOptimizer
    :param mb: The mini-batch size
    :param verbose: Whether to print dev error, and save_tensor evals
    :param epochs: maximum number of epochs to train for.
    :param learnrate: learnrate for gradient descent
    :param save: Save best model to *best_model_path*.
    :param opt: Optimization strategy. May be 'adam', 'ada', 'grad', 'momentum'
    :param decay: Parameter for decaying learn rate.
    :param evaluate: Evaluation metric
    :param predictions: Predictions selected from feed forward pass.
    :param logdir: Where to put the tensorboard data.
    :param random_seed: Random seed for TensorFlow initializers.
    :param model_name: Name for model
    :param clip_gradients: The limit on gradient size. If 0.0 no clipping is performed.
    :param make_histograms: Whether or not to make histograms for model weights and activations
    :param best_model_path: File to save best model to during training.
    :param save_tensors: A hashmap of str:Tensor mappings. Tensors are evaluated during training. Evaluations of these tensors on best model are accessible via property :any:`evaluated_tensors`.
    :param tensorboard: Whether to make tensorboard histograms of weights and activations, and graphs of dev_error.
    :return: :any:`Model`
    """

    def __init__(self, objective, placeholderdict,
                 maxbadcount=20, momentum=None, mb=1000, verbose=True,
                 epochs=50, learnrate=0.003, save=False, opt='grad',
                 decay=[1, 1.0], evaluate=None, predictions=None,
                 logdir='log/', random_seed=None, model_name='generic',
                 clip_gradients=0.0, make_histograms=False,
                 best_model_path='/tmp/model.ckpt',
                 save_tensors={}, tensorboard=False, train_evaluate=None):

        self.objective = objective
        for t in tf.get_collection('losses'):
            self.objective += t
        self._placeholderdict = placeholderdict
        self.maxbadcount = maxbadcount
        self.momentum = momentum
        self.mb = mb
        self.verbose = verbose
        self.epochs = epochs
        self.learnrate = learnrate
        self.save = save
        self.opt = opt
        self.decay = decay
        self.epoch_times = []
        self.evaluate = evaluate
        self.train_evaluate = train_evaluate
        self._best_dev_error = float('inf')
        self.predictor = predictions
        self.random_seed = random_seed
        self.session = tf.Session()
        if self.random_seed is not None:
            tf.set_random_seed(self.random_seed)
        self.model_name = model_name
        self.clip_gradients = clip_gradients
        self.tensorboard = tensorboard
        self.make_histograms = make_histograms
        if self.make_histograms:
            self.tensorboard = True
        self.histogram_summaries = []
        if not logdir.endswith('/'):
            self.logdir = logdir + '/'
        else:
            self.logdir = logdir
        os.system('mkdir ' + self.logdir)
        self.save_tensors = save_tensors
        self._completed_epochs = 0.0
        self._best_completed_epochs = 0.0
        self._evaluated_tensors = {}
        self.deverror = []
        self._badcount = 0
        self.batch = tf.Variable(0)
        self.train_eval = []
        self.dev_spot = []
        self.train_spot = []

        # ================================================================
        # ======================For tensorboard===========================
        # ================================================================
        if tensorboard:
            self._init_summaries()

        # =============================================================================
        # ===================OPTIMIZATION STRATEGY=====================================
        # =============================================================================
        optimizer = OPT[self.opt]
        decay_step = self.decay[0]
        decay_rate = self.decay[1]
        global_step = tf.Variable(0, trainable=False) #keeps track of the mini-batch iteration

        if not (decay_step == 1 and decay_rate == 1.0):
            self.learnrate = tf.train.exponential_decay(self.learnrate, self.batch*self.mb,
                                                   decay_step, decay_rate, name='learnrate_decay')


        if self.clip_gradients > 0.0:
            params = tf.trainable_variables()
            self.gradients = tf.gradients(self.objective, params)
            if self.clip_gradients > 0.0:
                self.gradients, self.gradients_norm = tf.clip_by_global_norm(
                    self.gradients, self.clip_gradients)
            grads_and_vars = zip(self.gradients, params)
            if self.opt == 'mom':
                self.train_step = optimizer(self.learnrate,
                                       self.momentum).apply_gradients(grads_and_vars,
                                                                      global_step=self.batch,
                                                                      name="train")
            else:
                self.train_step = optimizer(self.learnrate).apply_gradients(grads_and_vars,
                                                                       global_step=self.batch,
                                                                       name="train")
        else:
            if self.opt == 'mom':
                self.train_step = optimizer(self.learnrate,
                                       self.momentum).minimize(self.objective,
                                                               global_step=self.batch)
            else:
                self.train_step = optimizer(self.learnrate).minimize(self.objective,
                                                                global_step=self.batch)

        # =============================================================================
        # ===================Initialize graph =====================================
        # =============================================================================
        self.session.run(tf.initialize_all_variables())
        if save:
            self.saver = tf.train.Saver()
            self.best_model_path = best_model_path
            self.save_path = self.saver.save(self.session, self.best_model_path)

    # ======================================================================
    # ================Properites============================================
    # ======================================================================
    @property
    def placeholderdict(self):
        '''
        Dictionary of model placeholders
        '''
        return self._placeholderdict

    @property
    def best_dev_error(self):
        """
        The best dev error reached during training.
        """
        return self._best_dev_error

    @property
    def average_secs_per_epoch(self):
        """
        The average number of seconds to complete an epoch.
        """
        return np.sum(np.array(self.epoch_times))/self._completed_epochs

    @property
    def evaluated_tensors(self):
        '''
        A dictionary of evaluations on best model for tensors and keys specified by *save_tensors* argument to constructor.
        '''
        return self._evaluated_tensors
    
    @property
    def completed_epochs(self):
        '''
        Number of epochs completed during training (fractional)
        '''
        return self._completed_epochs

    @property
    def best_completed_epochs(self):
        '''
        Number of epochs completed during at point of best dev eval during training (fractional)
        '''
        return self._best_completed_epochs

    def plot_train_dev_eval(self):
        plt.plot(self.dev_spot, self.deverror, label='dev')
        plt.plot(self.train_spot, self.train_eval, label='train')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.savefig('testfig.pdf')

    def predict(self, data, supplement=None):
        """

        :param data:  :any:`DataSet` to make predictions from.
        :return: A set of predictions from feed forward defined by :any:`self.predictions`
        """
        fd = get_feed_list(data, self.placeholderdict, supplement=supplement,
                           dropouts=tf.get_collection('dropout_prob'),
                           dropout_flag='eval')
        return self.session.run(self.predictor,
                                feed_dict=fd)

    def eval(self, tensor_in, data, supplement=None):
        """
        Evaluation of model.

        :param data: :any:`DataSet` to evaluate on.
        :return: Result of evaluating on data for :any:`self.evaluate`
        """
        fd = get_feed_list(data, self.placeholderdict, supplement=supplement,
                           dropouts=tf.get_collection('dropout_prob'),
                           dropout_flag='eval')
        return self.session.run(tensor_in, feed_dict=fd)

    def train(self, train, dev=None, supplement=None, eval_schedule='epoch', train_dev_eval_factor = 0):
        """

        :param data: :any:`DataSet` to train on.
        :return: A trained :any:`Model`
        """
        self._completed_epochs = 0.0
        if self.save:
            self.saver.restore(self.session, self.best_model_path)
        # ========================================================
        # ===========Check data to see if dev eval================
        # ========================================================
        if eval_schedule == 'epoch':
            eval_schedule = train.num_examples
        self._badcount = 0
        start_time = time.time()
        # ============================================================================================
        # =============================TRAINING=======================================================
        # ============================================================================================
        counter = 0
        train_eval_counter = 0
        while self._completed_epochs < self.epochs: # keeps track of the epoch iteration
            # ==============PER MINI-BATCH=====================================

            newbatch = train.next_batch(self.mb)
            fd = get_feed_list(newbatch, self.placeholderdict, supplement,
                               dropouts=tf.get_collection('dropout_prob'))
            self.session.run(self.train_step, feed_dict=fd)
            counter += self.mb
            train_eval_counter += self.mb
            self._completed_epochs += float(self.mb)/float(train.num_examples)
            if self.train_evaluate and train_eval_counter >= train_dev_eval_factor*eval_schedule:
                self.train_eval.append(self.eval(self.evaluate, train, supplement))
                self.train_spot.append(self._completed_epochs)
                if np.isnan(self.train_eval[-1]):
                    print("Aborting training...train evaluates to nan.")
                    break
                if self.verbose:
                    print("epoch: %f train eval: %.10f" % (self._completed_epochs, self.train_eval[-1]))
                train_eval_counter = 0

            if (counter >= eval_schedule or self._completed_epochs >= self.epochs):
                #=================PER eval_schedule==================================
                self._log_summaries(dev, supplement)
                counter = 0
                if dev:
                    self.deverror.append(self.eval(self.evaluate, dev, supplement))
                    self.dev_spot.append(self._completed_epochs)
                    if np.isnan(self.deverror[-1]):
                        print("Aborting training...dev evaluates to nan.")
                        break
                    if self.verbose:
                        print("epoch: %f dev error: %.10f" % (self._completed_epochs, self.deverror[-1]))
                    for tname in self.save_tensors:
                        self._evaluated_tensors[tname] = self.eval(self.save_tensors[tname], dev, supplement)
                        if self.verbose:
                            print("\t%s: %s" % (tname, self._evaluated_tensors[tname]))
                    # ================Early Stopping====================================
                    if self.deverror[-1] < self.best_dev_error:
                        self._badcount = 0
                        self._best_dev_error = self.deverror[-1]
                        if self.save:
                            self.save_path = self.saver.save(self.session, self.best_model_path)
                        self._best_completed_epochs = self._completed_epochs
                    else:
                        self._badcount += 1
                    if self._badcount > self.maxbadcount:
                        print('badcount exceeded: %d' % self._badcount)
                        break
                    # ==================================================================
            self.epoch_times.append(time.time() - start_time)
            start_time = time.time()


    # ================================================================
    # ======================For tensorboard===========================
    # ================================================================
    def _init_summaries(self):
        if self.make_histograms:
            self.histogram_summaries.extend(map(tf.histogram_summary,
                                      [var.name for var in tf.trainable_variables()],
                                      tf.trainable_variables()))

            self.histogram_summaries.extend(map(tf.histogram_summary,
                                      ['normalization/'+n.name for n in tf.get_collection('normalized_activations')],
                                      tf.get_collection('normalized_activations')))

            self.histogram_summaries.extend(map(tf.histogram_summary,
                                      ['activation/'+a.name for a in tf.get_collection('activation_layers')],
                                      tf.get_collection('activation_layers')))
        self.loss_summary = tf.scalar_summary('Loss', self.objective)
        self.dev_error_summary = tf.scalar_summary('dev_error', self.evaluate)
        summary_directory = os.path.join(self.logdir,
                                         self.model_name + '-' +
                                         datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self._summary_writer = tf.train.SummaryWriter(summary_directory,
                                                      self.session.graph.as_graph_def())

    def _log_summaries(self,  dev, supplement):
        fd = get_feed_list(dev, self.placeholderdict, supplement=supplement,
                           dropouts=tf.get_collection('dropout_prob'),
                           dropout_flag='eval')
        if self.tensorboard:
            if self.make_histograms:
                sum_str = self.session.run(self.histogram_summaries, fd)
                for summary in sum_str:
                    self._summary_writer.add_summary(summary, self._completed_epochs)
            loss_sum_str = self.session.run(self.loss_summary, fd)
            self._summary_writer.add_summary(loss_sum_str, self._completed_epochs)
        if dev:
            if self.tensorboard:
                dev_sum_str = self.session.run(self.dev_error_summary, fd)
                self._summary_writer.add_summary(dev_sum_str, self._completed_epochs)



