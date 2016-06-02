# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import matplotlib.pyplot as plt
import tensorflow as tf


# Read the data into a string.
def read_data(filename):
    """
    :param filename: A zip file to open and read from
    :return: A list of the space delimited tokens from the textfile.
    """
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return f.read(name).split()
    f.close()


def build_dataset(words, vocabulary_size):
    """
    :param words: A list of word tokens from a text file
    :param vocabulary_size: How many word tokens to keep.
    :return: data (text transformed into list of word ids 'UNK'=0), count (list of pairs (word:word_count) indexed by word id), dictionary (word:id hashmap), reverse_dictionary (id:word hashmap)
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(data, batch_size, num_skips, skip_window):
    """
    :param data: list of word ids corresponding to text
    :param batch_size: Size of batch to retrieve
    :param num_skips: How many times to reuse an input to generate a label.
    :param skip_window: How many words to consider left and right.
    :return:
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


# Step 4: Build and train a skip-gram model.
class SkipGramVecs(object):
    """
    Trains a skip gram model from `Distributed Representations of Words and Phrases and their Compositionality`_

    :param textfile: Plain text file or zip file with plain text files.
    :param vocabulary_size: How many words to use from text
    :param batch_size: mini-batch size
    :param embedding_size: Dimension of the embedding vector.
    :param skip_window: How many words to consider left and right.
    :param num_skips: How many times to reuse an input to generate a label.
    :param valid_size: Random set of words to evaluate similarity on.
    :param valid_window: Only pick dev samples in the head of the distribution.
    :param num_sampled: Number of negative examples to sample.
    :param num_steps: How many mini-batch steps to take
    :param verbose: Whether to calculate and print similarities for a sample of words
    """

    def __init__(self, textfile, vocabulary_size=12735,
                 batch_size=128, embedding_size=128, skip_window=1, num_skips=2,
                 valid_size=16, valid_window=100, num_sampled=64, num_steps=100000,
                 verbose=False):

        if not textfile.endswith('.zip'):
            ziptext = textfile + '.zip'
            os.system('zip ' + ziptext + ' ' + textfile)
            textfile = ziptext
        words = read_data(textfile)
        print('Data size', len(words))

        self.data, self.count, self.dictionary, self.reverse_dictionary = build_dataset(words, vocabulary_size)
        del words  # Hint to reduce memory.

        batch, labels = generate_batch(self.data, batch_size=8, num_skips=2, skip_window=1)
        for i in range(8):
            print(batch[i], '->', labels[i, 0])
            print(self.reverse_dictionary[batch[i]], '->', self.reverse_dictionary[labels[i, 0]])

        # We pick a random validation set to sample nearest neighbors. Here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))

        graph = tf.Graph()

        with graph.as_default():

            # Input data.
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                embeddings = tf.Variable(
                        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                        tf.truncated_normal([vocabulary_size, embedding_size],
                                            stddev=1.0 / math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            loss = tf.reduce_mean(
                    tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                                   num_sampled, vocabulary_size))

            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                    normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                    valid_embeddings, normalized_embeddings, transpose_b=True)

        with tf.Session(graph=graph) as session:
            # We must initialize all variables before we use them.
            tf.initialize_all_variables().run()

            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = generate_batch(
                        self.data, batch_size, num_skips, skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if verbose and step % 10000 == 0:
                    sim = similarity.eval()
                    for i in xrange(valid_size):
                        valid_word = self.reverse_dictionary[valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in xrange(top_k):
                            close_word = self.reverse_dictionary[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
            self.final_embeddings = normalized_embeddings.eval()

    def plot_embeddings(self, filename='tsne.png', num_terms=500):
            """
            Plot tsne reduction of learned word embeddings in 2-space.

            :param filename: File to save plot to.
            :param num_terms: How many words to plot.
            """
            plot_tsne(self.final_embeddings,
                      [self.reverse_dictionary[i] for i in xrange(num_terms)],
                      filename, num_terms)

def plot_tsne(embeddings, labels, filename='tsne.png', num_terms=500):
    """
    Makes tsne plot to visualize word embeddings. Need sklearn, matplotlib for this to work.

    :param filename: Location to save labeled tsne plots
    :param num_terms: Num of words to plot
    """
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:num_terms, :])
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig(filename)
    except ImportError:
        print("Please install sklearn and matplotlib to visualize embeddings.")
