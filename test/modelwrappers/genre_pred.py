from __future__ import print_function
import tensorflow as tf
from antk.core import loader
from antk.core import generic_model
from antk.core import node_ops
import argparse

def return_parser():
    parser = argparse.ArgumentParser(description="For testing")
    parser.add_argument("datadir", metavar="DATA_DIRECTORY", type=str,
                        help="The directory where train, dev, and test data resides. ")
    # parser.add_argument("config", metavar="CONFIG", type=str,
    #                     help="The config file for building the ant architecture.")
    parser.add_argument("-threshold", metavar="THRESHOLD", type=float, default=0.1,
                        help="The threshold for determining if an item is in a class")
    parser.add_argument("-mb", metavar="MINIBATCH", type=int, default=500,
                        help="The size of minibatches for stochastic gradient descent.")
    parser.add_argument("-learnrate", metavar="LEARNRATE", type=float, default=0.001,
                        help="The stepsize for gradient descent.")
    parser.add_argument("-verbose", metavar="VERBOSE", type=bool, default=True,
                        help="Whether or not to print dev evaluations during training.")
    parser.add_argument("-maxbadcount", metavar="MAXBADCOUNT", type=int, default=20,
                        help="The threshold for early stopping.")
    parser.add_argument("-epochs", metavar="EPOCHS", type=int, default=100,
                        help="The maximum number of epochs to train for.")
    parser.add_argument("-random_seed", metavar="RANDOM_SEED", type=int,
                        help="For reproducible results.")
    parser.add_argument("-eval_rate", metavar="EVAL_RATE", type=int, default=500,
                        help="How often (in terms of number of data points) to evaluate on dev.")
    return parser

args = return_parser().parse_args()
data = loader.read_data_sets(args.datadir, hashlist=['genre', 'words'])
data.show()
x = tf.placeholder(tf.float32, [None, 12734])
y = node_ops.dnn(x, [100, 100, 100, 100, 19], activation='sigmoid', name='dist', initrange=20)
y_ = tf.placeholder(tf.float32, [None, 19])

objective = node_ops.other_cross_entropy(y, y_)
detection = node_ops.detection(y, args.threshold)

# recall percentage of actual genres detected
recall = node_ops.recall(y, y_, detects=detection)
# precision: percentage of the genres we predicted that were correct
precision = node_ops.precision(y, y_, detects=detection)
fscore = node_ops.fscore(precisions=precision, recalls=recall)
placeholderdict = {'words': x, 'genre': y_}
model = generic_model.Model(objective, placeholderdict,
                                mb=args.mb,
                                learnrate=args.learnrate,
                                verbose=args.verbose,
                                maxbadcount=args.maxbadcount,
                                epochs=100,
                                evaluate=objective,
                                predictions=detection,
                                model_name='genrepred',
                                random_seed=500,
                                save_tensors={'fscore': fscore, 'precision': precision, 'recall': recall, 'dist': tf.get_collection('activation_layers')[0]})
z = model.train(data.train, dev=data.dev, eval_schedule=args.eval_rate)
