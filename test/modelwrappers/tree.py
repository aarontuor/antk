from __future__ import print_function
import tensorflow as tf
import argparse
from antk.core import config
from antk.core import generic_model
from antk.core import loader
from antk.models import tree_model

def return_parser():
    parser = argparse.ArgumentParser(description="For testing")
    parser.add_argument("datadir", metavar="DATA_DIRECTORY", type=str,
                        help="The directory where train, dev, and test data resides. ")
    parser.add_argument("config", metavar="CONFIG", type=str,
                        help="The config file for building the ant architecture.")
    parser.add_argument("-initrange", metavar="INITRANGE", type=float, default=0.00001,
                        help="A value determining the initial size of the weights.")
    parser.add_argument("-kfactors", metavar="KFACTORS", type=int, default=20,
                        help="The rank of the low rank factorization.")
    parser.add_argument("-lamb", metavar="LAMBDA", type=float, default=0.01,
                        help="The coefficient for l2 regularization")
    parser.add_argument("-mb", metavar="MINIBATCH", type=int, default=500,
                        help="The size of minibatches for stochastic gradient descent.")
    parser.add_argument("-learnrate", metavar="LEARNRATE", type=float, default=0.00001,
                        help="The stepsize for gradient descent.")
    parser.add_argument("-verbose", metavar="VERBOSE", type=bool, default=True,
                        help="Whether or not to print dev evaluations during training.")
    parser.add_argument("-maxbadcount", metavar="MAXBADCOUNT", type=int, default=20,
                        help="The threshold for early stopping.")
    parser.add_argument("-epochs", metavar="EPOCHS", type=int, default=100,
                        help="The maximum number of epochs to train for.")
    parser.add_argument("-random_seed", metavar="RANDOM_SEED", type=int, default=500,
                        help="For reproducible results.")
    parser.add_argument("-eval_rate", metavar="EVAL_RATE", type=int, default=500,
                        help="How often (in terms of number of data points) to evaluate on dev.")
    return parser

if __name__ == '__main__':

    args = return_parser().parse_args()

    data = loader.read_data_sets(args.datadir, folders=['train', 'test', 'dev', 'user', 'item'])
    data.train.labels['ratings'] = loader.center(data.train.labels['ratings'], axis=None)
    data.dev.labels['ratings'] = loader.center(data.dev.labels['ratings'], axis=None)
    data.user.features['age'] = loader.center(data.user.features['age'], axis=None)
    data.item.features['year'] = loader.center(data.item.features['year'], axis=None)
    data.user.features['age'] = loader.maxnormalize(data.user.features['age'])
    data.item.features['year'] = loader.maxnormalize(data.item.features['year'])

    x = tree_model.tree(data, args.config,
                        initrange=args.initrange,
                        kfactors=args.kfactors,
                        lamb =args.lamb,
                        mb=args.mb,
                        learnrate=args.learnrate,
                        verbose=args.verbose,
                        maxbadcount=args.maxbadcount,
                        epochs=args.epochs,
                        random_seed=args.random_seed,
                        eval_rate=args.eval_rate)
    #print stuff here to file.
