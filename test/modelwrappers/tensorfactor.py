from __future__ import print_function
import tensorflow as tf
import argparse
from antk.core import config
from antk.core import generic_model
from antk.core import loader
from antk.models import tensor_factor_model


def return_parser():
    parser = argparse.ArgumentParser(description="For testing")
    parser.add_argument("datadir", metavar="DATA_DIRECTORY", type=str,
                        help="The directory where train, dev, and test data resides. ")
    # parser.add_argument("config", metavar="CONFIG", type=str,
    #                     help="The config file for building the ant architecture.")
    parser.add_argument("logfile", metavar="LOGFILE", type=str,
                        help="For recording results.")
    parser.add_argument("-initrange", metavar="INITRANGE", type=float, default=0.0001,
                        help="A value determining the initial size of the weights.")
    parser.add_argument("-iembed", metavar="IEMBED", type=int, default=50,
                        help="The size of item embeddings.")
    parser.add_argument("-uembed", metavar="UEMBED", type=int, default=50,
                        help="The rank of the low rank factorization.")
    parser.add_argument("-gembed", metavar="GEMBED", type=int, default=50,
                        help="The rank of the low rank factorization.")
    parser.add_argument("-lamb", metavar="LAMBDA", type=float, default=1,
                        help="The coefficient for l2 regularization")
    parser.add_argument("-mb", metavar="MINIBATCH", type=int, default=500,
                        help="The size of minibatches for stochastic gradient descent.")
    parser.add_argument("-learnrate", metavar="LEARNRATE", type=float, default=0.00003,
                        help="The stepsize for gradient descent.")
    parser.add_argument("-verbose", metavar="VERBOSE", type=bool, default=True,
                        help="Whether or not to print dev evaluations during training.")
    parser.add_argument("-maxbadcount", metavar="MAXBADCOUNT", type=int, default=20,
                        help="The threshold for early stopping.")
    parser.add_argument("-epochs", metavar="EPOCHS", type=int, default=10,
                        help="The maximum number of epochs to train for.")
    parser.add_argument("-random_seed", metavar="RANDOM_SEED", type=int, default=None,
                        help="For reproducible results.")
    parser.add_argument("-eval_rate", metavar="EVAL_RATE", type=int, default=10000,
                        help="How often (in terms of number of data points) to evaluate on dev.")
    return parser

if __name__ == '__main__':

    args = return_parser().parse_args()

    x = tensor_factor_model.tensorfactor(args.datadir,
                        initrange=args.initrange,
                        uembed=args.uembed,
                        iembed=args.iembed,
                        gembed=args.gembed,
                        lamb =args.lamb,
                        mb=args.mb,
                        learnrate=args.learnrate,
                        verbose=args.verbose,
                        maxbadcount=args.maxbadcount,
                        epochs=args.epochs,
                        random_seed=args.random_seed,
                        eval_rate=args.eval_rate)

    with open(args.logfile, 'a') as out:

        x_err = x._best_dev_error
        if x_err > 100 or x_err == float('inf') or x_err == float('nan'):
            x_err = 100

        out.write("lamb=%f uembed=%d iembed=%d gembed=%d learnrate=%f mb=%d initrange=%f epochs=%f "
                  "avspe=%f bestdev=%f\n"
                  % (args.lamb, args.uembed, args.iembed, args.gembed, args.learnrate, args.mb,
                     args.initrange, x.completed_epochs, x.average_secs_per_epoch, x_err))
