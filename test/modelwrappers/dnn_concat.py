import argparse
from antk.core import loader
from antk.models import dnn_concat_model

def return_parser():
    parser = argparse.ArgumentParser(description="For testing")
    parser.add_argument("datadir", metavar="DATA_DIRECTORY", type=str,
                        help="The directory where train, dev, and test data resides. ")
    parser.add_argument("config", metavar="CONFIG", type=str,
                        help="The config file for building the ant architecture.")
    parser.add_argument("-layers", metavar="LAYERS", nargs='+',
                        type=int, default=[16,16,16,8,8,8],
                        help="A list of hidden layer sizes.")
    parser.add_argument("-initrange", metavar="INITRANGE", type=float,
                        default=1e-5,
                        help="Range to initialize embedding vectors.")
    parser.add_argument("-act", metavar="ACTIVATION", type=str,
                        default='tanhlecun',
                        help="The hidden layer activation. May be 'tanh', 'sigmoid', 'tanhlecun', 'relu', 'relu6'.")
    parser.add_argument("-bn", metavar="BATCH_NORMALIZATION", type=bool,
                        default=True,
                        help="Whether or not to use batch normalization on neural net layers.")
    parser.add_argument("-kp", metavar="KEEP_PROB",
                        type=float, default="0.95",
                        help="The keep probability for drop out.")
    parser.add_argument("-cs", metavar="CONCAT_SIZE", type=int, default=24,
                        help="Size of layer after concatenation operation")
    parser.add_argument("-uembed", metavar="UEMBED", type=int, default=32,
                        help="Size of user embeddings.")
    parser.add_argument("-iembed", metavar="IEMBED", type=int, default=32,
                        help="Size of item embeddings.")
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
    parser.add_argument("-eval_rate", metavar="EVAL_RATE", type=int, default=500,
                        help="How often (in terms of number of data points) to evaluate on dev.")
    return parser

if __name__ == '__main__':

    args = return_parser().parse_args()
    data = loader.read_data_sets(args.datadir, hashlist=['user', 'item', 'ratings'],
                                 folders=['train', 'test', 'dev'])
    data.train.labels['ratings'] = loader.center(data.train.labels['ratings'])
    data.dev.labels['ratings'] = loader.center(data.dev.labels['ratings'])
    x = dnn_concat_model.dnn_concat(data, args.config,
                        layers=args.layers,
                        activation=args.act,
                        initrange=args.initrange,
                        bn=args.bn,
                        keep_prob=args.kp,
                        concat_size=args.cs,
                        uembed=args.uembed,
                        iembed=args.iembed,
                        mb=args.mb,
                        learnrate=args.learnrate,
                        verbose=args.verbose,
                        maxbadcount=args.maxbadcount,
                        epochs=args.epochs,
                        eval_rate=args.eval_rate)
#print stuff to file here
