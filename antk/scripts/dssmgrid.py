from __future__ import print_function
import argparse
from antk.core import loader
import numpy

def return_parser():
    parser = argparse.ArgumentParser(description="Command line utility for performing grid search on a matrix factorization model.")
    parser.add_argument("datadir", type=str,
                        help="data directory for conducting search.")
    parser.add_argument("configfile", type=str,
                        help="Config file for conducting search.")
    parser.add_argument("logfile", type=str,
                        help="log file for conducting search.")
    return parser

if __name__ == '__main__':
    args = return_parser().parse_args()
    data = loader.read_data_sets(args.datadir, folders=['item', 'user', 'dev', 'test', 'train'])
    data.train.labels['ratings'] = loader.center(data.train.labels['ratings'])
    data.dev.labels['ratings'] = loader.center(data.dev.labels['ratings'])
    data.user.features['age'] = loader.center(data.user.features['age'])
    data.item.features['year'] = loader.center(data.item.features['year'])
    data.user.features['age'] = loader.max_norm(data.user.features['age'])
    data.item.features['year'] = loader.max_norm(data.item.features['year'])
    data.dev.features['time'] = loader.center(data.dev.features['time'])
    data.dev.features['time'] = loader.max_norm(data.dev.features['time'])
    data.train.features['time'] = loader.center(data.train.features['time'])
    data.train.features['time'] = loader.max_norm(data.train.features['time'])

    # x = dsmodel.dssm(data, args.configfile)
    mb = [500, 1000, 10000, 20000, 40000, 80000,50, 100, 200]
    arguments = [[data],
              [args.configfile],
              [0.00001],
              [2, 5, 10, 20, 50, 100, 200, 500, 1000],
              [0.0001, 0.001, 0.01, 0.1, 0.3, 1],
              mb,
              [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
              [True],
              map(lambda x: 10*numpy.ceil(numpy.log(data.train.num_examples / x)), mb),
              [1000],
              [500]]
    argumentnames = ['data',
              'config',
              'initrange',
              'kfactors',
              'lamb',
              'mb',
              'learnrate',
              'verbose',
              'maxbadcount',
              'epochs',
              'random_seed']
    # antsearch.gridsearch(args.logfile, '.', 'dsmodel', 'dssm', arguments, argumentnames)



