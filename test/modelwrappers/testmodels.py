
from __future__ import print_function
import argparse
from antk.core import loader
from antk.models import mfmodel
from antk.models import tree_model
from antk.models import dssm_model
from antk.models import dssm_restricted_model
from antk.models import dsaddmodel
from antk.models import dnn_concat_model

def return_parser():
    parser = argparse.ArgumentParser(description="For testing")
    parser.add_argument("datadir", metavar="DATA_DIRECTORY", type=str,
                        help="The directory where train, dev, and test data resides. ")
    return parser

if __name__ == '__main__':

    args = return_parser().parse_args()
    data = loader.read_data_sets(args.datadir, folders=['train', 'dev', 'user', 'item'])
    data.train.labels['ratings'] = loader.center(data.train.labels['ratings'])
    data.dev.labels['ratings'] = loader.center(data.dev.labels['ratings'])
    data.user.features['age'] = loader.center(data.user.features['age'], axis=None)
    data.item.features['year'] = loader.center(data.item.features['year'], axis=None)
    data.user.features['age'] = loader.maxnormalize(data.user.features['age'])
    data.item.features['year'] = loader.maxnormalize(data.item.features['year'])
    data.show()
    print('=================mfmodel============================')
    x = mfmodel.mf(data, 'mf.config', epochs=1)
    print('=================treemodel============================')
    x2 = tree_model.tree(data, 'tree.config', epochs=1)
    print('=================dssmmodel============================')
    x3 = dssm_model.dssm(data, 'dssm.config', epochs=1)
    print('=================dnnconcat============================')
    x4 = dnn_concat_model.dnn_concat(data, 'dnn_concat.config', epochs=1)
    print('=================mult_dnn_concat============================')
    x5 = dnn_concat_model.dnn_concat(data, 'dnn_mult_concat.config', epochs=1)
    print('=================dsadd============================')
    x5 = dsaddmodel.dsadd(data, 'dssm.config', epochs=1)
    print('=================dssmrestricted============================')
    x6 = dssm_restricted_model(data, 'dssm.config', epochs=1)
