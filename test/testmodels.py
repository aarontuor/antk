import os
import argparse

def return_parser():
    parser = argparse.ArgumentParser(description="For testing")
    parser.add_argument("datadir", metavar="DATA_DIRECTORY", type=str,
                        help="The directory where train, dev, and test data resides. ")
    return parser

if __name__ == '__main__':
    args = return_parser().parse_args()
    # print('================================mf===================================')
    # os.system('python modelwrappers/mf.py ' + args.datadir + ' config/mf.config')
    # print('================================tree===================================')
    # os.system('python modelwrappers/tree.py ' + args.datadir + ' config/tree.config')
    print('================================dnn_concat===================================')
    os.system('python modelwrappers/dnn_concat.py ' + args.datadir + ' config/dnn_concat_resnet.config')
    # print('================================mult_dnn_concat===================================')
    # os.system('python modelwrappers/dnn_concat.py ' + args.datadir + ' config/mult_dnn_concat.config')
    # print('================================dsadd===================================')
    # os.system('python modelwrappers/dsadd.py ' + args.datadir + ' config/dssm.config')
    # print('================================dssm===================================')
    # os.system('python modelwrappers/dssm.py ' + args.datadir + ' config/dssm.config')
    # print('================================dssm_restricted===================================')
    # os.system('python modelwrappers/dssm_restricted.py ' + args.datadir + ' config/dssm_restricted.config')


