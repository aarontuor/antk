# import os
# import random
# import argparse
# # from mfmodel import mf
# from antk.core import loader
# from itertools import product
#
# def return_parser():
#     parser = argparse.ArgumentParser(description="Command line utility for performing grid search on a matrix factorization model.")
#     parser.add_argument("datadir", type=str,
#                         help="data directory for conducting search.")
#     parser.add_argument("grid spec", type=str,
#                         help="Grid spec file for conducting search.")
#     parser.add_argument("configfile", type=str,
#                         help="Config file for model.")
#     parser.add_argument("logfile", type=str,
#                         help="log file for conducting search.")
#     return parser
#
# if __name__ == '__main__':
#     args = return_parser().parse_args()
#     data = loader.read_data_sets(args.datadir, hashlist=['item', 'user', 'ratings'])
#     data.train.labels['ratings'] = loader.center(data.train.labels['ratings'])
#     data.dev.labels['ratings'] = loader.center(data.dev.labels['ratings'])
#     lamb=[0.0001, 0.001, 0.01, 0.1, 0.3, 1]
#     kfactors=[2, 5, 10, 20, 50, 100, 200, 500, 1000]
#     learnrate=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
#     epochs=1000
#     mb=[500, 1000, 10000, 20000, 40000, 80000,50, 100, 200]
#     for m in mb:
#         for l in lamb:
#             for n in learnrate:
#                 for k in kfactors:
#                     mymodel = mf(data, args.configfile, lamb=0,
#                                  kfactors=k,
#                                  learnrate=n,
#                                  verbose=True,
#                                  epochs=epochs,
#                                  maxbadcount=int(80000/m),
#                                  mb=m,
#                                  initrange=0.0001,
#                                  )
#                     with open(args.logfile, 'a') as logfile:
#                         logfile.write('kfactors=%d,'
#                                       'learnrate=%f,'
#                                       'lamb=%f,'
#                                       'epochs=%d,'
#                                       'avg_secs_per_epoch=%f,'
#                                       'dev_error=%f,'
#                                       'minibatch=%d\n' % (k,
#                                                           n,
#                                                           l,
#                                                           mymodel.epoch_counter,
#                                                           mymodel.average_secs_per_epoch,
#                                                           mymodel.best_dev_error,
#                                                           m))
#                     os.system('rm -rf log/*')