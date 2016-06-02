'''February 2016 Aaron Tuor
Make vectors of indices and values from sparse matrix format text files
'''
import argparse
import sys
import re
import math
import os
import numpy as np
import scipy.sparse as sps
import scipy.io
import random
from antk.core import loader
slash = '/'
if os.name == 'nt':
    slash = '\\'  # so this works in Windows
# Handle arguments

parser = argparse.ArgumentParser()
parser.add_argument('readpath', type=str,
                    help='Path to folder where data to be split is located.')
parser.add_argument('writepath', type=str,
                    help='Path to folder where split data is to be stored: Folder must contain, train, dev and'
                         'test directories.')
parser.add_argument('ratings',
                    help='Filename of ratings file datafiles to use in the split.')
args = parser.parse_args()
# slash ambivalent
if not args.readpath.endswith(slash):
    args.readpath += slash

if not args.writepath.endswith(slash):
    args.writepath += slash
loader.makedirs(args.writepath)
sparse = np.loadtxt(args.readpath + args.ratings)
print(sparse.shape)
num_ratings = sparse.shape[0]
order = range(num_ratings)
print(len(order))
random.shuffle(order)
sparse = sparse[order, :]
# time = loader.import_data(args.readpath + 'features_time.densetxt')
# time = time[order]

split_size = int(round(num_ratings*.1))

num_users = sparse[:,0].max()
num_items = sparse[:,1].max()

rows = sparse[:, 0] - 1
cols = sparse[:, 1] - 1
vals = sparse[:, 2]

# loader.export_data(args.writepath + 'dev/features_time.mat', time[0:split_size])
loader.export_data(args.writepath + 'dev/features_user.index', loader.HotIndex(rows[0:split_size], num_users))
loader.export_data(args.writepath + 'dev/features_item.index', loader.HotIndex(cols[0:split_size], num_items))
loader.export_data(args.writepath + 'dev/labels_ratings.mat', vals[0:split_size])

# loader.export_data(args.writepath + 'test/features_time.mat', time[split_size:split_size*2])
loader.export_data(args.writepath + 'test/features_user.index', loader.HotIndex(rows[split_size:split_size*2], num_users))
loader.export_data(args.writepath + 'test/features_item.index', loader.HotIndex(cols[split_size:split_size*2], num_items))
loader.export_data(args.writepath + 'test/labels_ratings.mat', vals[split_size:split_size*2])

# loader.export_data(args.writepath + 'train/features_time.mat', time[split_size*2:time.shape[0]])
loader.export_data(args.writepath + 'train/features_user.index', loader.HotIndex(rows[split_size*2:sparse.shape[0]],num_users))
loader.export_data(args.writepath + 'train/features_item.index', loader.HotIndex(cols[split_size*2:sparse.shape[0]], num_items))
loader.export_data(args.writepath + 'train/labels_ratings.mat', vals[split_size*2:sparse.shape[0]])

