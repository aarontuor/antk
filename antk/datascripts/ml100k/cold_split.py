#!/usr/bin/env python
'''February 2016 Aaron Tuor
Tool to make 5/5/5/5/80, coldDev, coldTest, dev, test, train datasplit for movielens 100k with imdb plot descriptions.
'''
import argparse
import os
import scipy.sparse as sps
import scipy.io
from antk.core import loader
import random
import numpy as np

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
parser.add_argument('-item_list', default=[], nargs='+',
                    help='One or more item datafiles to use in the split.')
parser.add_argument('-user_list', default=[], nargs='+',
                    help='List of user datafiles to use in the split.')
args = parser.parse_args()

# slash ambivalent
if not args.readpath.endswith(slash):
    args.readpath += slash
if not args.writepath.endswith(slash):
    args.writepath += slash

loader.makedirs(args.writepath, cold=True)

# read utility matrix and independently shuffle rows and columns
ratings = loader.import_data(args.readpath + args.ratings)
num_users = ratings.shape[0]
num_items = ratings.shape[1]
num_ratings = ratings.getnnz()
item_order = range(num_items)
user_order = range(num_users)
random.shuffle(item_order)
random.shuffle(user_order)

# save shuffled orders
scipy.io.savemat(args.writepath+'item_order.mat', {'data': np.array(item_order)})
scipy.io.savemat(args.writepath+'user_order.mat', {'data': np.array(user_order)})
ratings = ratings[:, item_order]
ratings = ratings[user_order, :]
split_size = num_ratings*0.05

# find item test_cold set
i = num_items
num_cold_test_items = 0
while num_cold_test_items < split_size:
    i = i - 1
    num_cold_test_items = num_cold_test_items + ratings[:, i].getnnz()
item_start_test_cold = i

# find item dev_cold set
num_cold_dev_items = 0
item_end_dev_cold = i
while num_cold_dev_items < split_size:
    i = i - 1
    num_cold_dev_items = num_cold_dev_items + ratings[:, i].getnnz()
item_start_dev_cold = i

#======================================================================
# find user test_cold set
i = num_users
num_cold_test = 0
while num_cold_test < split_size:
    i = i - 1
    num_cold_test = num_cold_test + ratings[i, :].getnnz()
user_start_test_cold = i
# ------------------------------------------------------------

# find user dev_cold set
num_cold_dev = 0
end_dev_cold = i
while num_cold_dev < split_size:
    i = i - 1
    num_cold_dev = num_cold_dev + ratings[i, :].getnnz()
user_start_dev_cold = i
# --------------------------------------------------------------

#========================================================================================
#========================================================================================
def toOnehot(X, dim):
    #empty one-hot matrix
    hotmatrix = np.zeros((X.shape[0], dim))
    #fill indice positions
    hotmatrix[np.arange(X.shape[0]), X] = 1
    hotmatrix = sps.csr_matrix(hotmatrix)
    return hotmatrix
#set target idx to 1
#save cold sets
item_test_cold = ratings[0:user_start_dev_cold, item_start_test_cold:num_items]
itemtestcoldusers, itemtestcolditems, itemtestcoldvalues = sps.find(item_test_cold)

scipy.io.savemat(args.writepath+'test_cold_item/features_user.mat', {'data': toOnehot(itemtestcoldusers, item_test_cold.shape[0])})
scipy.io.savemat(args.writepath+'test_cold_item/features_item.mat', {'data': toOnehot(itemtestcolditems, item_test_cold.shape[1])})
scipy.io.savemat(args.writepath+'test_cold_item/labels_values.mat', {'data': itemtestcoldvalues})

item_dev_cold = ratings[0:user_start_dev_cold, item_start_dev_cold:item_end_dev_cold]
itemdevcoldusers, itemdevcolditems, itemdevcoldvalues = sps.find(item_dev_cold)
scipy.io.savemat(args.writepath+'dev_cold_item/features_user.mat', {'data': toOnehot(itemdevcoldusers, item_dev_cold.shape[0])})
scipy.io.savemat(args.writepath+'dev_cold_item/features_item.mat', {'data': toOnehot(itemdevcolditems, item_dev_cold.shape[1])})
scipy.io.savemat(args.writepath+'dev_cold_item/labels_values.mat', {'data': itemdevcoldvalues})

user_test_cold = ratings[user_start_test_cold:num_users, 0:item_start_dev_cold]
usertestcoldusers, usertestcolditems, usertestcoldvalues = sps.find(user_test_cold)
scipy.io.savemat(args.writepath+'test_cold_user/features_user.mat', {'data': toOnehot(usertestcoldusers, user_test_cold.shape[0])})
scipy.io.savemat(args.writepath+'test_cold_user/features_item.mat', {'data': toOnehot(usertestcolditems, user_test_cold.shape[1])})
scipy.io.savemat(args.writepath+'test_cold_user/labels_values.mat', {'data': usertestcoldvalues})

user_dev_cold = ratings[user_start_dev_cold:end_dev_cold, 0:item_start_dev_cold]
userdevcoldusers, userdevcolditems, userdevcoldvalues = sps.find(user_dev_cold)
scipy.io.savemat(args.writepath+'dev_cold_user/features_user.mat', {'data': toOnehot(userdevcoldusers, user_dev_cold.shape[0])})
scipy.io.savemat(args.writepath+'dev_cold_user/features_item.mat', {'data': toOnehot(userdevcolditems, user_dev_cold.shape[1])})
scipy.io.savemat(args.writepath+'dev_cold_user/labels_values.mat', {'data': userdevcoldvalues})

both_cold = ratings[user_start_dev_cold:num_users, item_start_dev_cold: num_items]
bothcoldusers, bothcolditems, bothcoldvalues = sps.find(both_cold)
scipy.io.savemat(args.writepath+'both_cold/features_user.mat', {'data': toOnehot(bothcoldusers, both_cold.shape[0])})
scipy.io.savemat(args.writepath+'both_cold/features_item.mat', {'data': toOnehot(bothcolditems, both_cold.shape[1])})
scipy.io.savemat(args.writepath+'both_cold/labels_values.mat', {'data': bothcoldvalues})


# split
[i, j, v] = sps.find(ratings[0:user_start_dev_cold, 0:item_start_dev_cold])
num_ratings = ratings[0:user_start_dev_cold, 0:item_start_dev_cold].getnnz()
order = range(i.shape[0])
random.shuffle(order)
i = i[order]
j = j[order]
v = v[order]
# find dev set
split_size = int(round(split_size))
dev_users = i[range(split_size)]
dev_items = j[range(split_size)]
scipy.io.savemat(args.writepath+'dev/features_user.mat', {'data': toOnehot(dev_users, num_users)})
scipy.io.savemat(args.writepath+'dev/features_item.mat', {'data': toOnehot(dev_items, num_items)})
scipy.io.savemat(args.writepath+'dev/labels_values.mat', {'data': v[range(split_size)]})

# find test set
test_start = split_size
test_end = 2*split_size
test_users = i[test_start:test_end]
test_items = j[test_start:test_end]
scipy.io.savemat(args.writepath+'test/features_user.mat', {'data': toOnehot(test_users, num_users)})
scipy.io.savemat(args.writepath+'test/features_item.mat', {'data': toOnehot(test_items, num_items)})
scipy.io.savemat(args.writepath+'test/labels_values.mat', {'data': v[test_start:test_end]})

# find train set
train_start = test_end
train_users = i[range(train_start, num_ratings)]
train_items = j[range(train_start, num_ratings)]
scipy.io.savemat(args.writepath+'train/features_user.mat', {'data': toOnehot(train_users, num_users)})
scipy.io.savemat(args.writepath+'train/features_item.mat', {'data': toOnehot(train_items, num_items)})
scipy.io.savemat(args.writepath+'train/labels_values.mat', {'data': v[train_start:num_ratings]})


 numnoncoldratings = v[range(split_size)].shape[0] + v[range(test_start, test_end)].shape[0] + v[range(train_start, num_non_cold)].shape[0]
 assert(numcoldratings + numnoncoldratings == num_ratings)

# split extra user and item features
print('gothere')
for u in args.user_list:
    namestub = os.path.splitext(os.path.basename(u))[0]
    matrix = loader.import_data(args.readpath + u)
    scipy.io.savemat(args.writepath+'train/features_'+namestub+'.mat',
                     {'data': matrix[train_users.flatten()]})
    scipy.io.savemat(args.writepath+'dev/features_'+namestub+'.mat',
                     {'data': matrix[dev_users.flatten()]})
    scipy.io.savemat(args.writepath+'test/features_'+namestub+'.mat',
                     {'data': matrix[test_users.flatten()]})
    scipy.io.savemat(args.writepath+'test_cold_item/features_'+namestub+'.mat',
                     {'data': matrix[itemtestcoldusers.flatten()]})
    scipy.io.savemat(args.writepath+'test_cold_user/features_'+namestub+'.mat',
                     {'data': matrix[usertestcoldusers.flatten()]})
    scipy.io.savemat(args.writepath+'dev_cold_item/features_'+namestub+'.mat',
                     {'data': matrix[itemdevcoldusers.flatten()]})
    scipy.io.savemat(args.writepath+'dev_cold_user/features_'+namestub+'.mat',
                     {'data': matrix[userdevcoldusers.flatten()]})
    scipy.io.savemat(args.writepath+'both_cold/features_'+namestub+'.mat',
                     {'data': matrix[bothcoldusers.flatten()]})

for i in args.item_list:
    print(np.max(train_items))
    namestub = os.path.splitext(os.path.basename(i))[0]
    matrix = loader.import_data(args.readpath + i)
    scipy.io.savemat(args.writepath+'train/features_'+namestub+'.mat',
                     {'data': matrix[train_items.flatten()]})
    scipy.io.savemat(args.writepath+'dev/features_'+namestub+'.mat',
                     {'data': matrix[dev_items.flatten()]})
    scipy.io.savemat(args.writepath+'test/features_'+namestub+'.mat',
                     {'data': matrix[test_items.flatten()]})
    scipy.io.savemat(args.writepath+'test_cold_item/features_'+namestub+'.mat',
                     {'data': matrix[itemtestcolditems.flatten()]})
    scipy.io.savemat(args.writepath+'test_cold_user/features_'+namestub+'.mat',
                     {'data': matrix[usertestcolditems.flatten()]})
    scipy.io.savemat(args.writepath+'dev_cold_item/features_'+namestub+'.mat',
                     {'data': matrix[itemdevcolditems.flatten()]})
    scipy.io.savemat(args.writepath+'dev_cold_user/features_'+namestub+'.mat',
                     {'data': matrix[userdevcolditems.flatten()]})
    scipy.io.savemat(args.writepath+'both_cold/features_'+namestub+'.mat',
                     {'data': matrix[bothcolditems.flatten()]})

