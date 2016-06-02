#!/usr/bin/env python
'''Jan 2016 Aaron Tuor
Tool to process Movielens 100k Metadata
'''
import argparse
import numpy
import scipy.sparse as sps
import os
import scipy.io
from antk.core import loader

def return_parser():
    parser = argparse.ArgumentParser(description="Tool to process Movielens 100k user Metadata.")
    parser.add_argument('datapath', type=str, help='Path to ml-100k')
    parser.add_argument('outpath', type=str, help='Path to save created files to.')
    return parser

if __name__ == '__main__':
    slash = '/'
    if os.name == 'nt':
        slash = '\\'  # so this works in Windows

    args = return_parser().parse_args()
    if not args.datapath.endswith(slash):
        args.datapath += slash
    if not args.outpath.endswith(slash):
        args.outpath += slash
    with open(args.datapath + 'u.occupation', 'r') as infile:
        occlist = infile.read()
        occlist = occlist.split('\n')
        occmap = {k: v for k, v in zip(occlist, range(len(occlist)))}
    ages = []
    sexes = []
    occupations = []
    zipcodes = []
    with open(args.datapath + 'u.user') as infile:
        first = False
        lines = infile.readlines()
        for line in lines:
            user_feature_list = []
            line = line.split('|')
            # age==================================
            ages.append(float(line[1]))
            # sex==================================
            if line[2] == 'M':
                sexes.append(1)
            else:
                sexes.append(0)
            # occupation==================================
            occupations.append(occmap[line[3]])
            # first three zip code digits: section center facility (regional mail sorting centers)
            if line[4][0:3].isdigit():
                zipcodes.append(int(line[4][0:3]))
            else:
                zipcodes.append(0)

    loader.export_data(args.outpath + 'features_sex.index', loader.HotIndex(numpy.array(sexes), 2))
    loader.export_data(args.outpath + 'features_occ.index', loader.HotIndex(numpy.array(occupations), len(occlist) -1))
    loader.export_data(args.outpath + 'features_zip.index', loader.HotIndex(numpy.array(zipcodes), 1000))
    loader.export_data(args.outpath + 'features_age.mat', numpy.array(ages))

    # inspect processed data
    sex = loader.imatload(args.outpath + 'features_sex.index')
    print('sex:')
    print(sex.vec.shape, sex.dim)
    print(sex.vec[0:10])
    print('\n')

    occ = loader.imatload(args.outpath + 'features_occ.index')
    print('occ:')
    print(occ.vec.shape, occ.dim)
    print(occ.vec[0:10])
    print('\n')

    zip = loader.imatload(args.outpath + 'features_zip.index')
    print('zip:')
    print(zip.vec.shape, zip.dim)
    print(zip.vec[0:10])
    print('\n')

    age = loader.import_data(args.outpath + 'features_age.mat')
    print('age:')
    print(age.shape)
    print(age[0:10])