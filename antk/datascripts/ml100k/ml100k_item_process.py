#!/usr/bin/env python
'''March 2016 Aaron Tuor
Tool to process Movielens 100k Metadata
'''
import argparse
import numpy
import scipy.sparse as sps
import os
import scipy.io
from antk.core import loader

def return_parser():
    parser = argparse.ArgumentParser(description="Reads MovieLens 100k item meta data and converts to feature files. \n"
                                                 "features_item_month.index: "
                                                    "The produced files are: \n"
                                                 "A file storing a :any:`HotIndex` object of movie month releases. \n"
                                                 "\n\tfeatures_item_year.mat: "
                                                    "A file storing a numpy array of movie year releases.\n"
                                                 "\n\tfeatures_item_genre.mat: "
                                                    "A file storing a scipy sparse csr_matrix of one hot encodings "
                                                    "for movie genre.")
    parser.add_argument('datapath', type=str,
                        help='The path to ml-100k dataset. Usually "some_relative_path/ml-100k')
    parser.add_argument('outpath', type=str, default='',
                        help='The path to the folder to store the processed Movielens 100k item data feature files.')
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
    monthlist = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthmap = {k: v for k, v in zip(monthlist, range(12))}
    month = []
    year = []
    genre = []

    with open(args.datapath + 'u2.item') as infile:
        first = False
        lines = infile.readlines()
        for line in lines:
            line = line.split('|')
            # month released==================================
            date = line[2].split('-')
            month.append(monthmap[date[1]])
            # year released==================================
            year.append(float(int(date[2])))
            genres = line[5:len(line)]
            for i in range(len(genres)):
                genres[i] = float(genres[i])
            genre.append(genres)

    # print(month_matrix.vec.shape, month_matrix.dim, genre_matrix.vec.shape, genre_matrix.dim, year_matrix.shape)
    loader.export_data(args.outpath + 'features_month.index', loader.HotIndex(numpy.array(month), 12))
    loader.export_data(args.outpath + 'features_year.mat', numpy.array(year))
    loader.export_data(args.outpath + 'features_genre.mat', numpy.array(genre))

    # inspect processed data
    month = loader.import_data(args.outpath + 'features_month.index')
    print('month:')
    print(month.vec.shape, month.dim)
    print(month.vec[0:20])
    print('\n')

    year = loader.import_data(args.outpath + 'features_year.mat')
    print('year:')
    print(year.shape)
    print(year[0:10])
    print('\n')

    genre = loader.import_data(args.outpath + 'features_genre.mat')
    print('genre:')
    print(genre.shape)
    print(genre[0:10])
    print('\n')
