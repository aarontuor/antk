#!/usr/bin/env python
'''February 2016 Aaron Tuor
Tool to make numpy .mat term_doc (counts), binary_term_doc (occurrence),
and tfidf_term_doc files from descriptions and dictionary files.
'''
import argparse
import os
import numpy as np
import scipy.sparse as sps
import scipy.io
from antk.core import loader

# Handle arguments
def return_parser():

    parser = argparse.ArgumentParser("Tool to make numpy .mat term_doc (counts), binary_term_doc (occurrence), "
                                     "and tfidf_term_doc files from descriptions and dictionary files.")
    parser.add_argument('datapath', type=str,
                        help='Path to folder where dictionary and descriptions are located, and created document '
                                    'term matrix will be saved.')
    parser.add_argument('dictionary', type=str,
                        help='Name of the file containing line separated words in vocabulary.')
    parser.add_argument('descriptions', type=str,
                        help='Name of the file containing line separated text descriptions.')
    parser.add_argument('doc_term_file', type=str,
                        help='Name of the file to save the created sparse document term matrix.')
    return parser

if __name__ == '__main__':
    # slash ambivalent
    slash = '/'
    if os.name == 'nt':
        slash = '\\'  # so this works in Windows
    args = return_parser().parse_args()
    if not args.datapath.endswith(slash):
        args.datapath += slash
    namestub = args.doc_term_file.split('.')[0]
    ext = args.doc_term_file.split('.')[-1]
    # make hashmap of words to Integers
    dictionaryFile = open(args.datapath + args.dictionary, 'r')
    lexicon = dictionaryFile.read().strip().split('\n')
    wordmap = {k: v for k, v in zip(lexicon, range(len(lexicon)))}
    # open read and write files
    outfile = open(args.datapath + args.doc_term_file, "w")
    descriptionFile = open(args.datapath + args.descriptions, 'r')
    # go through each line in file
    docterm = []
    for line in descriptionFile:
        countArray = [0] * len(wordmap)  # counts for words in each product description
        lineArray = line.split()
        for word in lineArray:
            if word in wordmap:
                countArray[wordmap[word]] += 1
        docterm.append(countArray)
    outfile.close()
    doc_term_matrix = sps.csr_matrix(np.array(docterm))
    binary_doc_term_matrix = doc_term_matrix.sign()
    tfidf_doc_term_matrix = tfidf(doc_term_matrix)
    print(doc_term_matrix.shape)
    print(binary_doc_term_matrix.shape)
    print(tfidf_doc_term_matrix.shape)
    scipy.io.savemat(args.datapath+args.doc_term_file, {'data': doc_term_matrix})
    scipy.io.savemat(args.datapath + namestub + '_binary' + ext, {'data': binary_doc_term_matrix})
    scipy.io.savemat(args.datapath + namestub + '_tfidf' + ext, {'data': tfidf_doc_term_matrix})