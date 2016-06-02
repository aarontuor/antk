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
import collections
import math
import random

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(12734 - 1))
    print(count)
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
            data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

# Handle arguments
def return_parser():
    parser = argparse.ArgumentParser(description="For getting word counts")
    parser.add_argument('descriptions', type=str,
                        help='Name of the file containing line separated text descriptions.')
    return parser

if __name__ == '__main__':
    args = return_parser().parse_args()
    with open(args.descriptions) as data:
        words = data.read().split()
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    del words  # Hint to reduce memory.
    print('Sample data', data[:10])
    labels = [reverse_dictionary[i] for i in range(len(dictionary))]
    with open('labels.txt', 'w') as l:
        for label in labels:
            l.write(label+'\n')
