#!/user/bin/env python
'''
normalize.py  : Text normalization script without beginning and end of sentence tokens.
authors       : Elliot Starks, Aaron Tuor, Samuel Nguyen
last modified : March, 2016
'''

import argparse
import re

def return_parser():
    parser = argparse.ArgumentParser(description='Given the path to a file, Capitalization and punctuation is removed, except for infix apostrophes, e.g. "hasn\'t", "David\'s". The normalized text is saved with "_norm" appended to the file name before the extension. The normalized text is saved in the same directory as the original text. Beginning and end of sentence tokens are not provided by this normalization script.')
    parser.add_argument('filepath', type=str,
                        help='The path to the file including filename')
    return parser

if __name__ == '__main__':
    parser = return_parser().parse_args()
    infile = open(parser.filepath, 'r')
    namestub = parser.filepath.split('.')[0].strip()
    extension = parser.filepath.split('.')[1].strip()
    newfile = namestub + '_norm.' + extension
    outfile = open(newfile, 'w')

    # Pre-compile regular expressions
    regexPunc = re.compile(r"[^A-Za-z0-9']")

    # Process each line
    for line in infile:
        line = regexPunc.sub(" ", line)#replace punctuation with space excepting apostrophes
        line = line.lower() # make everything lowercase
        wordlist = line.split(" ")#make an array of words to individually work on words

    # Process each word; write words delimited by space to file
        for word in wordlist:
            word = word.strip('\'') #yank out leading and trailing quotation marks
            word = word.strip()
            if word != '':
                outfile.write(word)
                outfile.write(' ')
        outfile.write("\n")
    outfile.close()



