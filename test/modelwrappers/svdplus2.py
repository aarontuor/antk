#!/usr/bin/env python2

from __future__ import print_function
import tensorflow as tf
import argparse
from antk.core import config
from antk.core import generic_model
from antk.core import loader
from antk.models import svdplus_model2


def return_parser():
    parser = argparse.ArgumentParser(description="For testing")
    parser.add_argument("datadir", metavar="DATA_DIRECTORY", type=str,
                        help="The directory where train, dev, and test data resides. ")
    parser.add_argument("config", metavar="CONFIG", type=str,
                        help="The config file for building the ant architecture.")
    return parser

if __name__ == '__main__':

    args = return_parser().parse_args()

    x = svdplus_model2.mf(args.datadir, args.config)
