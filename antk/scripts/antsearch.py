import os
import random
import argparse
import numpy as np
from itertools import product
import sys


mod_globals = globals().copy()

def gridsearch(log, modulepath, module, model, arguments, argnames):

    _import_module({module: modulepath})
    for x in product(*(arguments)):
        print(x)
        mymodel = eval(model, mod_globals, locals())(*x)
        with open(log, 'a') as logfile:
            for k, v in zip(argnames, x):
                logfile.write(k + '=' + str(v) + ',')
            logfile.write('epochs=%d, '
                          'avg_secs_per_epoch=%f,'
                          'dev_error=%f\n,' % (mymodel.epoch_counter,
                                             mymodel.average_secs_per_epoch,
                                             mymodel.best_dev_error))
            os.system('rm -rf models/log/*')

def _import_module(files):
        '''
        Import node functions from modules in import parameter of constructor.
        '''
        for name in files:
            try:
                if files[name] is not None:
                    sys.path.append(files[name])
                m = __import__(name=name, globals=globals(), locals=locals(), fromlist="*")
                try:
                    attrlist = m.__all__
                except AttributeError:
                    attrlist = dir(m)
                for attr in [a for a in attrlist if '__' not in a]:
                    mod_globals[attr] = getattr(m, attr)
            except ImportError, e:
                sys.stderr.write('Unable to read %s/%s.py\n' % (files[name], name))
                sys.exit(1)