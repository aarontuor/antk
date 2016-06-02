import pytest
import os
from antk.core import config
from antk.models import mfmodel
from antk.models import dssm_model
from antk.models import tree_model
# content of test_sample.py

config_files = os.listdir('configs')
for file in config_files:
    config.testGraph(os.path.dirname(os.path.abspath(__file__)) + '/configs/' + file, '-', 'testpics/', file.split('.')[0])


dssm_model.dssm('/home/aarontuor/data/ml100k/coldsplit', 'configs/dssm.config', epochs=2)

mfmodel.mf('/home/aarontuor/data/ml100k/coldsplit', 'configs/mf.config', epochs=2)


tree_model.tree('/home/aarontuor/data/ml100k/coldsplit', 'configs/tree.config', epochs=2)

