import os
print('================================mf===================================')
os.system('python modelwrappers/mf.py /home/aarontuor/data/ml100k config/mf.config')
print('================================tree===================================')
os.system('python modelwrappers/tree.py /home/aarontuor/data/ml100k config/tree.config')
print('================================dnn_concat===================================')
os.system('python modelwrappers/dnn_concat.py /home/aarontuor/data/ml100k config/dnn_concat.config')
print('================================mult_dnn_concat===================================')
os.system('python modelwrappers/dnn_concat.py /home/aarontuor/data/ml100k config/mult_dnn_concat.config')
print('================================dsadd===================================')
os.system('python modelwrappers/dsadd.py /home/aarontuor/data/ml100k config/dssm.config')
print('================================dssm===================================')
os.system('python modelwrappers/dssm.py /home/aarontuor/data/ml100k config/dssm.config')
print('================================dssm_restricted===================================')
os.system('python modelwrappers/dssm_restricted.py /home/aarontuor/data/ml100k config/dssm_restricted.config')


