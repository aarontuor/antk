from antk.core import loader
import sys

datadir = sys.argv[1]
mb = int(sys.argv[2])
data = loader.read_data_sets(datadir, folders=['train','dev'])
data.dev.mix_after_epoch(False)
data.show()

for i in range(40):
    nb = data.dev.next_batch(mb)
    print('examples' + str(nb.num_examples))
    nb.show()
    print('index nonmix' + str(data.dev.index_in_epoch))
data.dev.mix_after_epoch(True)
for i in range(5):
    nb = data.dev.next_batch(mb)
    print('examples' + str(nb.num_examples))
    nb.show()
    print('index mix' + str(data.dev.index_in_epoch))
