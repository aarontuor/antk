import os, time
import numpy as np

"""
Example script for piping between processes without writing to files for live stream. Probably using sockets would
be a better choice for a mock timing trial for this end of the pipeline.
"""

def derive_features(stream_name, pipe_name):
    '''
    Place holder for feature derivation function (could be a class like BatchBuffer below)
    :param stream_name: Where data is coming from (in this case it is a file. Could be a socket or pipe)
    :param pipe_name: Whatever the pipe is called
    '''
    pipeout = os.open(pipe_name, os.O_WRONLY)
    with open(stream_name, 'r') as stream:
        for line in stream:
            # Feature derivation here
            os.write(pipeout, line)
            print('derive_features wrote %s to pipe.' % line)
            # to see what is going on
            time.sleep(0.1)
    os.write(pipeout, 'X\n')

class BatchBuffer():
    '''For reading and buffering a stream of input data'''

    def __init__(self, pipe_name, batch_size):
        '''

        :param pipe_name: Name of pipe to read from
        :param batch_size: Number of time steps to send
        :return: A BatchBuffer object that reads from the named pipe
        '''

        self.pipe_name = pipe_name
        self.pipein = open(pipe_name, 'r')
        self.batch_size = batch_size
        self.batch_count = 0

    def next_batch(self):
        '''
        :return: Next batch_size time steps from data stream
        '''
        batch_text = ''
        for i in range(self.batch_size):
            batch_text += self.pipein.readline()[:-1]

        if batch_text.endswith('X'):
            print('Reached end of file.')
            return None
        else:
            # Here is where you make the appropriate arrangement to hand to the model you are using
            batch = np.asarray(np.matrix(batch_text).reshape(self.batch_size, -1))
            self.batch_count += 1
            print('Returning batch %s\n%s' % (self.batch_count, batch))
            return batch

pipe_name = 'file_to_batcher'
if not os.path.exists(pipe_name):
    os.mkfifo(pipe_name)
pid = os.fork()

if pid != 0:
    data_stream = BatchBuffer(pipe_name, 5)
    while(True):
        mat = data_stream.next_batch()
        if mat is None:
            break
else:
    derive_features('rat.csv', pipe_name)