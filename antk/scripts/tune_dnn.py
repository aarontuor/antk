import os
import random
import argparse


#Take in eithter a 1(error) or 0(success)
#Terminate the whole script if too many errors in a row occur
def update_failure_counter(is_failure, num_failures):

    if(is_failure):
        print('TensorFlow Error Occured.')
        num_failures += 1
    else:
        num_failures = 0

    if(num_failures == 10):
        exit(1)

    return num_failures


parser = argparse.ArgumentParser()

parser.add_argument('-model_location', dest='model_location', type=str, required=True)
parser.add_argument('-datadir', dest='datadir', type=str, required=True)
parser.add_argument('-out_file', dest='out_file', type=str, required=True)
parser.add_argument('-range', type=int, default=100)


args = parser.parse_args()



num_failures = 0

for run_num in range(0,9999999):

    activations = ['-sig', '-tanh', '-relu', '-relu6']
    optimizers = ['-softplus', '-adam', '-ada', '-grad', '-mom']

    nlayers = random.randint(1,10)
    nunits = random.randint(5,1000)
    learnrate = random.uniform(0.00009, 0.001)
    mb = random.randint(1,10000)
    keep_prob = random.uniform(0.0,1.0)
    decay_step = random.randint(1, 50)
    decay = random.uniform(0.9, 1.0)
    activation = activations[random.randint(0,3)] #Choose activation from above list
    optimizer = optimizers[random.randint(0,4)] #Choose optimizer from above list

    line = 'python ' + args.model_location + ' -datadir ' + args.datadir + ' -nlayers ' + str(nlayers) + ' -nunits ' + str(nunits) + ' -learnrate ' + str(learnrate) + ' -mb ' + str(mb) + ' -keep_prob ' + str(keep_prob) + ' -decay ' + str(decay_step) + ' ' + str(decay) + ' ' + str(activation) + ' ' + str(optimizer)

    with open(args.out_file, 'a') as output:
        output.write('nlayers=' + str(nlayers) + ',')
        output.write('nunits=' + str(nunits) + ',')
        output.write('learnrate=' + str(learnrate) + ',')
        output.write('mb=' + str(mb) + ',')
        output.write('keep_prob=' + str(keep_prob) + ',')
        output.write('decay_step=' + str(decay_step) + ',')
        output.write('decay=' + str(decay) + ',')
        output.write('activation=' + str(activation) + ',')
        output.write('optimizer=' + str(optimizer))

    is_failure = os.system(line + ' >> ' + args.out_file)

    if(is_failure):
        with open(args.out_file, 'a') as output:
            output.write('\n')

    num_failures = update_failure_counter(is_failure, num_failures)
