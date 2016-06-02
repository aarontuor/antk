import argparse
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import random

#TODO : fit line to points on graph

parser = argparse.ArgumentParser()

parser.add_argument('-in_file', dest='in_file', type=str, required=True)
parser.add_argument('-out_dir', dest='out_dir', type=str, required=True)

args = parser.parse_args()

colors = [name for name in matplotlib.colors.cnames.keys()]

hyper_params = ['nlayers', 'nunits', 'learnrate', 'mb', 'keep_prob', 'decay_step', 'decay', 'optimizer', 'activation']

NUM_BARS_PER_CLUSTER = len(hyper_params)

BAR_WIDTH = 0.1
X_AXIS_LIMIT = 10

def is_successful_experiment(experiment_line):
    return (hyper_param in experiment_line and 'Accuracy' in experiment_line)

def clean_line(experiment_line):
    experiment_line = experiment_line[:-1] # remove new line character
    experiment_line = experiment_line.split(',') #make into list
    return experiment_line

def get_pair(experiment_line):

    for pair in experiment_line:
        if hyper_param in pair:
            hyp_value = pair.split('=')[1]
        if 'Accuracy' in pair:
            accuracies_string = pair.split('=')[1]

    accuracies_list = accuracies_string.replace('[', '').replace(']', '').split(' ') #convert string to list
    accuracies_list = [float(i) for i in accuracies_list]
    return hyp_value, accuracies_list

def add_new_accuracies_to_dict(accuracies_dict, accuracies_list):

    if hyp_value in accuracies_dict.keys():
        #accuracies_dict[hyp_value].append(accuracies_list)
        accuracies_dict[hyp_value] = np.vstack([accuracies_dict[hyp_value], accuracies_list]) # add additional row to matrix
    else:
        #accuracies_dict[hyp_value] = [accuracies_list]
        accuracies_dict[hyp_value] = np.zeros((0,NUM_OUTPUTS), dtype=float) # create empty matrix for the new hyp_value
        accuracies_dict[hyp_value] = np.vstack([accuracies_dict[hyp_value], accuracies_list]) # add first row to matrix

    return accuracies_dict

def make_bar_plot(accuracies_dict):

    figure, subplot_axes = plt.subplots()

    cluster_counter = 0

    hyp_values = []

    num_clusters = len(accuracies_dict.keys())

    for hyp_value, accuracies_matrix in accuracies_dict.iteritems():
        bar_placements = [cluster_counter+BAR_WIDTH*bar_num for bar_num in range(0,NUM_OUTPUTS)]
        average_accuracies = accuracies_matrix.mean(axis=0)
        subplot_axes.bar(bar_placements, average_accuracies, BAR_WIDTH, color=[color for color in colors[0:NUM_OUTPUTS]])

        cluster_counter += X_AXIS_LIMIT/num_clusters

        hyp_values.append(hyp_value)

    subplot_axes.set_xlim((0, X_AXIS_LIMIT))
    subplot_axes.set_ylim(0, 1.0)

    subplot_axes.set_xticks([i*2+0.4/2 for i in range(0,len(hyp_values))])
    subplot_axes.set_xticklabels(hyp_values)

    plt.xlabel(hyper_param)
    plt.ylabel('Accuracy')
    plt.title('Tuning')
    plt.grid(True)

    return plt


def make_scatterplot(accuracies_dict):

    for hyp_value, accuracies_matrix in accuracies_dict.iteritems():
        average_accuracies = accuracies_matrix.mean(axis=0) # Find average accuracy for each output
        for index, accuracy in enumerate(average_accuracies):

            plt.plot([hyp_value], [accuracy], color=colors[index], marker='.')

    plt.xlabel(hyper_param)
    plt.ylabel('Accuracy')
    plt.title('Tuning')
    plt.grid(True)

    return plt

def addLegend(plt):

    label_names = get_label_names()

    handles = []

    for index, label_name in enumerate(label_names):

        handles.append(mpatches.Patch(color=colors[index], label=label_name))
    plt.legend(handles=handles, loc='upper right', prop={'size':6})
    return plt


def get_label_names():
    with open(args.in_file, 'r') as in_file:
        for experiment_line in in_file:
            experiment_line = clean_line(experiment_line)
            for pair in experiment_line:
                if 'Label_Names' in pair:
                    label_names = pair.split('=')[1]
                    label_names = label_names.replace('[', '').replace(']', '').split(' ') #convert string to list
                    print label_names
                    return label_names
    return []



for hyper_param in hyper_params:
    with open(args.in_file, 'r') as in_file:
        accuracies_dict = {}
        for experiment_line in in_file:

            if not is_successful_experiment(experiment_line):
                continue

            experiment_line = clean_line(experiment_line)

            hyp_value, accuracies_list = get_pair(experiment_line)

            NUM_OUTPUTS = len(accuracies_list)

            accuracies_dict = add_new_accuracies_to_dict(accuracies_dict, accuracies_list)

        if hyper_param == 'activation' or hyper_param == 'optimizer': #BAR PLOT

            plt = make_bar_plot(accuracies_dict)

        else: #SCATTERPLOT

            plt = make_scatterplot(accuracies_dict)

        plt = addLegend(plt)

        plt.savefig(args.out_dir+'/'+hyper_param+'_tuning.png')

        plt.clf() # clear current figure
