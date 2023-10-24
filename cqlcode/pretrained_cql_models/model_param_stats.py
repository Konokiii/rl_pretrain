import os

import torch
import torch.nn as nn
import numpy as np
import math
import sys
sys.path.append('../')
import SimpleSAC
import matplotlib.pyplot as plt
from model_alias import *


PRETRAINED_MODEL_FOLDER = './'
MODEL_SAVE_FOLDER = './pretrainedQNets/'
FIGURE_SAVE_FOLDER = './figures/'

STATS_AVAILABLE = [
    'mean',
    'std',
    'max',
    'min',
    'norm',
    'avgNorm'
]

# This is a subset of STATS_AVAILABLE.
STATS_AVAILABLE_FOR_BIAS = [
    'mean',
    'std',
    'max',
    'min',
    'norm',
    'avgNorm'
]


# Output a dictionary containing statistics described in STATS_AVAILABLE.
def compute_statistic(value_mat, measures):
    stats_dict = {}

    for m in measures:
        if m == 'mean':
            res = value_mat.mean().item()
        elif m == 'std':
            res = value_mat.std().item()
        elif m == 'max':
            res = value_mat.max().item()
        elif m == 'min':
            res = value_mat.min().item()
        elif m == 'norm':
            res = value_mat.norm().item()
        elif m == 'avgNorm':
            res = value_mat.norm().item() / value_mat.flatten().shape[0]

        stats_dict[m] = res

    return stats_dict


def load_layers(model_name):
    model_path = os.path.join(MODEL_SAVE_FOLDER, model_name)
    if os.path.exists(model_path):
        model = torch.load(model_path)
    else:
        print('Path Not Found: ', model_path)
        return

    # Load qf1 only for now:
    qf1 = model['qf1']

    layers = list(qf1.hidden_layers)
    W_list = [l.weight for l in layers]
    bias_list = [l.bias for l in layers]

    return W_list, bias_list


# Load statistics to a 2D numpy array to plot.
def prepare_values(model_name, measures):
    W_list, bias_list = load_layers(model_name)
    column_len = len(W_list)
    row_len_W = len(measures)
    measures_b = [m for m in measures if m in STATS_AVAILABLE_FOR_BIAS]
    row_len_b = len(measures_b)

    table_W = np.zeros((row_len_W, column_len))
    table_b = np.zeros((row_len_b, column_len))

    for j in range(column_len):
        W, b = W_list[j], bias_list[j]

        stats_dict_W = compute_statistic(W, measures)
        for i in range(row_len_W):
            table_W[i][j] = stats_dict_W[measures[i]]

        stats_dict_b = compute_statistic(b, measures_b)
        for i in range(row_len_b):
            table_b[i][j] = stats_dict_b[measures_b[i]]

    return table_W, table_b


# Generate latex table.
def generate_table(stats_names, stats, column_names, model_list):
    table_Ws, table_bs = [], []
    for model in model_list:
        table1, table2 = prepare_values(model, stats)
        table_Ws.append(table1)
        table_bs.append(table2)

    print("\nNow generate latex table:\n")
    print('\hline')
    column_names = [''] + column_names
    for i, measure in enumerate(stats):
        column_names[0] = '\\textbf{%s}' % stats_names
        col_name_line = ''
        for col in column_names:
            col_name_line += str(col) + ' & '
        col_name_line = col_name_line[:-2] + '\\\\'
        print(col_name_line)
        print("		\\hline ")
        for e, model in enumerate(model_list):
            row_string = model + '\\emph{(Weight)}' if e == 0 else model
            for j in range(len(column_names) - 1):
                mean = table_Ws[e][i, j]
                row_string += (' & %.3f ' % mean)
            row_string += '\\\\'
            print(row_string)

        print('\\hdashline')

        if measure in STATS_AVAILABLE_FOR_BIAS:
            for e, model in enumerate(model_list):
                row_string = model + '\\emph{(Bias)}' if e == 0 else model
                for j in range(len(column_names) - 1):
                    mean = table_bs[e][i, j]
                    row_string += (' & %.3f ' % mean)
                row_string += '\\\\'
                print(row_string)
            print('\\hline')


# def generate_table(row_names, row_stats, column_names, env_list):
#     row_names_b = [row_names[i] for i in range(len(row_stats)) if row_stats[i] in STATS_AVAILABLE_FOR_BIAS]
#     column_names = [''] + column_names
#     print("\nNow generate latex table:\n")
#     print('\hline')
#     for env in env_list:
#         first_col_name = 'Measure' + f'({env})'
#         column_names[0] = first_col_name
#         table_W, table_b = load_values(env, row_stats)
#
#         col_name_line = ''
#         for col in column_names:
#             col_name_line += str(col) + ' & '
#         col_name_line = col_name_line[:-2] + '\\\\'
#         print(col_name_line)
#         print("		\\hline ")
#         for i, row_name in enumerate(row_names):
#             row_string = row_name + '\_W'
#             for j in range(len(column_names)-1):
#                 mean = table_W[i, j]
#                 row_string += (' & %.3f ' % mean)
#             row_string += '\\\\'
#             print(row_string)
#         print('\\hdashline')
#         for i, row_name in enumerate(row_names_b):
#             row_string = row_name + '\_b'
#             for j in range(len(column_names)-1):
#                 mean = table_b[i, j]
#                 row_string += (' & %.3f ' % mean)
#             row_string += '\\\\'
#             print(row_string)
#         print('\\hline')

# generate_table(stats_names, stats, column_names, env_list)


# ################################### Plot Figure #######################################
def round_up(num, n_decimal):
    return math.ceil(num*10**n_decimal) / 10**n_decimal


def round_down(num, n_decimal):
    return math.floor(num * 10 ** n_decimal) / 10 ** n_decimal


def plot_hist(model_names, legend_names, layer_names, fig_name_prefix, add_default=False):
    all_W = []
    all_bias = []
    for name in model_names:
        W_list, b_list = load_layers(name)
        all_W.append(W_list)
        all_bias.append(b_list)

    for j in range(len(layer_names)):
        W_to_plot = [all_W[i][j].flatten().detach().numpy() for i in range(len(model_names))]
        bias_to_plot = [all_bias[i][j].flatten().detach().numpy() for i in range(len(model_names))]

        # Add default init weight and bias:
        if add_default:
            # TODO: Currently only do default init for one input dimension(walker2d); extend this later on.
            fan_in = 23
            default_init_layer = nn.Linear(int(fan_in), 256)
            W_to_plot.append(default_init_layer.weight.flatten().detach().numpy())
            bias_to_plot.append(default_init_layer.bias.flatten().detach().numpy())

        n_bins = 15
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        colors = ['red', 'blue', 'darkorange', 'lime', 'magenta', 'pink']
        colors = colors[:len(W_to_plot)]
        legend = legend_names + ['default']
        ax0.hist(W_to_plot, bins=n_bins, density=True, histtype='bar', color=colors, label=legend)
        ax0.legend(prop={'size': 15})
        ax0.set_title('Weight Matrices')
        start, end = ax0.get_xlim()
        ax0.xaxis.set_ticks(np.arange(round_down(start, 1), round_up(end, 1), 0.1))

        ax1.hist(bias_to_plot, bins=n_bins, density=True, histtype='bar', color=colors)
        ax1.set_title('Bias Matrices')
        start, end = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(round_down(start, 1), round_up(end, 1), 0.05))

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        figure_name = f'{fig_name_prefix}_{layer_names[j]}'
        plt.suptitle(figure_name)
        plt.savefig(FIGURE_SAVE_FOLDER + figure_name)


model_list = [
    'halfcheetah',
    'hopper',
    'walker2d',
    'ant',
    'customized'
]

# Pick some from STATS_AVAILABLE
stats = [
    'mean',
    'std',
    'max',
    'min',
    'norm',
]

# Order of the stats_names should match stats
stats_names = [
    'Mean',
    'Std',
    'Max',
    'Min',
    'Norm',
]


column_names = [
    'L2_1',
    'L2_2',
]

# TODO: Why do we need to import SimpleSAC?
# generate_table(stats_names, stats, column_names, model_list)

# models = [
#     ant_random_l2,
#     hopper_random_l2,
#     halfcheetah_random_l2,
#     walker_random_l2
# ]
# legends = [
#     'ant',
#     'hopper',
#     'halfcheetah',
#     'walker2d'
# ]
dimS = [3, 15, 50, 100, 300]
dimA = [3, 15, 50, 100, 300]
legend_list = [[f's{s}a{a}' for a in dimA] for s in dimS]

for i, legends in enumerate(legend_list):
    models = [abl_dimension[l] for l in legends]
    plot_hist(models, legends, layer_names=['L1', 'L2'], fig_name_prefix=f'Dim_Ablation{i}', add_default=True)
