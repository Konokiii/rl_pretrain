import os

import torch
import torch.nn as nn
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import model_alias


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

env_list = [
    'halfcheetah',
    'hopper',
    'walker2d',
    'ant'
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
    'L1',
    'L2',
    'L1',
    'L2',
    'L3'
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


def load_layers(env):
    # TODO Check isfile or not:
    two_models = [name for name in os.listdir(MODEL_SAVE_FOLDER) if env in name]
    model_l2_name = [name for name in two_models if 'l2' in name][0]
    model_l3_name = [name for name in two_models if 'l3' in name][0]
    model_l2 = torch.load(os.path.join(MODEL_SAVE_FOLDER, model_l2_name))
    model_l3 = torch.load(os.path.join(MODEL_SAVE_FOLDER, model_l3_name))

    # Load qf1 only for now:
    model_l2_qf1 = model_l2['qf1']
    model_l3_qf1 = model_l3['qf1']

    layers = list(model_l2_qf1.hidden_layers) + list(model_l3_qf1.hidden_layers)
    W_list = [l.weight for l in layers]
    bias_list = [l.bias for l in layers]

    return W_list, bias_list


# Load statistics to a 2D numpy array to plot.
def prepare_values(env, measures):
    W_list, bias_list = load_layers(env)
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
def generate_table(stats_names, stats, column_names, env_list):
    table_Ws, table_bs = [], []
    for env in env_list:
        table1, table2 = prepare_values(env, stats)
        table_Ws.append(table1)
        table_bs.append(table2)

    print("\nNow generate latex table:\n")
    print('\hline')
    column_names = [''] + column_names
    for i, measure in enumerate(stats):
        column_names[0] = '\\textbf{%s}' % stats_names[i]
        col_name_line = ''
        for col in column_names:
            col_name_line += str(col) + ' & '
        col_name_line = col_name_line[:-2] + '\\\\'
        print(col_name_line)
        print("		\\hline ")
        for e, env in enumerate(env_list):
            row_string = env + '\\emph{(Weight)}' if e == 0 else env
            for j in range(len(column_names) - 1):
                mean = table_Ws[e][i, j]
                row_string += (' & %.3f ' % mean)
            row_string += '\\\\'
            print(row_string)

        print('\\hdashline')

        if measure in STATS_AVAILABLE_FOR_BIAS:
            for e, env in enumerate(env_list):
                row_string = env + '\\emph{(Bias)}' if e == 0 else env
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


def plot_hist(environments, layer_names):
    all_W = []
    all_bias = []
    for env in environments:
        W_list, b_list = load_layers(env)
        all_W.append(W_list)
        all_bias.append(b_list)

    for j in range(len(layer_names)):
        W_to_plot = [all_W[i][j].flatten().detach().numpy() for i in range(len(environments))]
        bias_to_plot = [all_bias[i][j].flatten().detach().numpy() for i in range(len(environments))]

        # Add default init weight and bias:
        # TODO: Currently only do default init for one input dimension(env); extend this later on.
        fan_in = len(W_to_plot[0]) / 256
        default_init_layer = nn.Linear(int(fan_in), 256)
        W_to_plot.append(default_init_layer.weight.flatten().detach().numpy())
        bias_to_plot.append(default_init_layer.bias.flatten().detach().numpy())

        n_bins = 15
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        colors = ['red', 'blue', 'darkorange', 'magenta', 'lime']
        colors = colors[:len(W_to_plot)]
        legend = environments + ['default']
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
        plt.suptitle(column_names[j])
        plt.savefig(FIGURE_SAVE_FOLDER + layer_names[j])


column_names = [
    'L2_1',
    'L2_2',
    'L3_1',
    'L3_2',
    'L3_3'
]
plot_hist(env_list, column_names)
