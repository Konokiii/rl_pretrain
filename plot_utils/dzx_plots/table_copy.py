import os.path
import numpy as np
import pandas as pd
import json
# use this to generate the main table
from plot_utils.log_alias import *

base_measures = ['best_return_normalized', 'best_return',
                 'final_test_returns', 'final_test_normalized_returns',
                 'best_weight_diff',
                 'best_weight_sim',

                 'best_feature_diff',
                 'best_feature_sim',

                 "best_0_weight_diff",
                 "best_1_weight_diff",
                 "best_0_weight_sim",
                 "best_1_weight_sim",

                 'convergence_iter',
                 'convergence_update',

                 'final_feature_diff',
                 'final_weight_diff',
                 'final_feature_sim',
                 'final_weight_sim',

                 'best_5percent_normalized',
                 'best_10percent_normalized',
                 'best_25percent_normalized',
                 'best_50percent_normalized',
                 'best_100percent_normalized',
                 'best_later_half_normalized',
                 'last_four_normalized'
                 ]


def get_extra_dict_multiple_seeds(datafolder_path):
    # for a alg-dataset variant, obtain a dictionary with key-value pairs as measure:[avg across seeds, std across seeds]
    if not os.path.exists(datafolder_path):
        raise FileNotFoundError("Path does not exist: %s" % datafolder_path)
    # return a list, each entry is the final performance of a seed
    aggregate_dict = {}
    measures = base_measures
    for measure in measures:
        aggregate_dict[measure] = []
    aggregate_dict['weight_diff_100k'] = []  # TODO might want to extend later...
    aggregate_dict['feature_diff_100k'] = []

    for subdir, dirs, files in os.walk(datafolder_path):
        if 'extra_new.json' in files or 'extra.json' in files:
            if 'extra_new.json' in files:
                extra_dict_file_path = os.path.join(subdir, 'extra_new.json')
            else:
                extra_dict_file_path = os.path.join(subdir, 'extra.json')

            with open(extra_dict_file_path, 'r') as file:
                extra_dict = json.load(file)
                for measure in measures:
                    aggregate_dict[measure].append(float(extra_dict[measure]))
                # if 'weight_diff_100k' not in extra_dict:
                #     aggregate_dict['weight_diff_100k'].append(float(extra_dict['final_weight_diff']))
                #     aggregate_dict['feature_diff_100k'].append(float(extra_dict['final_feature_diff']))
                # else:
                #     print(extra_dict['feature_diff_100k'])
                #     aggregate_dict['weight_diff_100k'].append(float(extra_dict['weight_diff_100k']))
                #     aggregate_dict['feature_diff_100k'].append(float(extra_dict['feature_diff_100k']))

    for measure in measures:
        if len(aggregate_dict[measure]) == 0:
            print(datafolder_path, 'has nothing for measure:', measure)
        aggregate_dict[measure] = [np.mean(aggregate_dict[measure]), np.std(aggregate_dict[measure])]
    for measure in ['final_test_returns', 'final_test_normalized_returns', 'best_return', 'best_return_normalized',
                    'best_5percent_normalized', 'best_10percent_normalized', 'best_25percent_normalized',
                    'best_50percent_normalized', 'best_100percent_normalized', 'best_later_half_normalized',
                    'last_four_normalized', 'convergence_update'
                    ]:
        aggregate_dict[measure + '_std'] = [aggregate_dict[measure][1], ]
    return aggregate_dict


MUJOCO_4_ENVS = [
    'hopper',
    'halfcheetah',
    # 'walker2d',
    # 'ant'
]
MUJOCO_3_DATASETS = ['medium-expert']
all_envs = []
for e in MUJOCO_4_ENVS:
    for dataset in MUJOCO_3_DATASETS:
        all_envs.append('%s_%s' % (e, dataset))


def get_alg_dataset_dict(algs, envs):
    # load extra dict for all alg, all envs, all seeds
    alg_dataset_dict = {}
    for alg in algs:
        alg_dataset_dict[alg] = {}
        for env in envs:
            folderpath = os.path.join(data_path, '%s_%s' % (alg, env))
            alg_dataset_dict[alg][env] = get_extra_dict_multiple_seeds(folderpath)
    return alg_dataset_dict


def get_aggregated_value(alg_dataset_dict, alg, measure):
    # for an alg-measure pair, aggregate over all datasets
    value_list = []
    for dataset, extra_dict in alg_dataset_dict[alg].items():
        value_list.append(extra_dict[measure][0])  # each entry is the value from a dataset
    return np.mean(value_list), np.std(value_list)


OLD_ROWS = [
    'best_return_normalized',
    'best_return_normalized_std',

    'best_feature_diff',
    'best_weight_diff',
    "best_0_weight_diff",
    "best_1_weight_diff",

    'best_feature_sim',
    'best_weight_sim',
    "best_0_weight_sim",
    "best_1_weight_sim",

    'convergence_iter',

    'final_test_normalized_returns',
    'final_feature_diff',
    'final_weight_diff',
    'final_feature_sim',
    'final_weight_sim',
]
OLD_ROW_NAMES = ['Best Score',
                 'Best Std over Seeds',

                 'Best Feature Diff',
                 'Best Weight Diff',
                 'Best Weight Diff L0',
                 'Best Weight Diff L1',

                 'Best Feature Sim',
                 'Best Weight Sim',
                 'Best Weight Sim L0',
                 'Best Weight Sim L1',

                 'Convergence Iter',

                 'Final Score',
                 'Final Feature Diff',
                 'Final Weight Diff',
                 'Final Feature Sim',
                 'Final Weight Sim',
                 ]

NEW_PERFORMANCE_ROWS = [
    'best_return_normalized',
    'best_5percent_normalized',
    'best_10percent_normalized',
    'best_25percent_normalized',
    'best_50percent_normalized',
    'best_100percent_normalized',
    'last_four_normalized'
]
NEW_PERFORMANCE_ROW_NAMES = [
    'Best Score',
    'Best Score 5\\%',
    'Best Score 10\\%',
    'Best Score 25\\%',
    'Best Score 50\\%',
    'Best Score 100\\%',
    'Last Four Avg'
]

change_std_rows = [
    'best_return_normalized',
    'best_5percent_normalized',
    'best_10percent_normalized',
    'best_25percent_normalized',
    'best_50percent_normalized',
    'best_100percent_normalized',
    'best_later_half_normalized',
    'last_four_normalized',
    'convergence_update'
]

row_names_higher_is_better = [
    'Best Score',
    'Final Score',
    'Best Weight Sim',
    'Best Feature Sim',
    'Best Weight Sim L0',
    'Best Weight Sim L1',
    'Best Weight Sim FC',
    'Final Feature Sim',
    'Final Weight Sim',
    'Prev Feature Sim',
    'Prev Weight Sim',
    'Best Score 5\\%',
    'Best Score 10\\%',
    'Best Score 25\\%',
    'Best Score 50\\%',
    'Best Score 100\\%',
    'Last Four Avg',
]

MEASURE_HIGHER_IS_BETTER = [
    'final_test_returns',
    'final_test_normalized_returns',
    'best_return',
    'best_return_normalized',
    'best_5percent_normalized',
    'best_10percent_normalized',
    'best_25percent_normalized',
    'best_50percent_normalized',
    'best_100percent_normalized',
    'best_later_half_normalized',
    'last_four_normalized',
]

row_names_use_1_precision = [
    'Best Score', 'Best Std over Seeds', 'Convergence Iter',
    'Final Score',
    'Best Score 5\\%',
    'Best Score 10\\%',
    'Best Score 25\\%',
    'Best Score 50\\%',
    'Best Score 100\\%',
    'Last Four Avg'
]

"""table generation"""


def generate_aggregate_table(algs, alg_dataset_dict, column_names, best_value_bold=True, bold_threshold=0.05):
    print("\nNow generate latex table:\n")
    # each row is a measure, each column is an algorithm variant
    rows = NEW_PERFORMANCE_ROWS
    row_names = NEW_PERFORMANCE_ROW_NAMES

    table = np.zeros((2, len(rows), len(algs)))
    # each iter we generate a row
    for i, row in enumerate(rows):
        for j, alg in enumerate(algs):
            table[0, i, j], table[1, i, j] = get_aggregated_value(alg_dataset_dict, alg, row)
            if row in change_std_rows:  # TODO
                std_mean, std_std = get_aggregated_value(alg_dataset_dict, alg, row + '_std')
                table[1, i, j] = std_mean
            if row == 'final_test_normalized_returns':
                std_mean, std_std = get_aggregated_value(alg_dataset_dict, alg, 'final_test_normalized_returns_std')
                table[1, i, j] = std_mean

    max_values = np.max(table[0], axis=1)
    min_values = np.min(table[0], axis=1)

    col_name_line = ''
    for col in column_names:
        col_name_line += str(col) + ' & '
    col_name_line = col_name_line[:-2] + '\\\\'
    print(col_name_line)
    print("		\\hline ")
    for i, row_name in enumerate(row_names):
        row_string = row_name
        for j in range(len(algs)):
            mean, std = table[0, i, j], table[1, i, j]
            bold = False
            if best_value_bold:
                if row_name not in row_names_higher_is_better:
                    if mean <= (1 + bold_threshold) * min_values[i]:
                        bold = True
                else:
                    if mean >= (1 - bold_threshold) * max_values[i]:
                        bold = True
                if bold:
                    if 'Prev' in row_name:
                        row_string += (' & \\textbf{%.6f}' % (mean,))
                    elif row_name in row_names_use_1_precision:
                        row_string += (' & \\textbf{%.1f} $\pm$ %.1f' % (mean, std))
                    else:
                        row_string += (' & \\textbf{%.3f} $\pm$ %.1f' % (mean, std))
                else:
                    if 'Prev' in row_name:
                        row_string += (' & %.6f' % (mean,))
                    elif row_name in row_names_use_1_precision:
                        row_string += (' & %.1f $\pm$ %.1f' % (mean, std))
                    else:
                        row_string += (' & %.3f $\pm$ %.1f' % (mean, std))
        row_string += '\\\\'
        print(row_string)


def generate_aggregate_performance(algs, alg_dataset_dict, column_names, best_value_bold=True, bold_threshold=0.02,
                                   measure='best_return_normalized', row_name='Average over datasets'):
    # each row is a measure, each column is an algorithm variant
    rows = [measure]
    row_names = [row_name]

    table = np.zeros((2, len(rows), len(algs)))
    # each iter we generate a row
    for i, row in enumerate(rows):
        for j, alg in enumerate(algs):
            table[0, i, j], table[1, i, j] = get_aggregated_value(alg_dataset_dict, alg, row)
            if row in change_std_rows:  # TODO
                std_mean, std_std = get_aggregated_value(alg_dataset_dict, alg, row + '_std')
                table[1, i, j] = std_mean
            if row == 'final_test_normalized_returns':
                std_mean, std_std = get_aggregated_value(alg_dataset_dict, alg, 'final_test_normalized_returns_std')
                table[1, i, j] = std_mean

    max_values = np.max(table[0], axis=1)
    min_values = np.min(table[0], axis=1)

    print("		\\hline ")
    for i, row_name in enumerate(row_names):
        row_string = row_name
        for j in range(len(algs)):
            mean, std = table[0, i, j], table[1, i, j]
            bold = False
            if best_value_bold:
                if measure not in MEASURE_HIGHER_IS_BETTER:
                    if mean <= (1 + bold_threshold) * min_values[i]:
                        bold = True
                else:
                    if mean >= (1 - bold_threshold) * max_values[i]:
                        bold = True
                if bold:
                    row_string += (' & \\textbf{%.1f} $\pm$ %.1f' % (mean, std))
                else:
                    row_string += (' & %.1f $\pm$ %.1f' % (mean, std))
        row_string += '\\\\'
        print(row_string)


# TODO add an aggregate score at the end
def generate_per_env_score_table_new(algs, alg_dataset_dict, column_names, best_value_bold=True, bold_threshold=0.02,
                                     measure='best_return_normalized'):
    print("\nNow generate latex table:\n")
    # measure = 'best_100percent_normalized'
    # each row is a env-dataset pair, each column is an algorithm variant
    rows = []
    row_names = []
    for dataset in ['medium-expert']:
        for e in ['halfcheetah', 'hopper']:
            rows.append('%s_%s' % (e, dataset))
            row_names.append('%s-%s' % (e, dataset))

    table = np.zeros((2, len(rows), len(algs)))
    # each iter we generate a row
    for i, row in enumerate(rows):
        for j, alg in enumerate(algs):
            try:
                table[0, i, j], table[1, i, j] = alg_dataset_dict[alg][row][measure]
            except:
                print(alg)
                print(row)
                print(alg_dataset_dict[alg][row].keys())
                quit()

    max_values = np.max(table[0], axis=1)
    min_values = np.min(table[0], axis=1)

    col_name_line = ''
    for col in column_names:
        col_name_line += str(col) + ' & '
    col_name_line = col_name_line[:-2] + '\\\\'
    print(col_name_line)
    print("		\\hline ")
    for i, row_name in enumerate(row_names):
        row_string = row_name
        for j in range(len(algs)):
            mean, std = table[0, i, j], table[1, i, j]
            bold = False
            if best_value_bold:
                if measure not in MEASURE_HIGHER_IS_BETTER:
                    if mean <= (1 + bold_threshold) * min_values[i]:
                        bold = True
                else:
                    if mean >= (1 - bold_threshold) * max_values[i]:
                        bold = True
                if bold:
                    row_string += (' & \\textbf{%.1f} $\pm$ %.1f' % (mean, std))
                else:
                    row_string += (' & %.1f $\pm$ %.1f' % (mean, std))
        row_string += '\\\\'
        print(row_string)

    # if add_aggregate_result_in_the_end:
    #     print("		\\hline ")
    #     agg_mean, agg_std = np.mean(table[0], axis=0), np.mean(table[1], axis=0)
    #     print(agg_mean)
    #     print(agg_std)



def dzx_tuned_cql(bold_thres):
    algs = [
        iclr_cql,
        tuned_cql,
        iclr_cql_mdp_t1,
        tuned_mdp
    ]

    col_names = [
        'Best Score',
        'CQL',
        'Tuned CQL',
        'MDP',
        'Tuned MDP'
    ]

    envs = all_envs
    alg_dataset_dict = get_alg_dataset_dict(algs, envs)
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, bold_threshold=bold_thres)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, bold_threshold=bold_thres)

    col_names[0] = 'Average Last Four'
    generate_per_env_score_table_new(algs, alg_dataset_dict, col_names, measure='last_four_normalized', bold_threshold=bold_thres)
    generate_aggregate_performance(algs, alg_dataset_dict, col_names, measure='last_four_normalized', bold_threshold=bold_thres)



data_path = '../../code/checkpoints/final'
bold_thres = 0.01
dzx_tuned_cql(bold_thres)