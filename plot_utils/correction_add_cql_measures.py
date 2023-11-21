import os
import pandas as pd
import json
import numpy as np
import shutil


def get_other_score_measures(path):
    # return a dictionary
    with open(path, "r") as f:
        df = pd.read_csv(f, delimiter="\t", header=0)
        test_returns_norm = df["TestEpNormRet"].to_numpy()
        n = test_returns_norm.shape[0]
        test_returns_norm_sorted = np.sort(test_returns_norm)
        d = {
            'best_5percent_normalized': test_returns_norm_sorted[-int(n * 0.05):].mean(),
            'best_10percent_normalized': test_returns_norm_sorted[-int(n * 0.1):].mean(),
            'best_25percent_normalized': test_returns_norm_sorted[-int(n * 0.25):].mean(),
            'best_50percent_normalized': test_returns_norm_sorted[-int(n * 0.5):].mean(),
            'best_100percent_normalized': test_returns_norm_sorted.mean(),
            'best_later_half_normalized': test_returns_norm[int(n * 0.5):].mean(),
            'last_four_normalized': test_returns_norm[-1:-5:-1].mean(),
        }
    return d


hyperparameters = {'env':'env', 'dataset':'dataset', 'offline_data_ratio':'offRatio', 'mdppre_n_state':'ns',
                   'mdppre_policy_temperature':'pt', 'n_pretrain_epochs':'preEp', 'seed':'s'}
settings = []
setting_names = []
base_path = '../code/checkpoints/rebuttal/offRatio'
for root, dirs, files in os.walk(base_path):
    if '/cql' in root:
        for dir in dirs:
            # Go through every subfolder in this folder
            subfolder = os.path.join(root, dir)
            for file in os.listdir(subfolder):
                if file == 'progress.csv':
                    try:
                        extra_measures_dict = get_other_score_measures(os.path.join(subfolder, file))
                        ex = os.path.join(subfolder, 'extra.json')
                        ex_to_use = ex
                        if os.path.exists(ex):
                            # load extra.json
                            # print(ex_to_use)
                            with open(ex_to_use, 'r') as ex_file:
                                extra_dict = json.load(ex_file)

                            extra_dict.update(extra_measures_dict)
                            with open(ex_to_use, 'w') as ex_file:
                                json.dump(extra_dict, ex_file)
                    except Exception as e:
                        # shutil.rmtree(subfolder)
                        print(dir)
                        print(e)


#                         single_setting = {'qf_hidden_layer':2,
#                                           'pretrain_mode': 'mdp_same_noproj',
#                                           'mdppre_same_as_s_and_policy': True
#                                           }
#                         setting_with_prefix = str(dir).split('_')
#                         setting_names.append('_'.join(setting_with_prefix[:-1]))
#                         for key in hyperparameters.keys():
#                             for alias in setting_with_prefix:
#                                 if key == 'env' and alias in ['ant', 'hopper', 'walker2d', 'halfcheetah']:
#                                     single_setting['env'] = alias
#                                     break
#                                 if key == 'dataset' and alias in ['medium', 'medium-expert', 'medium-replay']:
#                                     single_setting['dataset'] = alias
#                                     break
#                                 if key == 'seed' and alias.startswith('same'):
#                                     continue
#                                 if alias.startswith(hyperparameters[key]):
#                                     prefix_len = len(hyperparameters[key])
#                                     value = alias[prefix_len:]
#                                     if key in ['mdppre_n_state', 'n_pretrain_epochs', 'seed']:
#                                         value = int(value)
#                                     elif key in ['offline_data_ratio', 'mdppre_policy_temperature']:
#                                         if value == 'inf3':
#                                             pass
#                                         else:
#                                             value = float(value)
#                                             if value == 1:
#                                                 value = int(value)
#
#                                     single_setting[key] = value
#                                     break
#                         settings.append(single_setting)
# print(len(settings), len(setting_names))
# with open('patch_settings.json', 'w') as settings_file:
#     json.dump(settings, settings_file)
# with open('patch_names.json', 'w') as names_file:
#     json.dump(setting_names, names_file)
