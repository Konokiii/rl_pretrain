import os

import torch
import torch.nn as nn
import sys

sys.path.append('../')
import SimpleSAC
from ml_collections import ConfigDict


pretrained_model_folder = './'
pretrained_models_names = [
    'cql_ant_1000_100_100_1_1_111_8_mdp_same_noproj_2_256_200.pth',
    'cql_halfcheetah_1000_100_100_1_1_17_6_mdp_same_noproj_2_256_200.pth',
    'cql_hopper_1000_100_100_1_1_11_3_mdp_same_noproj_2_256_200.pth',
    'cql_walker2d_1000_100_100_1_1_17_6_mdp_same_noproj_2_256_200.pth'
]
model_save_folder = './pretrainedQNets/'
def load_model(pretrain_model_name):
    pretrain_full_path = os.path.join(pretrained_model_folder, pretrain_model_name)
    if os.path.exists(pretrain_full_path):
        pretrain_dict = torch.load(pretrain_full_path)
        pretrained_agent = pretrain_dict['agent']
        pre_qf1 = pretrained_agent.qf1
        pre_qf2 = pretrained_agent.qf2
        sendback_path = os.path.join(model_save_folder, pretrain_model_name)
        if not os.path.exists(sendback_path):
            torch.save({'qf1': pre_qf1, 'qf2': pre_qf2}, sendback_path)
            print('Saved state_dict of Q functions to send back.')
        else:
            print('Pretrained Q functions are already saved.')