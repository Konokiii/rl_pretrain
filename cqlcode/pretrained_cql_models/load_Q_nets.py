import os
import torch
import sys

sys.path.append('../')

from model_alias import *
import SimpleSAC
from ml_collections import ConfigDict


def load_model(pretrain_model_name):
    pretrain_full_path = os.path.join(PRETRAINED_MODEL_FOLDER, pretrain_model_name)
    if os.path.exists(pretrain_full_path):
        pretrain_dict = torch.load(pretrain_full_path)
        pretrained_agent = pretrain_dict['agent']
        pre_qf1 = pretrained_agent.qf1
        pre_qf2 = pretrained_agent.qf2
        sendback_path = os.path.join(MODEL_SAVE_FOLDER, pretrain_model_name)
        print(sendback_path)
        if not os.path.exists(sendback_path):
            torch.save({'qf1': pre_qf1, 'qf2': pre_qf2}, sendback_path)
            print('Saved state_dict of Q functions to send back.')
        else:
            print('Pretrained Q functions are already saved.')


PRETRAINED_MODEL_FOLDER = './'
MODEL_SAVE_FOLDER = './pretrainedQNets/'

pretrained_models_names = [
    ant_iid_l2,
    hopper_iid_l2,
    halfcheetah_iid_l2,
    walker_iid_l2,

    ant_random_l2,
    hopper_random_l2,
    halfcheetah_random_l2,
    walker_random_l2
]

pretrained_models_names.extend(list(abl_dimension.values()))

for name in pretrained_models_names:
    load_model(name)
