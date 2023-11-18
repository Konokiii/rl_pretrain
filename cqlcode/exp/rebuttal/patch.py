import os
import sys

# with new logger
ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
ld_library_path += ':/workspace/.mujoco/mujoco210/bin:/usr/local/nvidia/lib:/usr/lib/nvidia'
os.environ['LD_LIBRARY_PATH'] = ld_library_path
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/workspace/.mujoco/mujoco210/'
import time
from copy import deepcopy
import uuid
import numpy as np
import pprint
import gym
import torch
import d4rl
from SimpleSAC.utils import get_user_flags, define_flags_with_default
from exp_scripts.grid_utils import *
from SimpleSAC.run_cql_watcher import run_single_exp, get_default_variant_dict
from SimpleSAC.conservative_sac import ConservativeSAC
from SimpleSAC.utils import WandBLogger
import argparse

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=int, default=0)
    args = parser.parse_args()
    setting = args.setting

    variant = get_default_variant_dict()  # this is a dictionary
    ###########################################################
    exp_prefix = 'cqlr3n'
    settings = [
        {'env': 'hopper',
         'dataset': 'medium-expert',
         'pretrain_mode': 'mdp_same_noproj',
         'qf_hidden_layer': 2,
         'mdppre_n_state': 3500,
         'mdppre_policy_temperature': 'case_mapping',
         'n_pretrain_epochs': 20,
         'mdppre_same_as_s_and_policy': True,
         'seed': 3
         }
    ]
    setting_names = [
        'cqlr3n_premdp_same_noproj_l2_ns3500_ptcase_mapping_preEp20_sameTrue_hopper_medium-expert'
    ]

    actual_setting = settings[setting]
    exp_name_full = setting_names[setting]
    """replace values"""
    for key, value in actual_setting.items():
        variant[key] = value

    data_dir = '/checkpoints'
    logger_kwargs = setup_logger_kwargs_dt(exp_name_full, variant['seed'], data_dir)
    variant["outdir"] = logger_kwargs["output_dir"]
    variant["exp_name"] = logger_kwargs["exp_name"]
    # TODO for now we set this to 3 for faster experiments
    variant['cql'].cql_n_actions = 3
    run_single_exp(variant)


if __name__ == '__main__':
    main()
