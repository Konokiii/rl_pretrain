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
from SimpleSAC.run_cql_dzx import run_single_exp, get_default_variant_dict
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

    variant = get_default_variant_dict() # this is a dictionary
    ###########################################################
    exp_prefix = 'cqlr3n_reg'
    settings = [
        'env', '', MUJOCO_4_ENVS,
        'dataset', '', MUJOCO_3_DATASETS,
        'pretrain_mode', 'pre', ['syn_mdp'],
        'n_epochs', 'ep', [200],
        'weight_reg_epochs', 'regEp', [5, 10, 20],
        'mdppre_n_state', 'ns', [100],
        'mdppre_policy_temperature', 'pt', [1],
        'mdppre_same_as_s_and_policy', 'same', [True],
        'seed', '', [42, 666, 1024, 2048, 4069],
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)
    exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix)

    """replace values"""
    cql_related_settings = ['policy_lr', 'cql_lagrange']
    for key, value in actual_setting.items():
        if key in cql_related_settings:
            setattr(variant['cql'], key, value)
        else:
            variant[key] = value

    data_dir = '/checkpoints'
    logger_kwargs = setup_logger_kwargs_dt(exp_name_full, variant['seed'], data_dir)
    variant["outdir"] = logger_kwargs["output_dir"]
    variant["exp_name"] = logger_kwargs["exp_name"]
    variant['policy_hidden_layer'] = variant['qf_hidden_layer']
    variant['cql'].cql_n_actions = 3
    run_single_exp(variant)


if __name__ == '__main__':
    main()
