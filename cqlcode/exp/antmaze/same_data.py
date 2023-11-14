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

    variant = get_default_variant_dict() # this is a dictionary
    ###########################################################
    exp_prefix = 'cql_tuned'
    settings = [
        'env', '', ['antmaze-large'],#ANTMAZE_3_ENVS,
        'dataset', '', ['diverse'],#ANTMAZE_2_DATASETS,
        # 'do_pretrain_only', 'dpo', [True],
        'pretrain_mode', 'pre', ['q_sprime'],  # 'none', 'q_sprime', 'mdp_q_sprime'
        'qf_hidden_layer', 'l', [3],
        'use_safe_q', 'safeQ', [False],
        'cql_lagrange', 'lag', [True],
        'seed', '', [0]#list(range(5))
    ]

    indexes, actual_setting, total, hyper2logname = get_setting_dt(settings, setting)
    exp_name_full = get_auto_exp_name(actual_setting, hyper2logname, exp_prefix)
    # replace default values with grid values

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
    # TODO for now we set this to 3 for faster experiments
    variant['policy_hidden_layer'] = variant['qf_hidden_layer']
    variant['cql'].cql_n_actions = 3
    variant['cql'].policy_lr = 1e-4
    variant['cql'].cql_min_q_weight = 5.0
    variant['cql'].cql_target_action_gap = 5.0
    run_single_exp(variant)


if __name__ == '__main__':
    main()
