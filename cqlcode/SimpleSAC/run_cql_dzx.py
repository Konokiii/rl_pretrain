import os
import sys
import torch.nn.functional as F

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

import absl.app
import absl.flags

from SimpleSAC.conservative_sac import ConservativeSAC_REG
from SimpleSAC.replay_buffer import batch_to_torch, get_d4rl_dataset_with_ratio, subsample_batch, \
    index_batch, get_mdp_dataset_with_ratio, get_d4rl_dataset_from_multiple_envs
from SimpleSAC.model import TanhGaussianPolicy, SamplerPolicy, FullyConnectedQFunctionPretrain, \
    FullyConnectedQFunctionPretrain2
from SimpleSAC.sampler import StepSampler, TrajSampler
from SimpleSAC.utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, \
    prefix_metrics
from SimpleSAC.utils import WandBLogger
# from viskit.logging import logger_other, setup_logger
from exp_scripts.grid_utils import *
from redq.utils.logx import EpochLogger

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


def get_dictionary_from_kwargs(**kwargs):
    d = {}
    for key, val in kwargs.items():
        d[key] = val
    return d


def get_default_variant_dict():
    # returns a dictionary that contains all the default hyperparameters
    return get_dictionary_from_kwargs(
        env='halfcheetah',
        dataset='medium',
        max_traj_length=1000,
        seed=42,
        device=DEVICE,
        save_model=True,
        batch_size=256,

        reward_scale=1.0,
        reward_bias=0.0,
        clip_action=0.999,

        policy_hidden_layer=2,
        policy_hidden_unit=256,
        qf_hidden_layer=2,
        qf_hidden_unit=256,
        orthogonal_init=False,
        policy_log_std_multiplier=1.0,
        policy_log_std_offset=-1.0,

        n_epochs=200,
        bc_epochs=0,
        n_pretrain_epochs=20,
        # q_sprime, proj0_q_sprime, proj1_q_sprime, proj2_q_sprime
        # 'mdp_same_proj', 'mdp_same_noproj', q_sprime_3x, proj0_q_sprime_3x, proj1_q_sprime_3x, q_noact_sprime
        pretrain_mode='none',
        n_pretrain_step_per_epoch=5000,
        n_train_step_per_epoch=5000,
        eval_period=1,
        eval_n_trajs=10,
        exp_prefix='cqltest',
        cql=ConservativeSAC_REG.get_default_config(),
        logging=WandBLogger.get_default_config(),
        do_pretrain_only=False,
        pretrain_data_ratio=1,
        offline_data_ratio=1,
        q_distill_weight=0,
        q_distill_pretrain_steps=0,  # will not use the q distill weight defined here
        distill_only=False,
        q_network_feature_lr_scale=1,  # 0 means pretrained/random features are frozen
        init_scheme=None,
        use_safe_q=True,

        # mdp pretrain related
        mdppre_n_traj=1000,
        mdppre_n_state=1000,
        mdppre_n_action=1000,
        mdppre_policy_temperature=1,
        mdppre_transition_temperature=1,
        mdppre_same_as_s_and_policy=False,  # if True, then action hyper will be same as state, tt will be same as pt.
        mdppre_random_start=False,

        hard_update_target_after_pretrain=True,  # if True, hard update target networks after pretraining stage.

        weight_reg_epochs=20,
        extra_q_loss_discount=1,
        weight_decay=0.01
    )


def get_convergence_index(ret_list, threshold_gap=2):
    best_value = max(ret_list)
    convergence_threshold = best_value - threshold_gap
    k_conv = len(ret_list) - 1
    for k in range(len(ret_list) - 1, -1, -1):
        if ret_list[k] >= convergence_threshold:
            k_conv = k
    return k_conv


def concatenate_weights_of_model_list(model_list, weight_only=True):
    concatenated_weights = []
    for model in model_list:
        for name, param in model.named_parameters():
            if not weight_only or 'weight' in name:
                concatenated_weights.append(param.view(-1))
    return torch.cat(concatenated_weights)


# when compute weight diff, just provide list of important layers...from
def get_diff_sim_from_layers(layersA, layersB):
    weights_A = concatenate_weights_of_model_list(layersA)
    weights_B = concatenate_weights_of_model_list(layersB)
    weight_diff = torch.mean((weights_A - weights_B) ** 2).item()
    weight_sim = float(F.cosine_similarity(weights_A.reshape(1, -1), weights_B.reshape(1, -1)).item())
    return weight_diff, weight_sim


def get_weight_diff(agent1, agent2):
    # weight diff, agent class should have layers_for_weight_diff() func
    with torch.no_grad():
        weight_diff, weight_sim = get_diff_sim_from_layers(agent1.layers_for_weight_diff(),
                                                           agent2.layers_for_weight_diff())

        layers_A1, layers_A2, layers_Afc = agent1.layers_for_weight_diff_extra()
        layers_B1, layers_B2, layers_Bfc = agent2.layers_for_weight_diff_extra()
        weight_diff1, weight_sim1 = get_diff_sim_from_layers(layers_A1, layers_B1)
        weight_diff2, weight_sim2 = get_diff_sim_from_layers(layers_A2, layers_B2)
        weight_difffc, weight_simfc = get_diff_sim_from_layers(layers_Afc, layers_Bfc)

        return weight_diff, weight_sim, weight_diff1, weight_sim1, weight_diff2, weight_sim2, weight_difffc, weight_simfc


def get_feature_diff(agent1, agent2, dataset, device, ratio=0.01, seed=0):
    # feature diff: for each data point, get difference of feature from old and new network
    # compute l2 norm of this diff, average over a number of data points.
    # agent class should have features_from_batch() func
    with torch.no_grad():
        n_total_data = dataset['observations'].shape[0]
        # average_feature_l2_norm_list = []
        average_feature_sim_list = []
        average_feature_mse_list = []
        num_feature_timesteps = int(n_total_data * ratio)
        if num_feature_timesteps % 2 == 1:  # avoid potential sampling issue
            num_feature_timesteps = num_feature_timesteps + 1
        np.random.seed(seed)
        idxs_all = np.random.choice(np.arange(0, n_total_data), size=num_feature_timesteps, replace=False)
        batch_size = 1000
        n_done = 0
        i = 0
        while True:
            if n_done >= num_feature_timesteps:
                break
            idxs = idxs_all[i * batch_size:min((i + 1) * batch_size, num_feature_timesteps)]

            batch = index_batch(dataset, idxs)
            batch = batch_to_torch(batch, device)

            old_feature = agent1.features_from_batch_no_grad(batch)
            new_feature = agent2.features_from_batch_no_grad(batch)
            feature_diff = old_feature - new_feature

            # feature_l2_norm = torch.norm(feature_diff, p=2, dim=1, keepdim=True)
            # average_feature_l2_norm_list.append(feature_l2_norm.mean().item())

            feature_mse = torch.mean(feature_diff ** 2).item()
            average_feature_mse_list.append(feature_mse)

            feature_sim = float(F.cosine_similarity(old_feature, new_feature).mean().item())
            average_feature_sim_list.append(feature_sim)
            i += 1
            n_done += 1000
        return np.mean(average_feature_mse_list), np.mean(average_feature_sim_list), num_feature_timesteps


def main():
    variant = get_default_variant_dict()  # this is a dictionary

    # for grid experiments, simply 1. get default params. 2. modify some of the params. 3. change exp name
    exp_name_full = 'testonly'
    data_dir = '/checkpoints'
    logger_kwargs = setup_logger_kwargs_dt(exp_name_full, variant['seed'], data_dir)
    variant["outdir"] = logger_kwargs["output_dir"]
    variant["exp_name"] = logger_kwargs["exp_name"]
    run_single_exp(variant)


def save_extra_dict(variant, logger, dataset,
                    ret_list, ret_normalized_list, iter_list, step_list,
                    agent_init, agent_after_pretrain, agent_e20, agent, best_agent,
                    best_return, best_return_normalized, best_step, best_iter,
                    return_e20, return_normalized_e20, additional_dict=None):
    """get extra dict"""
    # get convergence steps
    conv_k = get_convergence_index(ret_list)
    convergence_iter, convergence_step = iter_list[conv_k], step_list[conv_k]
    # get weight and feature diff
    if agent_e20 is not None:
        e20_weight_diff, e20_weight_sim, wd0_e20, ws0_e20, wd1_e20, ws1_e20, wdfc_e20, wsfc_e20 = get_weight_diff(
            agent_e20, agent_after_pretrain)
        e20_feature_diff, e20_feature_sim, _ = get_feature_diff(agent_e20, agent_after_pretrain, dataset,
                                                                variant['device'])
    else:
        e20_weight_diff, e20_weight_sim = -1, -1
        e20_feature_diff, e20_feature_sim = -1, -1
        wd0_e20, ws0_e20, wd1_e20, ws1_e20, wdfc_e20, wsfc_e20 = -1, -1, -1, -1, -1, -1,
    final_weight_diff, final_weight_sim, wd0_fin, ws0_fin, wd1_fin, ws1_fin, wdfc_fin, wsfc_fin = get_weight_diff(agent,
                                                                                                                  agent_after_pretrain)
    final_feature_diff, final_feature_sim, _ = get_feature_diff(agent, agent_after_pretrain, dataset, variant['device'])
    best_weight_diff, best_weight_sim, wd0_best, ws0_best, wd1_best, ws1_best, wdfc_best, wsfc_best = get_weight_diff(
        best_agent, agent_after_pretrain)
    best_feature_diff, best_feature_sim, num_feature_timesteps = get_feature_diff(best_agent, agent_after_pretrain,
                                                                                  dataset, variant['device'])

    init_weight_diff, init_weight_sim, wd0_init, ws0_init, wd1_init, ws1_init, wdfc_init, wsfc_init = get_weight_diff(
        agent, agent_init)
    init_feature_diff, init_feature_sim, _ = get_feature_diff(agent, agent_init, dataset, variant['device'])

    pre_weight_diff, pre_weight_sim, wd0_pre, ws0_pre, wd1_pre, ws1_pre, wdfc_pre, wsfc_pre = get_weight_diff(
        agent_after_pretrain, agent_init)
    pre_feature_diff, pre_feature_sim, _ = get_feature_diff(agent_after_pretrain, agent_init, dataset,
                                                            variant['device'])

    # save extra dict
    extra_dict = {
        'final_weight_diff': final_weight_diff,
        'final_weight_sim': final_weight_sim,
        'final_feature_diff': final_feature_diff,
        'final_feature_sim': final_feature_sim,

        "final_0_weight_diff": wd0_fin,
        "final_1_weight_diff": wd1_fin,
        "final_fc_weight_diff": wdfc_fin,
        "final_0_weight_sim": ws0_fin,
        "final_1_weight_sim": ws1_fin,
        "final_fc_weight_sim": wsfc_fin,

        'best_weight_diff': best_weight_diff,
        'best_weight_sim': best_weight_sim,
        'best_feature_diff': best_feature_diff,
        'best_feature_sim': best_feature_sim,

        "best_0_weight_diff": wd0_best,
        "best_1_weight_diff": wd1_best,
        "best_fc_weight_diff": wdfc_best,
        "best_0_weight_sim": ws0_best,
        "best_1_weight_sim": ws1_best,
        "best_fc_weight_sim": wsfc_best,

        'init_weight_diff': init_weight_diff,
        'init_weight_sim': init_weight_sim,
        'init_feature_diff': init_feature_diff,
        'init_feature_sim': init_feature_sim,

        "init_0_weight_diff": wd0_init,
        "init_1_weight_diff": wd1_init,
        "init_fc_weight_diff": wdfc_init,
        "init_0_weight_sim": ws0_init,
        "init_1_weight_sim": ws1_init,
        "init_fc_weight_sim": wsfc_init,

        'pre_weight_diff': final_weight_diff,
        'pre_weight_sim': final_weight_sim,
        'pre_feature_diff': final_feature_diff,
        'pre_feature_sim': final_feature_sim,

        "pre_0_weight_diff": wd0_pre,
        "pre_1_weight_diff": wd1_pre,
        "pre_fc_weight_diff": wdfc_pre,
        "pre_0_weight_sim": ws0_pre,
        "pre_1_weight_sim": ws1_pre,
        "pre_fc_weight_sim": wsfc_pre,

        'e20_weight_diff': e20_weight_diff,  # unique to cql due to more training updates
        'e20_weight_sim': e20_weight_sim,
        'e20_feature_diff': e20_feature_diff,
        'e20_feature_sim': e20_feature_sim,

        "e20_0_weight_diff": wd0_e20,
        "e20_1_weight_diff": wd1_e20,
        "e20_fc_weight_diff": wdfc_e20,
        "e20_0_weight_sim": ws0_e20,
        "e20_1_weight_sim": ws1_e20,
        "e20_fc_weight_sim": wsfc_e20,

        'final_test_returns': float(ret_list[-1]),
        'final_test_normalized_returns': float(ret_normalized_list[-1]),
        'best_return': float(best_return),
        'best_return_normalized': float(best_return_normalized),
        'test_returns_e20': float(return_e20),
        'test_normalized_returns_e20': float(return_normalized_e20),

        'convergence_step': convergence_step,
        'convergence_iter': convergence_iter,
        'best_step': best_step,
        'best_iter': best_iter,
        'num_feature_timesteps': num_feature_timesteps,
    }
    if additional_dict is not None:
        extra_dict.update(additional_dict)
    # print()
    # for key, val in extra_dict.items():
    #     print(key, val, type(val))
    logger.save_extra_dict_as_json(extra_dict, 'extra.json')


def get_additional_dict(additional_dict_with_list):
    additional_dict = {}
    for key in additional_dict_with_list:
        additional_dict[key] = np.mean(additional_dict_with_list[key])
    return additional_dict


def run_single_exp(variant):
    if variant['mdppre_same_as_s_and_policy']:
        variant['mdppre_n_action'] = variant['mdppre_n_state']
        variant['mdppre_transition_temperature'] = variant['mdppre_policy_temperature']

    logger = EpochLogger(variant["outdir"], 'progress.csv', variant["exp_name"])
    logger.save_config(variant)

    set_random_seed(variant['seed'])

    env_full = '%s-%s-v2' % (variant['env'], variant['dataset'])
    if variant['env'] == 'antmaze-umaze' and variant['dataset'] == 'play':
        env_full = 'antmaze-umaze-v2'

    eval_sampler = TrajSampler(gym.make(env_full).unwrapped, variant['max_traj_length'])

    pretrain_obs_dim = eval_sampler.env.observation_space.shape[0]
    pretrain_act_dim = eval_sampler.env.action_space.shape[0]

    policy_arch = '-'.join([str(variant['policy_hidden_unit']) for _ in range(variant['policy_hidden_layer'])])
    policy = TanhGaussianPolicy(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=policy_arch,
        log_std_multiplier=variant['policy_log_std_multiplier'],
        log_std_offset=variant['policy_log_std_offset'],
        orthogonal_init=variant['orthogonal_init'],
    )

    qf_arch = '-'.join([str(variant['qf_hidden_unit']) for _ in range(variant['qf_hidden_layer'])])
    qf1 = FullyConnectedQFunctionPretrain(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=qf_arch,
        orthogonal_init=variant['orthogonal_init'],
    )
    qf2 = FullyConnectedQFunctionPretrain(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=qf_arch,
        orthogonal_init=variant['orthogonal_init'],
    )

    target_qf1 = deepcopy(qf1)
    target_qf2 = deepcopy(qf2)

    if variant['cql'].target_entropy >= 0.0:
        variant['cql'].target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    agent = ConservativeSAC_REG(variant['cql'], policy, qf1, qf2, target_qf1, target_qf2, variant)
    agent.torch_to_device(variant['device'])
    agent_init = deepcopy(agent)
    if variant['save_model']:
        save_dict = {'agent': agent_init, 'variant': variant, 'epoch': 0}
        logger.save_dict(save_dict, 'agent_random.pth')

    sampler_policy = SamplerPolicy(policy, variant['device'])

    agent_after_pretrain = deepcopy(agent)

    """offline stage"""
    print("============================ OFFLINE STAGE STARTED! ============================")
    # TODO we can load offline dataset here... but load pretrain dataset earlier
    dataset = get_d4rl_dataset_with_ratio(eval_sampler.env, variant['offline_data_ratio'])
    print("D4RL dataset loaded for", env_full)
    dataset['rewards'] = dataset['rewards'] * variant['reward_scale'] + variant['reward_bias']
    dataset['actions'] = np.clip(dataset['actions'], -variant['clip_action'], variant['clip_action'])
    max_reward = max(dataset['rewards'])
    # when discount is 0.99
    safe_q_max = max_reward * 100 if variant['use_safe_q'] else None
    print('max reward:', max_reward)

    # Modified: Load synthetic MDP data for possible regularization
    if variant['pretrain_mode'] in ['syn_mdp']:
        reg_dataset = get_mdp_dataset_with_ratio(variant['mdppre_n_traj'],
                                                 variant['mdppre_n_state'],
                                                 variant['mdppre_n_action'],
                                                 variant['mdppre_policy_temperature'],
                                                 variant['mdppre_transition_temperature'],
                                                 ratio=variant['pretrain_data_ratio'],
                                                 random_start=variant['mdppre_random_start'])
        np.random.seed(0)
        index2state = 2 * np.random.rand(variant['mdppre_n_state'], pretrain_obs_dim) - 1
        index2action = 2 * np.random.rand(variant['mdppre_n_action'], pretrain_act_dim) - 1
        index2state, index2action = index2state.astype(np.float32), index2action.astype(np.float32)
    elif variant['pretrain_mode'] in ['offline_centroid']:
        mean_sprime = dataset['next_observations'].mean(dim=0)

    best_agent = deepcopy(agent)
    agent_e20, return_e20, return_normalized_e20 = None, 0, 0
    best_step, best_iter = 0, 0
    iter_list, step_list, ret_list, ret_normalized_list = [], [], [], []
    best_return, best_return_normalized = -np.inf, -np.inf
    viskit_metrics = {}
    st = time.time()
    additional_dict_with_list = {}
    for additional in ['feature_diff_last_iter', 'feature_sim_last_iter', 'weight_diff_last_iter',
                       'weight_sim_last_iter',
                       'wd0_li', 'ws0_li', 'wd1_li', 'ws1_li', 'wdfc_li', 'wsfc_li']:
        additional_dict_with_list[additional] = []

    for epoch in range(variant['n_epochs']):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(variant['n_train_step_per_epoch']):
                batch_offline = subsample_batch(dataset, variant['batch_size'])
                batch_offline = batch_to_torch(batch_offline, variant['device'])

                extra_q_loss = None
                if variant['pretrain_mode'] != 'none' and epoch < variant['weight_reg_epochs']:
                    if variant['pretrain_mode'] in ['syn_mdp']:
                        batch_reg = subsample_batch(reg_dataset, variant['batch_size'])
                        batch_reg['observations'] = index2state[batch_reg['observations']]
                        batch_reg['actions'] = index2action[batch_reg['actions']]
                        batch_reg['next_observations'] = index2state[batch_reg['next_observations']]
                    elif variant['pretrain_mode'] in ['offline_centroid']:
                        batch_reg = deepcopy(batch_offline)
                        batch_reg['next_observations'] = np.tile(mean_sprime, (variant['batch_size'], 1))
                    elif variant['pretrain_mode'] in ['offline_self']:
                        batch_reg = batch_offline

                    batch_reg = batch_to_torch(batch_reg, variant['device'])
                    extra_q_loss = agent.pretrain(batch_reg, pretrain_mode=variant['pretrain_mode'])

                metrics.update(prefix_metrics(agent.train(batch_offline, bc=epoch < variant['bc_epochs'],
                                                          safe_q_max=safe_q_max,
                                                          extra_q_loss=extra_q_loss,
                                                          extra_q_loss_discount=variant['extra_q_loss_discount'],
                                                          ), 'sac', connector_string='_'))

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % variant['eval_period'] == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, variant['eval_n_trajs'], deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                metrics['average_normalizd_return'] = np.mean(
                    [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) * 100 for t in trajs]
                )

                # record best return and other things
                iter_list.append(epoch + 1)
                step_list.append((epoch + 1) * variant['n_train_step_per_epoch'])
                ret_normalized_list.append(metrics['average_normalizd_return'])
                ret_list.append(metrics['average_return'])
                if metrics['average_normalizd_return'] > best_return_normalized:
                    best_return = metrics['average_return']
                    best_return_normalized = metrics['average_normalizd_return']
                    best_agent = deepcopy(agent)
                    best_iter = epoch + 1
                    best_step = (epoch + 1) * variant['n_train_step_per_epoch']
                    if variant['save_model']:
                        save_dict = {'agent': best_agent, 'variant': variant, 'epoch': best_iter}
                        logger.save_dict(save_dict, 'agent_best.pth')

            if (epoch + 1) == 20:
                agent_e20 = deepcopy(agent)
                return_e20, return_normalized_e20 = metrics['average_return'], metrics['average_normalizd_return']

            if variant['save_model'] and (epoch + 1) in (100, 300):
                save_dict = {'agent': agent, 'variant': variant, 'epoch': epoch + 1}
                logger.save_dict(save_dict, 'agent_e%d.pth' % (epoch + 1))

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        viskit_metrics.update(metrics)

        logger.log_tabular("Iteration", epoch + 1)
        logger.log_tabular("Steps", (epoch + 1) * variant['n_train_step_per_epoch'])
        logger.log_tabular("TestEpRet", viskit_metrics['average_return'])
        logger.log_tabular("TestEpNormRet", viskit_metrics['average_normalizd_return'])

        things_to_log = ['sac_log_pi', 'sac_policy_loss', 'sac_qf1_loss', 'sac_qf2_loss', 'sac_alpha_loss', 'sac_alpha',
                         'sac_average_qf1', 'sac_average_qf2', 'average_traj_length',
                         'sac_combined_loss', 'sac_cql_conservative_loss', 'sac_qf_average_loss']
        for m in things_to_log:
            if m not in viskit_metrics:
                logger.log_tabular(m, 0)
            else:
                logger.log_tabular(m, viskit_metrics[m])

        logger.log_tabular("total_time", time.time() - st)
        logger.log_tabular("train_time", viskit_metrics["train_time"])
        logger.log_tabular("eval_time", viskit_metrics["eval_time"])
        logger.log_tabular("current_hours", (time.time() - st) / 3600)
        logger.log_tabular("est_total_hours", (variant['n_epochs'] / (epoch + 1) * (time.time() - st)) / 3600)

        logger.dump_tabular()
        sys.stdout.flush()  # flush at end of each epoch for results to show up in hpc

    """get extra dict"""
    # additional_dict = get_additional_dict(additional_dict_with_list)
    save_extra_dict(variant, logger, dataset,
                    ret_list, ret_normalized_list, iter_list, step_list,
                    agent_init, agent_after_pretrain, agent_e20, agent, best_agent,
                    best_return, best_return_normalized, best_step, best_iter,
                    return_e20, return_normalized_e20)
    if variant['save_model']:
        save_dict = {'agent': agent, 'variant': variant, 'epoch': variant['n_epochs']}
        logger.save_dict(save_dict, 'agent_final.pth')


if __name__ == '__main__':
    main()
