"""Use this to help us know where the training logs are"""

"""
DT related
"""
# DT baselines
dt = 'dt-rerun-data_size_dt_1.0'
chibiT = 'chibiT-rerun'

# DT data size experiment
dt_finetune_data_size_0_1 = 'dt-rerun-data_size_dt_0.1'
dt_finetune_data_size_0_25 = 'dt-rerun-data_size_dt_0.25'
dt_finetune_data_size_0_5 = 'dt-rerun-data_size_dt_0.5'
dt_finetune_data_size_0_75 = 'dt-rerun-data_size_dt_0.75'
dt_finetune_data_size_1 = dt

chibiT_finetune_data_size_0_1 = 'chibiT-rerun-data_size_dt_0.1'
chibiT_finetune_data_size_0_25 = 'chibiT-rerun-data_size_dt_0.25'
chibiT_finetune_data_size_0_5 = 'chibiT-rerun-data_size_dt_0.5'
chibiT_finetune_data_size_0_75 = 'chibiT-rerun-data_size_dt_0.75'
chibiT_finetune_data_size_1 = chibiT

# DT with different model sizes
dt_model_size_default = dt
dt_model_size_4layer_256 = 'dt_embed_dim256_n_layer4_n_head4'
dt_model_size_6layer_512 = 'dt_embed_dim512_n_layer6_n_head8'
dt_model_size_12layer_768 = 'dt_embed_dim768_n_layer12_n_head12'

chibiT_model_size_default = chibiT
chibiT_model_size_4layer_256 = 'chibiT_embed_dim256_n_layer4_n_head4'
chibiT_model_size_6layer_512 = 'chibiT_embed_dim512_n_layer6_n_head8'
chibiT_model_size_12layer_768 = 'chibiT_embed_dim768_n_layer12_n_head12'


# DT with markov chain pretraining
dt_mc_1step_vocab10 = 'chibiT-rerun-syn_ngram1_nvocab10_temperature1.0'
dt_mc_1step_vocab100 = 'chibiT-rerun-syn_ngram1_nvocab100_temperature1.0'
dt_mc_1step_vocab1000 = 'chibiT-rerun-syn_ngram1_nvocab1000_temperature1.0'
dt_mc_1step_vocab10000 = 'chibiT-rerun-syn_ngram1_nvocab10000_temperature1.0'
dt_mc_1step_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature1.0'
dt_mc_1step_vocab100000 = 'chibiT-rerun-syn_ngram1_nvocab100000_temperature1.0'

dt_mc_2step_vocab50257 = 'chibiT-rerun-syn_ngram2_nvocab50257_temperature1.0'
dt_mc_3step_vocab50257 = 'chibiT-rerun-syn_ngram3_nvocab50257_temperature1.0'
dt_mc_4step_vocab50257 = 'chibiT-rerun-syn_ngram4_nvocab50257_temperature1.0'
dt_mc_5step_vocab50257 = 'chibiT-rerun-syn_ngram5_nvocab50257_temperature1.0'

dt_mc_temp0_1_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature0.1'
dt_mc_temp0_2_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature0.2'
dt_mc_temp0_4_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature0.4'
dt_mc_temp0_8_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature0.8'
dt_mc_temp1_0_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature1.0'
dt_mc_temp10_0_vocab50257 = 'chibiT-rerun-syn_ngram1_nvocab50257_temperature10.0'


dt_mc_2step_vocab100 = 'chibiT-rerun-syn_ngram2_nvocab100_temperature1.0'
dt_mc_3step_vocab100 = 'chibiT-rerun-syn_ngram3_nvocab100_temperature1.0'
dt_mc_4step_vocab100 = 'chibiT-rerun-syn_ngram4_nvocab100_temperature1.0'
dt_mc_5step_vocab100 = 'chibiT-rerun-syn_ngram5_nvocab100_temperature1.0'

dt_mc_temp0_1_vocab100 = 'chibiT-rerun-syn_ngram1_nvocab100_temperature0.1'
dt_mc_temp0_2_vocab100 = 'chibiT-rerun-syn_ngram1_nvocab100_temperature0.2'
dt_mc_temp0_4_vocab100 = 'chibiT-rerun-syn_ngram1_nvocab100_temperature0.4'
dt_mc_temp0_8_vocab100 = 'chibiT-rerun-syn_ngram1_nvocab100_temperature0.8'
dt_mc_temp10_0_vocab100 = 'chibiT-rerun-syn_ngram1_nvocab100_temperature10.0'


dt_same_data = 'same_new_ft'


"""
CQL related
"""

# CQL baselines
cql_base = 'cqlr3_prenone_l2_qflrs1'
cql_fd_pretrain = 'cqlr3_preq_sprime_l2_qflrs1'
cql_random_pretrain = 'cqlr3_prerand_q_sprime_l2'
cql_random_1000_state = 'cqlr3_prerandom_fd_1000_state_l2'

# CQL data size
cql_no_pretrain_0_1_data = 'cqlr3_prenone_l2_dr0.1'
cql_no_pretrain_0_25_data = 'cqlr3_prenone_l2_dr0.25'
cql_no_pretrain_0_5_data = 'cqlr3_prenone_l2_dr0.5'
cql_no_pretrain_0_75_data = 'cqlr3_prenone_l2_dr0.75'
cql_no_pretrain_1_data = cql_base


cql_pretrain_0_1_data = 'cqlr3_preq_sprime_l2_dr0.1'
cql_pretrain_0_25_data = 'cqlr3_preq_sprime_l2_dr0.25'
cql_pretrain_0_5_data = 'cqlr3_preq_sprime_l2_dr0.5'
cql_pretrain_0_75_data = 'cqlr3_preq_sprime_l2_dr0.75'
cql_pretrain_1_data = cql_fd_pretrain

# with target network update



# CQL pretrain epoch

# CQL pretrain and offline data size variants
cql_fd_pretrain_data_ratio_0_01 = 'cqlr3_preq_sprime_l2_pdr0.01'
cql_fd_pretrain_data_ratio_0_1 = 'cqlr3_preq_sprime_l2_pdr0.1'
cql_fd_pretrain_data_ratio_0_25 = 'cqlr3_preq_sprime_l2_pdr0.25'
cql_fd_pretrain_data_ratio_0_5 = 'cqlr3_preq_sprime_l2_pdr0.5'
cql_fd_pretrain_data_ratio_0_75 = 'cqlr3_preq_sprime_l2_pdr0.75'
cql_fd_pretrain_data_ratio_1 = cql_fd_pretrain


cql_mdp_pretrain_data_ratio_0_01 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr0.01'
cql_mdp_pretrain_data_ratio_0_1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr0.1'
cql_mdp_pretrain_data_ratio_0_25 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr0.25'
cql_mdp_pretrain_data_ratio_0_5 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr0.5'
cql_mdp_pretrain_data_ratio_0_75 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr0.75'
cql_mdp_pretrain_data_ratio_1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue'


cql_fd_finetune_data_ratio_0_01 = 'cqlr3_preq_sprime_l2_dr0.01'
cql_fd_finetune_data_ratio_0_1 = 'cqlr3_preq_sprime_l2_dr0.1'
cql_fd_finetune_data_ratio_0_25 = 'cqlr3_preq_sprime_l2_dr0.25'
cql_fd_finetune_data_ratio_0_5 = 'cqlr3_preq_sprime_l2_dr0.5'
cql_fd_finetune_data_ratio_0_75 = 'cqlr3_preq_sprime_l2_dr0.75'
cql_fd_finetune_data_ratio_1 = cql_fd_pretrain

cql_mdp_finetune_data_ratio_0_01 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.01'
cql_mdp_finetune_data_ratio_0_1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.1'
cql_mdp_finetune_data_ratio_0_25 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.25'
cql_mdp_finetune_data_ratio_0_5 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.5'
cql_mdp_finetune_data_ratio_0_75 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.75'
cql_mdp_finetune_data_ratio_1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue'


cql_fd_finetune_both_ratio_0_01 = 'cqlr3_preq_sprime_l2_bothdr0.01'
cql_fd_finetune_both_ratio_0_1 = 'cqlr3_preq_sprime_l2_bothdr0.1'
cql_fd_finetune_both_ratio_0_25 = 'cqlr3_preq_sprime_l2_bothdr0.25'
cql_fd_finetune_both_ratio_0_5 = 'cqlr3_preq_sprime_l2_bothdr0.5'
cql_fd_finetune_both_ratio_0_75 = 'cqlr3_preq_sprime_l2_bothdr0.75'
cql_fd_finetune_both_ratio_1 = cql_fd_pretrain


cql_mdp_with_target_hard_update = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr1_True'
cql_mdp_with_target_hard_update_1 = cql_mdp_with_target_hard_update
cql_mdp_with_target_hard_update_0_5 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.5_True'
cql_mdp_with_target_hard_update_0_1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.1_True'
cql_mdp_with_target_hard_update_0_25 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.25_True'
cql_mdp_with_target_hard_update_0_75 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.75_True'
cql_mdp_with_target_hard_update_0_01 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue_pdr1_dr0.01_True'


cql_rl_with_target_hard_update = 'cqlr3_preq_sprime_l2_pdr1_dr1_True'


cql_rl_with_target_hard_update_0_01 = 'cqlr3_preq_sprime_l2_pdr1_dr0.01_True'
cql_rl_with_target_hard_update_0_1 = 'cqlr3_preq_sprime_l2_pdr1_dr0.1_True'
cql_rl_with_target_hard_update_0_25 = 'cqlr3_preq_sprime_l2_pdr1_dr0.25_True'
cql_rl_with_target_hard_update_0_5 = 'cqlr3_preq_sprime_l2_pdr1_dr0.5_True'
cql_rl_with_target_hard_update_0_75 = 'cqlr3_preq_sprime_l2_pdr1_dr0.75_True'
cql_rl_with_target_hard_update_1 = cql_rl_with_target_hard_update


# CQL 3x data
# 'q_noact_sprime', 'q_sprime_3x', 'proj0_q_sprime_3x', 'proj1_q_sprime_3x'
cql_fd_3x_data = 'cqlr3_preq_sprime_3x_l2'
cql_fd_3x_data_with_projection = 'cqlr3_preproj0_q_sprime_3x_l2'
cql_fd_3x_data_cross_task = 'cqlr3_preproj1_q_sprime_3x_l2'

# CQL alternative pretrain scheme
cql_no_action_predict_next_state = 'cqlr3_preq_noact_sprime_l2'

# CQL cross domain
cql_fd_pretrain_same_task_with_projection = 'cqlr3_preproj0_q_sprime_l2'
cql_fd_pretrain_cross_task1 = 'cqlr3_preproj1_q_sprime_l2'
cql_fd_pretrain_cross_task2 = 'cqlr3_preproj2_q_sprime_l2'

# CQL MDP
cql_mdp_pretrain_nstate_base = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd20_sameTrue'

cql_mdp_pretrain_nstate1 = 'cqlr3_premdp_q_sprime_l2_ns1_pt1_sd20_sameTrue'
cql_mdp_pretrain_nstate10 = 'cqlr3_premdp_q_sprime_l2_ns10_pt1_sd20_sameTrue'
cql_mdp_pretrain_nstate100 = 'cqlr3_premdp_q_sprime_l2_ns100_pt1_sd20_sameTrue'
cql_mdp_pretrain_nstate1000 = cql_mdp_pretrain_nstate_base
cql_mdp_pretrain_nstate10000= 'cqlr3_premdp_q_sprime_l2_ns10000_pt1_sd20_sameTrue'
cql_mdp_pretrain_nstate50257= 'cqlr3_premdp_q_sprime_l2_ns50257_pt1_sd20_sameTrue'

cql_mdp_pretrain_temperature0_01 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt0.01_sd20_sameTrue'
cql_mdp_pretrain_temperature0_1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt0.1_sd20_sameTrue'
cql_mdp_pretrain_temperature1 = cql_mdp_pretrain_nstate_base
cql_mdp_pretrain_temperature10 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt10_sd20_sameTrue'
cql_mdp_pretrain_temperature100 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt100_sd20_sameTrue'

cql_mdp_pretrain_state_action_dim1 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd1_sameTrue'
cql_mdp_pretrain_state_action_dim5 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd5_sameTrue'
cql_mdp_pretrain_state_action_dim20 = cql_mdp_pretrain_nstate_base
cql_mdp_pretrain_state_action_dim50 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd50_sameTrue'
cql_mdp_pretrain_state_action_dim200 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd200_sameTrue'
cql_mdp_pretrain_state_action_dim1000 = 'cqlr3_premdp_q_sprime_l2_ns1000_pt1_sd1000_sameTrue'

cql_mdp_pretrain_same_dim_no_projection = 'cqlr3_premdp_same_noproj_l2_ns1000_pt1_sameTrue'
cql_mdp_pretrain_same_dim_with_projection = 'cqlr3_premdp_same_proj_l2_ns1000_pt1_sameTrue'


# new ones
cql_jul = 'cqlr3n_prenone_l2'
cql_jul_fd_pretrain = 'cqlr3n_preq_sprime_l2'
cql_jul_mdp_noproj_s1_t1 = 'cqlr3n_premdp_same_noproj_l2_ns1_pt1_sameTrue'
cql_jul_mdp_noproj_s10_t1 = 'cqlr3n_premdp_same_noproj_l2_ns10_pt1_sameTrue'
cql_jul_mdp_noproj_s100_t1 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue'
cql_jul_mdp_noproj_s1000_t1 = 'cqlr3n_premdp_same_noproj_l2_ns1000_pt1_sameTrue'
cql_jul_mdp_noproj_s10000_t1 = 'cqlr3n_premdp_same_noproj_l2_ns10000_pt1_sameTrue'
cql_jul_mdp_noproj_s100000_t1 = 'cqlr3n_premdp_same_noproj_l2_ns100000_pt1_sameTrue'

cql_jul_mdp_noproj_s100_t0_0001 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt0.0001_sameTrue'
cql_jul_mdp_noproj_s100_t0_001 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt0.001_sameTrue'
cql_jul_mdp_noproj_s100_t0_01 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt0.01_sameTrue'
cql_jul_mdp_noproj_s100_t0_1 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt0.1_sameTrue'
cql_jul_mdp_noproj_s100_t10 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt10_sameTrue'
cql_jul_mdp_noproj_s100_t100 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt100_sameTrue'
cql_jul_mdp_noproj_s100_t1000 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1000_sameTrue'


cql_jul_pdr0_001_dr0_001 = 'cqlr3n_prenone_l2_pdr0.001_dr0.001'
cql_jul_pdr0_001_dr1 = 'cqlr3n_prenone_l2_pdr0.001_dr1'
cql_jul_pdr1_dr0_001 = 'cqlr3n_prenone_l2_pdr1_dr0.001'
cql_jul_pdr1_dr1 = cql_jul

cql_jul_same_data_pdr0_001_dr0_001 = 'cqlr3n_preq_sprime_l2_pdr0.001_dr0.001'
cql_jul_same_data_pdr0_001_dr1 = 'cqlr3n_preq_sprime_l2_pdr0.001_dr1'
cql_jul_same_data_pdr1_dr0_001 = 'cqlr3n_preq_sprime_l2_pdr1_dr0.001'
cql_jul_same_data_pdr1_dr1 = cql_jul_fd_pretrain

cql_jul_mdp_pdr0_001_dr0_001 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_pdr0.001_dr0.001'
cql_jul_mdp_pdr0_001_dr1 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_pdr0.001_dr1'
cql_jul_mdp_pdr1_dr0_001 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_pdr1_dr0.001'
cql_jul_mdp_pdr1_dr1 = cql_jul_mdp_noproj_s100_t1

# 07/13/2023 exps:
cql_1x = 'cqlr3n_prenone_l2_ep200'
cql_2x = 'cqlr3n_prenone_l2_ep400'

cql_mdp_t0001 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt0.001_sameTrue'
cql_mdp_t001 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt0.01_sameTrue'
cql_mdp_t01 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt0.1_sameTrue'
cql_mdp_t1 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue'
cql_mdp_t10 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt10_sameTrue'
cql_mdp_t100 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt100_sameTrue'
cql_mdp_t1000 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1000_sameTrue'
cql_mdp_tinf = 'cqlr3n_premdp_same_noproj_l2_ns100_ptinf_sameTrue'
cql_mdp_tinf2 = 'cqlr3n_premdp_same_noproj_l2_ns100_ptinf2_sameTrue'

cql_same_mse = 'cqlr3_preq_sprime_l2'
cql_same_mle = 'cqlr3_preq_mle_l2'

# same pretraining from Watcher:
cql_same_new = 'cqlr3n_preq_sprime_l2'

# 08/02/2023 ant + fix_sprime:
cql_mdp_tfix = 'cqlr3n_premdp_same_noproj_l2_ns100_ptfix_sprime_sameTrue'
cql_mdp_tmean = 'cqlr3n_premdp_same_noproj_l2_ns100_ptmean_sprime_sameTrue'
cql_same = 'cqlr3n_preq_sprime_l2'

# 08/07/2023 cluster results:
cql_mdp_sigma001N = 'cqlr3n_premdp_same_noproj_l2_ns100_ptsigma0.01N_sameTrue'
cql_mdp_sigma01N = 'cqlr3n_premdp_same_noproj_l2_ns100_ptsigma0.1N_sameTrue'
cql_mdp_sigma1N = 'cqlr3n_premdp_same_noproj_l2_ns100_ptsigma1N_sameTrue'
cql_mdp_sigma2N = 'cqlr3n_premdp_same_noproj_l2_ns100_ptsigma2N_sameTrue'
cql_mdp_sigma001S = 'cqlr3n_premdp_same_noproj_l2_ns100_ptsigma0.01S_sameTrue'
cql_mdp_sigma01S = 'cqlr3n_premdp_same_noproj_l2_ns100_ptsigma0.1S_sameTrue'
cql_mdp_sigma1S = 'cqlr3n_premdp_same_noproj_l2_ns100_ptsigma1S_sameTrue'
cql_mdp_sigma2S = 'cqlr3n_premdp_same_noproj_l2_ns100_ptsigma2S_sameTrue'

# 08/10/2023 n_state ablation:
cql_mdp_ns1 = 'cqlr3n_premdp_same_noproj_l2_ns1_pt1_sameTrue'
cql_mdp_ns10 = 'cqlr3n_premdp_same_noproj_l2_ns10_pt1_sameTrue'
cql_mdp_ns100 = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue'
cql_mdp_ns1000 = 'cqlr3n_premdp_same_noproj_l2_ns1000_pt1_sameTrue'
cql_mdp_ns10000 = 'cqlr3n_premdp_same_noproj_l2_ns10000_pt1_sameTrue'
cql_mdp_ns100000 = 'cqlr3n_premdp_same_noproj_l2_ns100000_pt1_sameTrue'

# 08/12/2023 less pretraining:
cql_pR0001_pT125k, cql_pR0001_pT250k, cql_pR0001_pT500k, cql_pR0001_pT750k, cql_pR0001_pT1m = \
    'cqlr3n_premdp_same_noproj_preRatio0.001_preEp25_l2_ns100_pt1_sameTrue',\
    'cqlr3n_premdp_same_noproj_preRatio0.001_preEp50_l2_ns100_pt1_sameTrue',\
    'cqlr3n_premdp_same_noproj_preRatio0.001_preEp100_l2_ns100_pt1_sameTrue',\
    'cqlr3n_premdp_same_noproj_preRatio0.001_preEp150_l2_ns100_pt1_sameTrue',\
    'cqlr3n_premdp_same_noproj_preRatio0.001_preEp200_l2_ns100_pt1_sameTrue',

cql_pR0005_pT125k, cql_pR0005_pT250k, cql_pR0005_pT500k, cql_pR0005_pT750k, cql_pR0005_pT1m = \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preEp25_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preEp50_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preEp100_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preEp150_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preEp200_l2_ns100_pt1_sameTrue',

cql_pR001_pT125k, cql_pR001_pT250k, cql_pR001_pT500k, cql_pR001_pT750k, cql_pR001_pT1m = \
    'cqlr3n_premdp_same_noproj_preRatio0.01_preEp25_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.01_preEp50_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.01_preEp100_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.01_preEp150_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.01_preEp200_l2_ns100_pt1_sameTrue',

cql_pR005_pT125k, cql_pR005_pT250k, cql_pR005_pT500k, cql_pR005_pT750k, cql_pR005_pT1m = \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preEp25_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preEp50_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preEp100_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preEp150_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preEp200_l2_ns100_pt1_sameTrue',

cql_pR01_pT125k, cql_pR01_pT250k, cql_pR01_pT500k, cql_pR01_pT750k, cql_pR01_pT1m = \
    'cqlr3n_premdp_same_noproj_preRatio0.1_preEp25_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.1_preEp50_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.1_preEp100_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.1_preEp150_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.1_preEp200_l2_ns100_pt1_sameTrue',

cql_pR025_pT125k, cql_pR025_pT250k, cql_pR025_pT500k, cql_pR025_pT750k, cql_pR025_pT1m = \
    'cqlr3n_premdp_same_noproj_preRatio0.25_preEp25_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.25_preEp50_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.25_preEp100_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.25_preEp150_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.25_preEp200_l2_ns100_pt1_sameTrue',

cql_pR05_pT125k, cql_pR05_pT250k, cql_pR05_pT500k, cql_pR05_pT750k, cql_pR05_pT1m = \
    'cqlr3n_premdp_same_noproj_preRatio0.5_preEp25_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.5_preEp50_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.5_preEp100_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.5_preEp150_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.5_preEp200_l2_ns100_pt1_sameTrue',

cql_pR075_pT125k, cql_pR075_pT250k, cql_pR075_pT500k, cql_pR075_pT750k, cql_pR075_pT1m = \
    'cqlr3n_premdp_same_noproj_preRatio0.75_preEp25_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.75_preEp50_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.75_preEp100_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.75_preEp150_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.75_preEp200_l2_ns100_pt1_sameTrue',

cql_pR1_pT125k, cql_pR1_pT250k, cql_pR1_pT500k, cql_pR1_pT750k, cql_pR1_pT1m = \
    'cqlr3n_premdp_same_noproj_preRatio1_preEp25_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio1_preEp50_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio1_preEp100_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio1_preEp150_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio1_preEp200_l2_ns100_pt1_sameTrue',

# 08/19/2023:
cql_pR0001_pT1k, cql_pR0001_pT5k, cql_pR0001_pT10k, cql_pR0001_pT20k, cql_pR0001_pT50k = \
    'cqlr3n_premdp_same_noproj_preRatio0.001_preStep1000_preEp1_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.001_preStep1000_preEp5_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.001_preStep1000_preEp10_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.001_preStep1000_preEp20_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.001_preStep1000_preEp50_l2_ns100_pt1_sameTrue',

cql_pR0005_pT1k, cql_pR0005_pT5k, cql_pR0005_pT10k, cql_pR0005_pT20k, cql_pR0005_pT50k = \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preStep1000_preEp1_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preStep1000_preEp5_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preStep1000_preEp10_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preStep1000_preEp20_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.005_preStep1000_preEp50_l2_ns100_pt1_sameTrue',

cql_pR001_pT1k, cql_pR001_pT5k, cql_pR001_pT10k, cql_pR001_pT20k, cql_pR001_pT50k = \
    'cqlr3n_premdp_same_noproj_preRatio0.01_preStep1000_preEp1_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.01_preStep1000_preEp5_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.01_preStep1000_preEp10_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.01_preStep1000_preEp20_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.01_preStep1000_preEp50_l2_ns100_pt1_sameTrue',

cql_pR005_pT1k, cql_pR005_pT5k, cql_pR005_pT10k, cql_pR005_pT20k, cql_pR005_pT50k = \
    'cqlr3n_premdp_same_noproj_preRatio0.05_preStep1000_preEp1_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.05_preStep1000_preEp5_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.05_preStep1000_preEp10_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.05_preStep1000_preEp20_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.05_preStep1000_preEp50_l2_ns100_pt1_sameTrue',

cql_pR01_pT1k, cql_pR01_pT5k, cql_pR01_pT10k, cql_pR01_pT20k, cql_pR01_pT50k = \
    'cqlr3n_premdp_same_noproj_preRatio0.1_preStep1000_preEp1_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.1_preStep1000_preEp5_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.1_preStep1000_preEp10_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.1_preStep1000_preEp20_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.1_preStep1000_preEp50_l2_ns100_pt1_sameTrue',

cql_pR025_pT1k, cql_pR025_pT5k, cql_pR025_pT10k, cql_pR025_pT20k, cql_pR025_pT50k = \
    'cqlr3n_premdp_same_noproj_preRatio0.25_preStep1000_preEp1_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.25_preStep1000_preEp5_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.25_preStep1000_preEp10_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.25_preStep1000_preEp20_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.25_preStep1000_preEp50_l2_ns100_pt1_sameTrue',

cql_pR05_pT1k, cql_pR05_pT5k, cql_pR05_pT10k, cql_pR05_pT20k, cql_pR05_pT50k = \
    'cqlr3n_premdp_same_noproj_preRatio0.5_preStep1000_preEp1_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.5_preStep1000_preEp5_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.5_preStep1000_preEp10_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.5_preStep1000_preEp20_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.5_preStep1000_preEp50_l2_ns100_pt1_sameTrue',

cql_pR075_pT1k, cql_pR075_pT5k, cql_pR075_pT10k, cql_pR075_pT20k, cql_pR075_pT50k = \
    'cqlr3n_premdp_same_noproj_preRatio0.75_preStep1000_preEp1_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.75_preStep1000_preEp5_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.75_preStep1000_preEp10_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.75_preStep1000_preEp20_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio0.75_preStep1000_preEp50_l2_ns100_pt1_sameTrue',

cql_pR1_pT1k, cql_pR1_pT5k, cql_pR1_pT10k, cql_pR1_pT20k, cql_pR1_pT50k = \
    'cqlr3n_premdp_same_noproj_preRatio1_preStep1000_preEp1_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio1_preStep1000_preEp5_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio1_preStep1000_preEp10_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio1_preStep1000_preEp20_l2_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_preRatio1_preStep1000_preEp50_l2_ns100_pt1_sameTrue',

cql_mdp_2x = 'cqlr3n_premdp_same_noproj_l2_ep400_ns100_pt1_sameTrue'

cql_finetune_slow067, cql_finetune_slow033, cql_finetune_slow01, cql_finetune_slow001 = \
    'cqlr3n_premdp_same_noproj_l2_lr0.67_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_l2_lr0.33_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_l2_lr0.1_ns100_pt1_sameTrue', \
    'cqlr3n_premdp_same_noproj_l2_lr0.01_ns100_pt1_sameTrue',

# 2023/08/27 Guess Init:
cql_init_layerG = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_initlayerGuass'
cql_init_modelG = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_initmodelGuass'
cql_init_wholeG = 'cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_initwholeGuass'

# 2023/08/30 20 seeds: 'new_iclr' results are with random start, while 'iclr' exps have fixed start.
# They all have wrong normalized score. Run 'fix_cql_normalized_score' to correct them.
iclr_cql = 'iclr_cqlr3n_prenone_l2'  # Without seed 4096 but 4069

iclr_cql_mdp_preT1 = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps1_preEp1'
iclr_cql_mdp_preT1k = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps1000_preEp1'
iclr_cql_mdp_preT10k = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps5000_preEp2'
iclr_cql_mdp_preT20k = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps5000_preEp4'
iclr_cql_mdp_preT25k = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps5000_preEp5'
iclr_cql_mdp_preT40k = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps5000_preEp8'
iclr_cql_mdp_preT50k = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps5000_preEp10'
iclr_cql_mdp_preT100k = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps5000_preEp20'
iclr_cql_mdp_preT500k = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps5000_preEp100'
iclr_cql_mdp_preT1m = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps5000_preEp200'
iclr_cql_mdp_preT2m = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps5000_preEp400'

iclr_cql_mdp_ns10 = 'iclr_cqlr3n_premdp_same_noproj_l2_ns10_pt1_sameTrue_preUps5000_preEp20'
iclr_cql_mdp_ns100 = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps5000_preEp20'
iclr_cql_mdp_ns1000 = 'iclr_cqlr3n_premdp_same_noproj_l2_ns1000_pt1_sameTrue_preUps5000_preEp20'
iclr_cql_mdp_ns10000 = 'iclr_cqlr3n_premdp_same_noproj_l2_ns10000_pt1_sameTrue_preUps5000_preEp20'
iclr_cql_mdp_ns100000 = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100000_pt1_sameTrue_preUps5000_preEp20'

iclr_cql_mdp_t0001 = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt0.001_sameTrue_preUps5000_preEp20'
iclr_cql_mdp_t001 = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt0.01_sameTrue_preUps5000_preEp20'
iclr_cql_mdp_t01 = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt0.1_sameTrue_preUps5000_preEp20'
iclr_cql_mdp_t1 = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt1_sameTrue_preUps5000_preEp20'
iclr_cql_mdp_t10 = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt10_sameTrue_preUps5000_preEp20'
iclr_cql_mdp_t100 = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_pt100_sameTrue_preUps5000_preEp20'
iclr_cql_iid_preT1m = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_ptinf3_sameTrue_preUps5000_preEp200'
iclr_cql_iid_preT100k = 'iclr_cqlr3n_premdp_same_noproj_l2_ns100_ptinf3_sameTrue_preUps5000_preEp20'

#2023/10/08 LR study:
cql_2x_3e5_safeQ = 'postICLR_tuned_cqlr3n_prenone_l2_ep400'
mdp_2x_3e5_safeQ = 'postICLR_tuned_cqlr3n_premdp_same_noproj_l2_ep400_ns100_pt1_sameTrue_preUps5000_preEp20'

cql_2x_3e5 = 'postICLR_tuned_cqlr3n_prenone_l2_ep400_safeQFalse'
cql_2x_3e4 = 'postICLR_tuned_cqlr3n_prenone_l2_ep400_safeQFalse_actorLR0.0003'
cql_2x_6e4 = 'postICLR_tuned_cqlr3n_prenone_l2_ep400_safeQFalse_actorLR0.0006'
cql_2x_1e3 = 'postICLR_tuned_cqlr3n_prenone_l2_ep400_safeQFalse_actorLR0.001'

mdp_2x_3e5 = 'postICLR_tuned_cqlr3n_premdp_same_noproj_l2_ep400_safeQFalse_ns100_pt1_sameTrue_preUps5000_preEp20'
mdp_2x_3e4 = 'postICLR_tuned_cqlr3n_premdp_same_noproj_l2_ep400_safeQFalse_actorLR0.0003_ns100_pt1_sameTrue_preUps5000_preEp20'
mdp_2x_6e4 = 'postICLR_tuned_cqlr3n_premdp_same_noproj_l2_ep400_safeQFalse_actorLR0.0006_ns100_pt1_sameTrue_preUps5000_preEp20'
mdp_2x_1e3 = 'postICLR_tuned_cqlr3n_premdp_same_noproj_l2_ep400_safeQFalse_actorLR0.001_ns100_pt1_sameTrue_preUps5000_preEp20'

cql_best_l2 = 'postICLR_best_cqlr3n_prenone_l2_ep200_safeQFalse'
cql_best_l3 = 'postICLR_best_cqlr3n_prenone_l3_ep200_safeQFalse'

mdp_best_l2 = 'postICLR_best_cqlr3n_premdp_same_noproj_l2_ep200_safeQFalse_ns100_pt1_sameTrue_preUps5000_preEp20'
mdp_best_l3 = 'postICLR_best_cqlr3n_premdp_same_noproj_l3_ep200_safeQFalse_ns100_pt1_sameTrue_preUps5000_preEp20'

# 2023/10/19:
cql_crude_init = 'Init_cqlr3n_premdp_same_noproj_l2_initcrude_init'







