from plot_utils.quick_plot_helper import quick_plot_with_full_name, quick_scatter_plot
from plot_utils.log_alias import *
from plot_global_variables import *


def get_full_names_with_datasets(base_names, datasets):
    # 'datasets' can be offline or online envs, offline envs will need to also have dataset name.
    # Output: List[List[base_name+datasets]]
    n = len(base_names)
    to_return = []
    for i in range(n):
        new_list = []
        for dataset in datasets:
            full_name = base_names[i] + '_' + dataset
            new_list.append(full_name)
        to_return.append(new_list)
    return to_return


def plot_cql_performance_curves(labels, base_names, shift=None, exp_name=None, datasets=LOCOMOTION_12_DATASETS,
                                y_axis=D4RL_NORMALIZED_SCORE_NAME):
    # aggregate
    aggregate_name = 'agg-cql' if not exp_name else 'agg-cql_' + exp_name
    quick_plot_with_full_name(
        labels,
        get_full_names_with_datasets(base_names, datasets),
        shift=shift,
        save_name_prefix=aggregate_name,
        base_data_folder_path=DATA_PATH,
        save_folder_path=FIGURE_SAVE_PATH,
        y_value=[y_axis],
        x_to_use=D4RL_X_AXIS,
        ymax=None,
        smooth=CURVE_SMOOTH,
        axis_font_size=AXIS_FONT_SIZE,
        legend_font_size=LEGEND_FONT_SIZE
    )

    # separate
    separate_name = 'ind-cql' if not exp_name else 'ind-cql_' + exp_name
    for dataset in datasets:
        quick_plot_with_full_name(
            labels,
            get_full_names_with_datasets(base_names, [dataset]),
            shift=shift,
            save_name_prefix=separate_name,
            base_data_folder_path=DATA_PATH,
            save_folder_path=FIGURE_SAVE_PATH,
            y_value=[y_axis],
            x_to_use=D4RL_X_AXIS,
            ymax=None,
            save_name_suffix=dataset,
            smooth=CURVE_SMOOTH,
            axis_font_size=AXIS_FONT_SIZE,
            legend_font_size=LEGEND_FONT_SIZE
        )


# # Finetune updates vs synthetic hyperparameters:
# labels = [
#     'CQL',
#     'CQL_MDP_S10',
#     'CQL_MDP_S100',
#     'CQL_MDP_S1000',
#     'CQL_MDP_S10000',
# ]
# base_names = [
#     iclr_cql,
#     iclr_cql_mdp_ns10,
#     iclr_cql_mdp_ns100,
#     iclr_cql_mdp_ns1000,
#     iclr_cql_mdp_ns10000,
# ]
# shift = [0, 20, 20, 20, 20]
# plot_cql_performance_curves(labels, base_names, shift=shift, exp_name='fineUps_S')
#
# labels = [
#     'CQL',
#     'CQL_MDP_\u03C40.1',
#     'CQL_MDP_\u03C41',
#     'CQL_MDP_\u03C410',
#     'CQL_IID',
# ]
# base_names = [
#     iclr_cql,
#     iclr_cql_mdp_t01,
#     iclr_cql_mdp_t1,
#     iclr_cql_mdp_t10,
#     iclr_cql_iid_preT100k,
# ]
# shift = [0, 20, 20, 20, 20]
# plot_cql_performance_curves(labels, base_names, shift=shift, exp_name='fineUps_Temp')
#
# labels = [
#     'CQL',
#     'CQL_MDP_10K',
#     'CQL_MDP_40K',
#     'CQL_MDP_100K',
#     'CQL_MDP_500K',
# ]
# base_names = [
#     iclr_cql,
#     iclr_cql_mdp_preT10k,
#     iclr_cql_mdp_preT40k,
#     iclr_cql_mdp_preT100k,
#     iclr_cql_mdp_preT500k,
# ]
# shift = [0, 2, 8, 20, 100]
# plot_cql_performance_curves(labels, base_names, shift=shift, exp_name='fineUps_preT')
# #
# # Antmaze:
# labels = [
#     'CQL',
#     'CQL_MDP',
#     'CQL_SAME',
#     'CQL_LAG',
#     'CQL_MDP_LAG',
#     'CQL_SAME_LAG'
# ]
# base_names = [
#     cql_subopt_antmaze,
#     cql_subopt_antmaze_mdp,
#     cql_subopt_antmaze_same,
#     cql_subopt_antmaze_lag,
#     cql_subopt_antmaze_mdp_lag,
#     cql_subopt_antmaze_same_lag
# ]
# shift = [0, 20, 20, 20, 20, 20]
# plot_cql_performance_curves(labels, base_names, shift=shift, exp_name='antmaze_subopt', datasets=ANTMAZE_6_DATASETS)
#
# labels = [
#     'CQL',
#     'CQL_MDP',
#     'CQL_SAME',
#     'CQL_LAG',
#     'CQL_MDP_LAG',
#     'CQL_SAME_LAG'
# ]
# base_names = [
#     cql_tuned_antmaze,
#     cql_tuned_antmaze_mdp,
#     cql_tuned_antmaze_same,
#     cql_tuned_antmaze_lag,
#     cql_tuned_antmaze_mdp_lag,
#     cql_tuned_antmaze_same_lag
# ]
# shift = [0, 20, 20, 20, 20, 20]
# plot_cql_performance_curves(labels, base_names, shift=shift, exp_name='antmaze_tuned', datasets=ANTMAZE_6_DATASETS)

# # aclpaper:
# labels = [
#     'CQL',
#     'CQL_MDP',
#     'CQL_IDENTITY',
#     'CQL_CASE_MAPPING'
# ]
# base_names = [
#     iclr_cql,
#     iclr_cql_mdp_t1,
#     cql_identity,
#     cql_case_mapping
# ]
# shift = [0, 20, 20, 20]
# plot_cql_performance_curves(labels, base_names, shift=shift, exp_name='aclpaper', datasets=LOCOMOTION_12_DATASETS)

# # Offline data study:
labels = [
    'CQL',
    'ns10',
    'ns100',
    'ns1000',
    'ns10000'
]
base_names = [[cql_offRatio_baseline[f'cql_offR{offR}_baseline'] for offR in [0.1, 0.2, 0.4, 0.6, 0.8, 1]]]
base_names += [[cql_offRatio_mdp[f'cql_offR{offR}_ns{ns}_tt1_preEp20'] for offR in [0.1, 0.2, 0.4, 0.6, 0.8, 1]] for ns in [10, 100, 1000, 10000]]
quick_scatter_plot(labels, base_names, datasets=LOCOMOTION_12_DATASETS, measure='last_four_normalized',
                   exp_name='ablation_ns', data_path=DATA_PATH, save_folder_path=FIGURE_SAVE_PATH,
                   legend_font_size=LEGEND_FONT_SIZE, axis_font_size=AXIS_FONT_SIZE)
labels = [
    'CQL',
    '\u03C40.1',
    '\u03C41',
    '\u03C410',
    'IID'
]
base_names = [[cql_offRatio_baseline[f'cql_offR{offR}_baseline'] for offR in [0.1, 0.2, 0.4, 0.6, 0.8, 1]]]
base_names += [[cql_offRatio_mdp[f'cql_offR{offR}_ns100_tt{t}_preEp20'] for offR in [0.1, 0.2, 0.4, 0.6, 0.8, 1]] for t in [0.1, 1, 10, 'inf3']]
quick_scatter_plot(labels, base_names, datasets=LOCOMOTION_12_DATASETS, measure='last_four_normalized',
                   exp_name='ablation_temp', data_path=DATA_PATH, save_folder_path=FIGURE_SAVE_PATH,
                   legend_font_size=LEGEND_FONT_SIZE, axis_font_size=AXIS_FONT_SIZE)
labels = [
    'CQL',
    '10K',
    '40K',
    '100K',
    '500K'
]
base_names = [[cql_offRatio_baseline[f'cql_offR{offR}_baseline'] for offR in [0.1, 0.2, 0.4, 0.6, 0.8, 1]]]
base_names += [[cql_offRatio_mdp[f'cql_offR{offR}_ns100_tt1_preEp{ep}'] for offR in [0.1, 0.2, 0.4, 0.6, 0.8, 1]] for ep in [2, 8, 20, 100]]
quick_scatter_plot(labels, base_names, datasets=LOCOMOTION_12_DATASETS, measure='last_four_normalized',
                   exp_name='ablation_preUps', data_path=DATA_PATH, save_folder_path=FIGURE_SAVE_PATH,
                   legend_font_size=LEGEND_FONT_SIZE, axis_font_size=AXIS_FONT_SIZE)







