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


def plot_cql_performance_curves(labels, base_names, shift=None, max_seeds=None, cutoff=None, scatter_plot=False, exp_name=None, datasets=LOCOMOTION_12_DATASETS,
                                y_axis=D4RL_NORMALIZED_SCORE_NAME, xlabel='Number of Updates', xticks=None):
    # aggregate
    aggregate_name = 'agg-cql' if not exp_name else 'agg-cql_' + exp_name
    quick_plot_with_full_name(
        labels,
        get_full_names_with_datasets(base_names, datasets),
        shift=shift,
        max_seeds=max_seeds,
        cutoff=cutoff,
        scatter_plot=scatter_plot,
        save_name_prefix=aggregate_name,
        base_data_folder_path=DATA_PATH,
        save_folder_path=FIGURE_SAVE_PATH,
        y_value=[y_axis],
        x_to_use=D4RL_X_AXIS,
        xlabel=xlabel,
        xticks=xticks,
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
            max_seeds=max_seeds,
            cutoff=cutoff,
            save_name_prefix=separate_name,
            base_data_folder_path=DATA_PATH,
            save_folder_path=FIGURE_SAVE_PATH,
            y_value=[y_axis],
            x_to_use=D4RL_X_AXIS,
            xlabel=xlabel,
            ymax=None,
            save_name_suffix=dataset,
            smooth=CURVE_SMOOTH,
            axis_font_size=AXIS_FONT_SIZE,
            legend_font_size=LEGEND_FONT_SIZE
        )

# # CQL + 100K more:
# labels = [
#     'CQL',
#     'CQL+MDP',
#     'CQL+IID'
# ]
# base_names = [
#     cql_2x,
#     iclr_cql_mdp_ns100,
#     iclr_cql_iid_preT100k
# ]
# shift = [0, -20, -20]
# cutoff = [220, 0, 0]
# plot_cql_performance_curves(labels, base_names, shift=shift, max_seeds=5, cutoff=cutoff, xlabel='Total Number of Updates',
#                             xticks=range(0, int(1.2e6), int(1e5)), exp_name='longer_baseline')

# Finetune updates vs synthetic hyperparameters:
xticks = [0.1e6, 0.2e6, 0.4e6, 0.6e6, 0.8e6, 1.0e6]
labels = [
    'CQL',
    'S10',
    'S100',
    'S1,000',
    'S10,000',
]
base_names = [
    iclr_cql,
    iclr_cql_mdp_ns10,
    iclr_cql_mdp_ns100,
    iclr_cql_mdp_ns1000,
    iclr_cql_mdp_ns10000,
]
plot_cql_performance_curves(labels, base_names, shift=None, scatter_plot=True, xlabel='Number of Finetune Updates', xticks=xticks, exp_name='fineUps_S')

labels = [
    'CQL',
    '\u03C40.1',
    '\u03C41',
    '\u03C410',
    'CQL+IID',
]
base_names = [
    iclr_cql,
    iclr_cql_mdp_t01,
    iclr_cql_mdp_t1,
    iclr_cql_mdp_t10,
    iclr_cql_iid_preT100k,
]
plot_cql_performance_curves(labels, base_names, shift=None, scatter_plot=True, xlabel='Number of Finetune Updates', xticks=xticks, exp_name='fineUps_Temp')

labels = [
    'CQL',
    'Pre10K',
    'Pre40K',
    'Pre100K',
    'Pre500K',
]
base_names = [
    iclr_cql,
    iclr_cql_mdp_preT10k,
    iclr_cql_mdp_preT40k,
    iclr_cql_mdp_preT100k,
    iclr_cql_mdp_preT500k,
]
plot_cql_performance_curves(labels, base_names, shift=None, scatter_plot=True, xlabel='Number of Finetune Updates', xticks=xticks, exp_name='fineUps_preT')

# # Antmaze:
# labels = [
#     'CQL',
#     'CQL+MDP',
#     'CQL+SAME',
#     'CQL+LAG',
#     'CQL+MDP+LAG',
#     'CQL+SAME+LAG'
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
# plot_cql_performance_curves(labels, base_names, shift=shift, max_seeds=5, xlabel='Total Number of Updates', exp_name='antmaze_subopt', datasets=ANTMAZE_6_DATASETS)
#
# labels = [
#     'CQL',
#     'CQL+MDP',
#     'CQL+SAME',
#     'CQL+LAG',
#     'CQL+MDP+LAG',
#     'CQL+SAME+LAG'
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
# plot_cql_performance_curves(labels, base_names, shift=shift, max_seeds=5, xlabel='Total Number of Updates', exp_name='antmaze_tuned_a3', datasets=ANTMAZE_6_DATASETS)

# # aclpaper:
# labels = [
#     'CQL',
#     'CQL+MDP',
#     'CQL+Identity',
#     'CQL+Mapping'
# ]
# base_names = [
#     iclr_cql,
#     iclr_cql_mdp_t1,
#     cql_identity,
#     cql_case_mapping
# ]
# shift = [0, 20, 20, 20]
# plot_cql_performance_curves(labels, base_names, shift=shift, max_seeds=5, xlabel='Total Number of Updates', exp_name='simpler_synthetic', datasets=LOCOMOTION_12_DATASETS)

# # Offline data study:
# labels = [
#     'CQL',
#     'S10',
#     'S100',
#     'S1,000',
#     'S10,000'
# ]
# base_names = [[cql_offRatio_baseline[f'cql_offR{offR}_baseline'] for offR in [0.1, 0.2, 0.4, 0.6, 0.8, 1]]]
# base_names += [[cql_offRatio_mdp[f'cql_offR{offR}_ns{ns}_pt1_preEp20'] for offR in [0.1, 0.2, 0.4, 0.6, 0.8, 1]] for ns in [10, 100, 1000, 10000]]
# quick_scatter_plot(labels, base_names, datasets=LOCOMOTION_12_DATASETS, measure='last_four_normalized',
#                    exp_name='agg-offratio_ablation_ns', data_path=DATA_PATH, save_folder_path=FIGURE_SAVE_PATH,
#                    legend_font_size=LEGEND_FONT_SIZE, axis_font_size=AXIS_FONT_SIZE, max_seeds=5)
# labels = [
#     'CQL',
#     '\u03C40.1',
#     '\u03C41',
#     '\u03C410',
#     'IID'
# ]
# base_names = [[cql_offRatio_baseline[f'cql_offR{offR}_baseline'] for offR in [0.1, 0.2, 0.4, 0.6, 0.8, 1]]]
# base_names += [[cql_offRatio_mdp[f'cql_offR{offR}_ns100_pt{t}_preEp20'] for offR in [0.1, 0.2, 0.4, 0.6, 0.8, 1]] for t in [0.1, 1, 10, 'inf3']]
# quick_scatter_plot(labels, base_names, datasets=LOCOMOTION_12_DATASETS, measure='last_four_normalized',
#                    exp_name='agg-offratio_ablation_temp', data_path=DATA_PATH, save_folder_path=FIGURE_SAVE_PATH,
#                    legend_font_size=LEGEND_FONT_SIZE, axis_font_size=AXIS_FONT_SIZE, max_seeds=5)
# labels = [
#     'CQL',
#     'Pre10K',
#     'Pre40K',
#     'Pre100K',
#     'Pre500K'
# ]
# base_names = [[cql_offRatio_baseline[f'cql_offR{offR}_baseline'] for offR in [0.1, 0.2, 0.4, 0.6, 0.8, 1]]]
# base_names += [[cql_offRatio_mdp[f'cql_offR{offR}_ns100_pt1_preEp{ep}'] for offR in [0.1, 0.2, 0.4, 0.6, 0.8, 1]] for ep in [2, 8, 20, 100]]
# quick_scatter_plot(labels, base_names, datasets=LOCOMOTION_12_DATASETS, measure='last_four_normalized',
#                    exp_name='agg-offratio_ablation_preUps', data_path=DATA_PATH, save_folder_path=FIGURE_SAVE_PATH,
#                    legend_font_size=LEGEND_FONT_SIZE, axis_font_size=AXIS_FONT_SIZE, max_seeds=5)







