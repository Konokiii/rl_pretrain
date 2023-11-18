from plot_utils.quick_plot_helper import quick_plot_with_full_name
from plot_utils.log_alias import *

# Locomotion dataset names:
MUJOCO_3_ENVS = ['hopper', 'walker2d', 'halfcheetah']
MUJOCO_4_ENVS = ['hopper', 'walker2d', 'halfcheetah', 'ant']
MUJOCO_3_DATASETS = ['medium', 'medium-replay', 'medium-expert']
LOCOMOTION_9_DATASETS = ['%s_%s' % (e, d) for e in MUJOCO_3_ENVS for d in MUJOCO_3_DATASETS]
LOCOMOTION_12_DATASETS = ['%s_%s' % (e, d) for e in MUJOCO_4_ENVS for d in MUJOCO_3_DATASETS]

# Antmaze dataset names:
ANTMAZE_3_LAYOUTS = ['antmaze-umaze', 'antmaze-medium', 'antmaze-large']
ANTMAZE_2_RULES = ['play', 'diverse']
ANTMAZE_6_DATASETS = ['%s_%s' % (e, d) for d in ANTMAZE_2_RULES for e in ANTMAZE_3_LAYOUTS]

# x/y Axis names:
D4RL_SCORE_NAME = 'TestEpRet'
D4RL_NORMALIZED_SCORE_NAME = 'TestEpNormRet'
D4RL_X_AXIS = 'Steps'

# Plot settings:
CURVE_SMOOTH = 5
FONT_SIZE = 10

# Load/Save paths:
DATA_PATH = '../../code/checkpoints/antmaze'
FIGURE_SAVE_PATH = '../../figures/antmaze'


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


def plot_cql_performance_curves(labels, base_names, exp_name=None, datasets=LOCOMOTION_12_DATASETS,
                                y_axis=D4RL_NORMALIZED_SCORE_NAME):
    # aggregate
    aggregate_name = 'agg-cql' if not exp_name else 'agg-cql_' + exp_name
    quick_plot_with_full_name(
        labels,
        get_full_names_with_datasets(base_names, datasets),
        save_name_prefix=aggregate_name,
        base_data_folder_path=DATA_PATH,
        save_folder_path=FIGURE_SAVE_PATH,
        y_value=[y_axis],
        x_to_use=D4RL_X_AXIS,
        ymax=None,
        smooth=CURVE_SMOOTH,
        axis_font_size=FONT_SIZE
    )

    # separate
    separate_name = 'ind-cql' if not exp_name else 'ind-cql_' + exp_name
    for dataset in datasets:
        quick_plot_with_full_name(
            labels,
            get_full_names_with_datasets(base_names, [dataset]),
            save_name_prefix=separate_name,
            base_data_folder_path=DATA_PATH,
            save_folder_path=FIGURE_SAVE_PATH,
            y_value=[y_axis],
            x_to_use=D4RL_X_AXIS,
            ymax=None,
            save_name_suffix=dataset,
            smooth=CURVE_SMOOTH,
            axis_font_size=FONT_SIZE
        )


labels = [

]
base_names = [

]
plot_cql_performance_curves(labels, base_names, exp_name='')

