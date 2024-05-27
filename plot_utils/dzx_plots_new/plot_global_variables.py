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
AXIS_FONT_SIZE = 20
LEGEND_FONT_SIZE = 14

# Load/Save paths:
DATA_PATH = '../../code/checkpoints/rebuttal/antmaze'
FIGURE_SAVE_PATH = '../../figures/antmaze10r'
