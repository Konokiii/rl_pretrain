
# plot_measures = ['best_return_normalized', 'best_weight_diff', 'best_feature_diff',
#                 'final_test_normalized_returns', 'final_weight_diff', 'final_feature_diff',
#                 'best_return_normalized_std', 'final_test_normalized_returns_std', 'convergence_iter',
#                  ]
# plot_y_labels = ['Best Normalized Score', 'Best Weight l2 Diff', 'Best Feature L2 Diff',
#                  'Final Normalized Score', 'Final Weight l2 Diff', 'Final Feature L2 Diff',
#                  'Best Normalized Score Std', 'Final Normalized Score Std', 'Convergence Iter',
#                  ]
plot_measures = ['best_return_normalized', 'best_return_normalized_std', 'convergence_iter',
                 'best_feature_diff', 'best_weight_diff', "best_0_weight_diff", "best_1_weight_diff",
                 'best_feature_sim', 'best_weight_sim', "best_0_weight_sim", "best_1_weight_sim",
                 ]
plot_y_labels = ['Best Normalized Score', 'Best Normalized Score Std', 'Convergence Iter',
                'Best Feature Diff', 'Best Weight Diff', 'Best Weight Diff L0',  'Best Weight Diff L1',
                 'Best Feature Sim', 'Best Weight Sim', 'Best Weight Sim L0', 'Best Weight Sim L1',
                 ]

prefix_list = ['cql_pretrain_epochs',
               # 'cql_layers',
               'rl_datasize',
               'dt_modelsize',
               'dt_pretrain_perturb']
caption_list = ['CQL with different forward dynamics pretraining epochs.',
                # 'CQL layers comparison. Y-axis in each figure shows an average measure across 9 MuJoCo datasets.',
                'with different offline dataset ratio. ',
                'DT with different model sizes.',
                'DT with pretraining, and different perturbation noise std into the pretrained mdoel. ',
                ]



def print_figures_latex(figure_folder, figure_names, sub_figure_captions, caption='', ref_label=''):
    # 12 subfigures, each for one plot measure
    print("\\begin{figure}[htb]")
    print("\\captionsetup[subfigure]{justification=centering}")
    print("\\centering")
    for i in range(len(figure_names)):
        figure_name = figure_names[i]
        sub_figure_caption = sub_figure_captions[i]

        print('\\begin{subfigure}[t]{.32\\linewidth}')
        print('\\centering')
        print('\\includegraphics[width=\\linewidth]{%s/%s}' % (figure_folder, figure_name))
        print('\\caption{%s}' % sub_figure_caption)
        if i in [2, 5, 8]:
            print('\\end{subfigure}\\\\')
        else:
            print('\\end{subfigure}')

    print('\\caption{%s}' % caption)
    print('\\label{%s}' % ref_label)
    print('\\end{figure}')
    print()


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


def gen_cql_curves(figure_folder, datasets, exp_name):
    figure_names = []
    subfigure_captions = [d[0].upper() + d[1:] for d in datasets]

    aggregate_name = 'agg-cql' if not exp_name else 'agg-cql_' + exp_name

    separate_name = 'ind-cql' if not exp_name else 'ind-cql_' + exp_name

    # performance-aggregated
    print("\\begin{figure}[htb]")
    print("\\centering")
    print('\\includegraphics[width=\\linewidth]{%s/%s}' % (figure_folder, aggregate_name + '_TestEpNormRet.png'))
    print('\\caption{%s}' % 'Performance curve of each setting averaged over 12 datasets.')
    print('\\label{%s}' % 'fig:cql-performance-agg-curves')
    print('\\end{figure}')
    print()

    # performance-individual
    for e in datasets:
        figure_names.append(separate_name + '_TestEpNormRet_%s.png' % e)

    caption = 'Learning curves for CQL, CQL with same task RL data pretraining, and CQL with MDP pretraining.'
    ref_label = 'fig:cql-performance-curves'
    print_figures_latex(
        figure_folder,
        figure_names,
        subfigure_captions,
        caption,
        ref_label,
    )


gen_cql_curves(figure_folder='dzx_figures/cql_antmaze', datasets=LOCOMOTION_12_DATASETS, exp_name=None)

