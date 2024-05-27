import os
import joblib
from tqdm import tqdm

import numpy as np
from torch import Tensor
from torch.nn import functional as F


def check_input(value):
    if isinstance(value, (float, int)):
        assert value > 0, "Numerical input must be a positive real number."
    elif isinstance(value, str):
        assert value == 'inf', "String input must be 'inf'."
    else:
        raise TypeError("Input must be a positive real number or the string 'inf'.")


def softmax_with_torch(x, temperature):
    return F.softmax(Tensor(x / temperature), dim=0).numpy()


def generate_mdp_data(n_traj, max_length, n_state, n_action, policy_temperature, transition_temperature, random_start=False, save_dir='./mdpdata'):
    # Check if temperatures are valid inputs
    check_input(policy_temperature)
    check_input(transition_temperature)

    # Set data file name and path
    data_file_name = 'mdp_traj%d_len%d_ns%d_na%d_pt%s_tt%s_rs%s.pkl' % (n_traj, max_length, n_state, n_action,
                                                                        str(policy_temperature),
                                                                        str(transition_temperature),
                                                                        str(random_start))
    data_save_path = os.path.join(save_dir, data_file_name)

    # Return if the MDP dataset is already generated
    if os.path.exists(data_save_path):
        dataset = joblib.load(data_save_path)
        print("Synthetic MDP data has already been generated. Loaded from:", data_save_path)
        return dataset

    # Set total number of synthetic MDP data
    n_data = n_traj * max_length

    #  Infinite policy and transition temperature is equivalent to generating i.i.d synthetic MDP data.
    if policy_temperature == transition_temperature == 'inf':
        states = np.random.randint(n_state, size=n_data)
        next_states = np.concatenate((states[1:], np.random.randint(n_state, size=1)))
        actions = np.random.randint(n_action, size=n_data)

    #  Otherwise, generate synthetic data according to MDP probability distributions.
    else:
        states = np.zeros(n_data, dtype=int)
        actions = np.zeros(n_data, dtype=int)
        next_states = np.zeros(n_data, dtype=int)
        i = 0
        for j_traj in tqdm(range(n_traj)):
            if random_start:
                np.random.seed(j_traj)
            else:
                np.random.seed(n_traj)

            state = np.random.randint(n_state)
            for t in range(max_length):
                states[i] = state
                # For each step, an action is taken, and a next state is decided. We assume that a fixed policy is
                # generating the data, so the action distribution only depends on the state.
                if policy_temperature == 'inf':
                    action_probs = None
                else:
                    np.random.seed(state)
                    action_probs = softmax_with_torch(np.random.rand(n_action), policy_temperature)

                np.random.seed(42 + j_traj * 1000 + t * 333)
                action = np.random.choice(n_action, p=action_probs)
                actions[i] = action

                if transition_temperature == 'inf':
                    next_state_probs = None
                else:
                    np.random.seed(state * 888 + action * 777)
                    next_state_probs = softmax_with_torch(np.random.rand(n_state), transition_temperature)

                np.random.seed(666 + j_traj * 1000 + t * 333)
                next_state = np.random.choice(n_state, p=next_state_probs)
                next_states[i] = next_state

                state = next_state
                i += 1

    data_dict = {'observations': states,
                 'actions': actions,
                 'next_observations': next_states}

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(data_dict, data_save_path)
    print("Synthetic MDP data generation finished. Saved to:", data_save_path)
    return data_dict
