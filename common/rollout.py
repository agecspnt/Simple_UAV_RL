import numpy as np
import torch # For potential tensor operations if agents output tensors directly, though usually numpy
import os
import time # For unique log file names

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon if hasattr(args, 'anneal_epsilon') else 0.0 # Handle if not set
        self.min_epsilon = args.min_epsilon

        # print(f"Initialized RolloutWorker with Epsilon: {self.epsilon}, Min Epsilon: {self.min_epsilon}, Anneal Rate: {self.anneal_epsilon / self.args.anneal_steps if hasattr(self.args, 'anneal_steps') and self.args.anneal_steps > 0 else 'N/A'}")
        # The above print statement was modified to avoid potential error if anneal_steps is 0 or not present for anneal_epsilon
        anneal_rate_info = 'N/A'
        if hasattr(self.args, 'anneal_steps') and self.args.anneal_steps > 0 and self.anneal_epsilon > 0:
            anneal_rate_info = self.anneal_epsilon / self.args.anneal_steps
        print(f"Initialized RolloutWorker. Default Epsilon (if not passed to generate_episode): {self.epsilon}, Min Epsilon: {self.min_epsilon}, Anneal Rate (config): {anneal_rate_info}")


    def generate_episode(self, episode_num=None, evaluate=False, epsilon=None, log_output_dir=None):
        """
        Generates a single episode of interaction.
        Args:
            episode_num (int, optional): The current episode number, for logging or other uses.
            evaluate (bool): Whether this episode is for evaluation (affects exploration).
            epsilon (float, optional): The exploration rate to use. If None, defaults to 0 if evaluate is True, 
                                     else uses self.epsilon (worker's internal state, which is not recommended if Runner manages it).
                                     It is best if Runner always provides this.
            log_output_dir (str, optional): The specific directory to save logs for this episode if evaluation logging is active.
        Returns:
            dict: A dictionary containing the episode's transitions and statistics.
                  Keys: 'o', 's', 'u', 'r', 'o_next', 's_next', 'avail_u', 
                        'avail_u_next', 'u_onehot', 'padded', 'terminated',
                        'episode_reward', 'episode_length'
        """
        if self.args.cuda: # If policy uses CUDA, hidden states might be on GPU
            self.agents.policy.init_hidden(1) # Batch size 1 for episode generation

        # Episode buffer to store transitions
        o, u, r, s, avail_u, u_onehot, terminated, padded = [], [], [], [], [], [], [], []
        o_next, s_next, avail_u_next = [], [], []
        episode_env_info = [] # <-- Add this line to store env_info per step

        # Reset environment and get initial state/obs
        # The environment's reset() method should return initial obs and state
        # obs_all_agents: list of observations for each agent
        # state: global state
        obs_all_agents, state = self.env.reset() 
        
        terminated_flag = False
        win_flag = False # If the environment provides success info
        episode_reward = 0
        step = 0

        # Epsilon for this episode
        # current_epsilon = 0 if evaluate else self.epsilon # Old way
        if epsilon is None: # Fallback if epsilon is not provided by caller
            current_epsilon = 0.0 if evaluate else self.epsilon
            # print(f"Warning: generate_episode using internal epsilon: {current_epsilon}") # For debugging
        else:
            current_epsilon = epsilon

        last_action = np.zeros((self.args.n_agents, self.args.n_actions)) # For recurrent policies if needed

        while not terminated_flag and step < self.episode_limit:
            obs_tensor_list = [torch.tensor(obs_i, dtype=torch.float32) for obs_i in obs_all_agents]
            
            actions = [] # List to store integer actions for each agent
            actions_onehot = [] # List to store one-hot actions

            for agent_id in range(self.n_agents):
                # Get available actions from environment if possible, otherwise assume all are available
                avail_actions_agent = self.env.get_avail_agent_actions(agent_id) if hasattr(self.env, 'get_avail_agent_actions') else np.ones(self.n_actions)
                
                # Agent chooses action
                # The choose_action method of Agents class should handle recurrent states internally if needed.
                action_int = self.agents.choose_action(obs_all_agents[agent_id], 
                                                       last_action[agent_id], 
                                                       agent_id, 
                                                       avail_actions_agent, 
                                                       current_epsilon, 
                                                       evaluate)
                
                # Convert action to one-hot
                action_onehot = np.zeros(self.n_actions)
                action_onehot[action_int] = 1
                
                actions.append(action_int)
                actions_onehot.append(action_onehot)
                last_action[agent_id] = action_onehot # Update last action

            # Environment step
            # Env expects a list/array of integer actions
            reward, terminated_flag, env_info = self.env.step(actions) 
            episode_env_info.append(env_info) # <-- Add this line to store env_info
            
            # Get next obs and state
            obs_all_agents_next = self.env.get_obs()
            state_next = self.env.get_state()

            # Store transition
            o.append(obs_all_agents)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1])) # Store actions as column vector
            u_onehot.append(actions_onehot)
            r.append([reward]) # Reward is a scalar, store as list for consistency in shape
            avail_u.append([self.env.get_avail_agent_actions(i) if hasattr(self.env, 'get_avail_agent_actions') else np.ones(self.n_actions) for i in range(self.n_agents)])
            
            o_next.append(obs_all_agents_next)
            s_next.append(state_next)
            avail_u_next.append([self.env.get_avail_agent_actions(i) if hasattr(self.env, 'get_avail_agent_actions') else np.ones(self.n_actions) for i in range(self.n_agents)])


            padded.append([0.]) # 0 for not padded
            terminated.append([1.0 if terminated_flag else 0.]) # 1 if terminated

            episode_reward += reward
            step += 1
            
            # Update current obs and state for next iteration
            obs_all_agents = obs_all_agents_next
            state = state_next

            if evaluate and hasattr(self.env, 'render'): # Render if in eval mode and env supports it
                self.env.render()

        # Handle last observation if episode didn't terminate due to true termination
        # but due to episode limit.
        # The last o_next, s_next should be the final observation.
        # If terminated by limit, the last transition is still valid.

        # Make sure all lists are of the same length (episode_limit or actual steps)
        # If episode ends early, pad the rest of the episode.
        for _ in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            s.append(np.zeros(self.state_shape))
            u.append(np.zeros((self.n_agents, 1)))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.]) # 1 for padded
            terminated.append([1.0]) # Mark as terminated for padding

        episode_transitions = {
            'o': np.array(o),
            's': np.array(s),
            'u': np.array(u),
            'r': np.array(r),
            'o_next': np.array(o_next),
            's_next': np.array(s_next),
            'avail_u': np.array(avail_u),
            'avail_u_next': np.array(avail_u_next),
            'u_onehot': np.array(u_onehot),
            'padded': np.array(padded),
            'terminated': np.array(terminated),
            'env_info': episode_env_info # <-- Add this line
        }
        
        # Add episode stats
        stats = {
            "episode_reward": episode_reward,
            "episode_length": step,
            "epsilon": current_epsilon # Log epsilon used for this episode
        }
        if 'is_success' in env_info: # If environment provides success metric
            stats["is_success"] = env_info['is_success']
        
        # Anneal epsilon (if not evaluating)
        # This was commented out before, and should remain so, as Runner handles annealing.
        # if not evaluate:
        #     self.epsilon = max(self.min_epsilon, self.epsilon - self.anneal_epsilon)

        log_file_path = None # Initialize log_file_path
        if evaluate and log_output_dir is not None: # Check for the specific log_output_dir for logging
            log_file_path = self.write_log(episode_transitions, stats, episode_num, log_output_dir)
        elif evaluate and self.args.log_dir is not None:
            # Fallback or warning if log_output_dir is not provided but old self.args.log_dir is
            # This case should ideally not be hit if Runner is updated correctly.
            print(f"Warning: Evaluation logging triggered but specific log_output_dir not provided. Falling back to self.args.log_dir: {self.args.log_dir}")
            log_file_path = self.write_log(episode_transitions, stats, episode_num, self.args.log_dir) # Use specific dir

        return episode_transitions, stats, log_file_path # <-- Modified to return log_file_path

    def write_log(self, episode_data, episode_stats, episode_num, target_log_dir):
        """
        Writes detailed interaction logs for an evaluation episode to a specified directory.
        Log format: text file, one line per step.
        Args:
            target_log_dir (str): The directory where the log file should be saved.
        """
        # Check if episode_num indicates it's not the first evaluation episode (e.g., not ending with _ep0)
        # episode_num is a string like "eval_T12345_ep15"
        if episode_num is not None and isinstance(episode_num, str) and "_ep" in episode_num:
            try:
                ep_val = int(episode_num.split("_ep")[-1])
                if ep_val != 0:
                    return None # Do not log if not the first evaluation episode
            except ValueError:
                # If parsing fails, proceed with original behavior (or add specific error handling)
                pass # Or print a warning, or log anyway depending on desired strictness

        if not target_log_dir: # If no target directory is provided, don't log.
            print("Warning: write_log called without a target_log_dir. Skipping log.")
            return None

        if not os.path.exists(target_log_dir):
            try:
                os.makedirs(target_log_dir)
            except OSError as e:
                print(f"Error creating log directory {target_log_dir}: {e}")
                return None
        
        # Create a unique log file name, e.g., alg_time_eval_epN.log
        # Timestamp is now part of the parent folder name, so log file name can be simpler if desired
        # but keeping original structure with timestamp for now, as it might still be useful for individual file sorting
        timestamp = time.strftime("%Y%m%d-%H%M%S") 
        log_file_name = f"{self.args.alg}_{timestamp}_eval_ep{episode_num if episode_num is not None else 'undef'}.log"
        log_path = os.path.join(target_log_dir, log_file_name)

        with open(log_path, "w") as f:
            f.write(f"Episode: {episode_num if episode_num is not None else 'N/A'}, Algorithm: {self.args.alg}\n")
            f.write(f"Total Reward: {episode_stats['episode_reward']:.2f}, Length: {episode_stats['episode_length']}\n")
            if "is_success" in episode_stats:
                f.write(f"Success: {episode_stats['is_success']}\n")

            # Add BS locations to the log if available in the environment
            if hasattr(self.env, 'BS_locations'):
                bs_locs = self.env.BS_locations
                # Format for consistency with visualizer parsing: [[x1, y1], [x2, y2]]
                if isinstance(bs_locs, np.ndarray):
                    bs_locs_list = bs_locs.tolist()
                else:
                    bs_locs_list = list(bs_locs) # Ensure it's a list of lists
                
                # Ensure formatting of numbers to float with one decimal place for consistency if desired, or just str()
                bs_locs_str = "[" + ", ".join([f"[{loc[0]:.1f}, {loc[1]:.1f}]" for loc in bs_locs_list]) + "]"
                f.write(f"BS Locations: {bs_locs_str}\n")
            else:
                f.write(f"BS Locations: Not available in env object or attribute name mismatch\n")

            f.write("-" * 30 + "\n")
            f.write("Step | Agent Actions | Reward | Terminated | UAV Locations (if available) | Collisions (from env_info)\n")
            f.write("-" * 30 + "\n")

            # Iterate through the actual steps taken in the episode
            for step_idx in range(episode_stats['episode_length']):
                actions_step = episode_data['u'][step_idx].flatten() # (n_agents,)
                reward_step = episode_data['r'][step_idx][0]
                terminated_step = episode_data['terminated'][step_idx][0]
                env_info_step = episode_data['env_info'][step_idx] if 'env_info' in episode_data and step_idx < len(episode_data['env_info']) else {} # <-- Get env_info for the step
                
                log_line = f"{step_idx:4d} | {actions_step} | {reward_step:7.2f} | {terminated_step:3.0f} | "
                
                # Try to get UAV locations from state or obs if interpretable for logging
                # This is highly environment-specific.
                # Assuming self.env.current_location exists and is relevant after a step,
                # but this data should ideally come from `env_info` or be part of `s` or `o` if logged.
                # For detailed logging, `env_info` from `env.step()` should carry such details.
                # Let's assume `env_info` from the *last* step (which is not stored per step here) might have details.
                # Or, we can try to reconstruct from `s` if its structure is known.
                
                # For now, let's make a placeholder.
                # The `DETAILED_DESIGN.md` for `Multi_UAV_env.py` does include `current_location`
                # and `is_collision` as attributes that are updated in `step`.
                # The `env_info` from `step` includes `collisions` and `is_success`.
                # We would need to modify `generate_episode` to store `env_info` per step if we want detailed logs from it.
                
                # Simpler log:
                # locations_str = "N/A" 
                # collisions_str = "N/A"
                # If we want to log current_location from the env, we need env to expose it or put it in state.
                # State `s` contains normalized locations. We could denormalize them.
                # `s[step_idx]` contains state at that step.
                # state components: arrive_state (N_AV), bs_assoc (N_AV*N_BS), location_norm (N_AV*2)
                
                state_at_step = episode_data['s'][step_idx]
                num_av = self.args.n_agents
                loc_start_idx = num_av + (num_av * self.env.n_bs if hasattr(self.env, 'n_bs') else num_av * 1) # Assuming n_bs=1 if not found
                
                uav_locs_norm = state_at_step[loc_start_idx : loc_start_idx + num_av * 2].reshape(num_av, 2)
                uav_locs_real = uav_locs_norm * self.env.max_dist if hasattr(self.env, 'max_dist') else uav_locs_norm * 1500 # Default max_dist

                locations_str = "[" + ", ".join([f"({loc[0]:.1f},{loc[1]:.1f})" for loc in uav_locs_real]) + "]"
                log_line += f"{locations_str} | "
                
                # Log collisions from env_info
                collisions_str = "N/A"
                if 'collisions' in env_info_step: # Assuming env_info contains a 'collisions' key (e.g., a list/array of booleans for each agent or a summary)
                    # Adjust formatting based on how 'collisions' is structured in env_info
                    # For example, if it's a list of booleans per agent:
                    if isinstance(env_info_step['collisions'], (list, np.ndarray)) and len(env_info_step['collisions']) == self.n_agents:
                        collisions_str = "[" + ", ".join(["C" if coll else "-" for coll in env_info_step['collisions']]) + "]"
                    else: # Otherwise, just convert to string
                        collisions_str = str(env_info_step['collisions'])
                
                log_line += f"{collisions_str}" # <-- Modified to log actual collision data

                f.write(log_line + "\n")
            
            f.write("-" * 30 + "\n")
            print(f"Evaluation log written to {log_path}")
        return log_path


if __name__ == '__main__':
    import sys
    import os
    # Add project root to sys.path for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # Assuming common is one level below project root
    sys.path.insert(0, project_root)

    from common.arguments import get_common_args
    # Need a mock environment and agents for testing
    
    class MockEnv:
        def __init__(self, args_env):
            self.n_a_agents = args_env.n_agents
            self.n_actions = args_env.n_actions
            self.episode_limit = args_env.episode_limit
            self.obs_shape = args_env.obs_shape
            self.state_shape = args_env.state_shape
            self.max_dist = 1500 # Example
            self.n_bs = 1 # Example for state parsing in log

            self._step = 0

        def reset(self):
            self._step = 0
            obs = [np.random.rand(self.obs_shape) for _ in range(self.n_a_agents)]
            state = np.random.rand(self.state_shape)
            return obs, state

        def step(self, actions):
            self._step += 1
            reward = np.random.rand() - 0.5 # Random reward
            done = self._step >= self.episode_limit
            info = {'is_success': done and np.random.rand() > 0.5, 'collisions': np.random.randint(0,2)}
            
            obs_next = [np.random.rand(self.obs_shape) for _ in range(self.n_a_agents)]
            state_next = np.random.rand(self.state_shape)
            # In a real env, these would be set before returning
            self.current_obs = obs_next 
            self.current_state = state_next
            return reward, done, info

        def get_obs(self): # Needed by rollout worker after step
            return [np.random.rand(self.obs_shape) for _ in range(self.n_a_agents)] # self.current_obs
        
        def get_state(self): # Needed by rollout worker after step
            return np.random.rand(self.state_shape) # self.current_state

        def get_avail_agent_actions(self, agent_id): # Mock
            return np.ones(self.n_actions)

    class MockAgents:
        def __init__(self, args_agents):
            self.n_agents = args_agents.n_agents
            self.n_actions = args_agents.n_actions
            self.args = args_agents # For cuda checks etc.
            # Mock policy for init_hidden
            self.policy = type('MockPolicy', (object,), {'init_hidden': lambda self_policy, bs: None})()


        def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate):
            if evaluate: # Greedy
                return np.argmax(np.random.rand(self.n_actions) * avail_actions) # "Random" greedy for test
            if np.random.rand() < epsilon: # Explore
                return np.random.choice(np.where(avail_actions == 1)[0])
            return np.argmax(np.random.rand(self.n_actions) * avail_actions) # "Random" greedy for test

    print("--- Testing RolloutWorker ---")
    # Use common_args for a more complete set of arguments
    # Need to parse them first if this script is run directly.
    # For simplicity, create a subset of args needed by RolloutWorker, Env, Agents.
    class TestArgs:
        def __init__(self):
            self.episode_limit = 5 # Short episodes for test
            self.n_actions = 4
            self.n_agents = 2
            self.state_shape = 10
            self.obs_shape = 8
            self.epsilon = 0.5
            self.min_epsilon = 0.01
            self.anneal_epsilon = (self.epsilon - self.min_epsilon) / 100 # Dummy anneal rate
            self.cuda = False
            self.log_dir = "./test_logs_rollout/"
            self.alg = "test_alg"
            self.anneal_steps = 100 # for print statement

    test_args = TestArgs()

    mock_env = MockEnv(test_args)
    mock_agents = MockAgents(test_args)

    worker = RolloutWorker(mock_env, mock_agents, test_args)

    print("\n--- Generating Training Episode ---")
    episode_data_train, stats_train, _ = worker.generate_episode(episode_num=1, evaluate=False)
    print(f"Training Episode Stats: {stats_train}")
    assert stats_train['episode_length'] <= test_args.episode_limit
    assert 'o' in episode_data_train
    assert episode_data_train['o'].shape == (test_args.episode_limit, test_args.n_agents, test_args.obs_shape)
    
    # Check epsilon annealing (conceptual, worker manages its copy)
    # initial_worker_eps = test_args.epsilon
    # assert worker.epsilon < initial_worker_eps or worker.epsilon == test_args.min_epsilon

    print("\n--- Generating Evaluation Episode (with logging) ---")
    # Ensure log directory exists or can be created
    if os.path.exists(test_args.log_dir):
        import shutil
        shutil.rmtree(test_args.log_dir) # Clean up previous test logs

    episode_data_eval, stats_eval, log_file_path = worker.generate_episode(episode_num=1, evaluate=True)
    print(f"Evaluation Episode Stats: {stats_eval}")
    assert stats_eval['epsilon'] == 0 # Epsilon should be 0 for evaluation
    assert os.path.exists(test_args.log_dir), "Log directory was not created."
    # Check if a log file was created (name is dynamic with timestamp)
    log_files = os.listdir(test_args.log_dir)
    assert len(log_files) > 0, "Log file was not created in the log directory."
    print(f"Log file created: {os.path.join(test_args.log_dir, log_files[0])}")


    print("\nRolloutWorker tests completed.")
    # Clean up test log directory
    if os.path.exists(test_args.log_dir):
        import shutil
        shutil.rmtree(test_args.log_dir) 