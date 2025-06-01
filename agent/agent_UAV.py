import numpy as np
import torch

class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        if args.alg == 'vdn':
            from policy.vdn_UAV import VDN
            self.policy = VDN(args)
        # Add other policies like QMix, QTRAN etc. here if needed
        # elif args.alg == 'qmix':
        #     from policy.qmix import QMix
        #     self.policy = QMix(args)
        else:
            raise Exception("Unsupported algorithm: {}".format(args.alg))

        self.epsilon = args.epsilon if hasattr(args, 'epsilon') else 0.05 # Default epsilon if not in args
        self.min_epsilon = args.min_epsilon if hasattr(args, 'min_epsilon') else 0.01
        self.anneal_epsilon = args.anneal_epsilon if hasattr(args, 'anneal_epsilon') else (self.epsilon - self.min_epsilon) / 100000 # Default anneal rate
        print(f"Initialized Agents with Epsilon: {self.epsilon}, Min Epsilon: {self.min_epsilon}, Anneal Steps: {(self.epsilon - self.min_epsilon) / self.anneal_epsilon if self.anneal_epsilon > 0 else 'N/A'}")


    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        """
        Chooses an action for a single agent.
        Args:
            obs (np.ndarray): Local observation for the agent.
            last_action (np.ndarray): Last action taken by the agent (one-hot). Not directly used by VDN Q-network but can be part of obs.
            agent_num (int): The agent's id.
            avail_actions (np.ndarray): Boolean array indicating available actions. Assumed all actions are available if None.
            epsilon (float): Current exploration rate, passed from Runner/RolloutWorker.
            evaluate (bool): Flag to set deterministic action selection (no exploration).
        Returns:
            int: The chosen action.
        """
        inputs = obs.copy() # Ensure obs is not modified

        if not evaluate and np.random.uniform() < epsilon:
            if avail_actions is not None:
                avail_actions_ind = np.nonzero(avail_actions)[0]
                if len(avail_actions_ind) == 0: # Should not happen if env guarantees one available action
                    action = np.random.choice(self.n_actions) # Fallback
                else:
                    action = np.random.choice(avail_actions_ind)
            else:
                action = np.random.choice(self.n_actions)
        else:
            # Greedy action
            # Policy's method is responsible for converting obs to tensor and moving to device
            q_values_tensor = self.policy.get_agent_q_values(inputs, agent_num) 
            
            # Process Q-values (e.g., convert to numpy, handle unavailable actions)
            q_values_np = q_values_tensor.squeeze(0).cpu().detach().numpy()

            if avail_actions is not None:
                q_values_np[avail_actions == 0] = -float('inf')  # Mask unavailable actions
            
            action = np.argmax(q_values_np)
        
        # The epsilon annealing is handled by the Runner, not here.
        # self.epsilon = max(self.min_epsilon, self.epsilon - self.anneal_epsilon) # DO NOT UNCOMMENT

        return action

    def train(self, batch, train_step, epsilon=None):
        """
        Trains the policy.
        Args:
            batch: A batch of experiences from the replay buffer.
            train_step (int): The current training step (for logging or annealing).
            epsilon (float, optional): Current exploration rate for Q-learning updates (e.g. for target policy smoothing).
                                       Not typically passed to VDN.learn directly, but VDN might use its own.
        Returns:
            float: The loss from the training step.
        """
        # The train method of Agents class calls the learn method of the VDN policy.
        # The VDN policy's learn method handles TD target calculation, loss, and backprop.
        # The batch format needs to match what policy.learn() expects.
        # `common/replay_buffer.py` stores: (s, o, u, r, s_next, o_next, terminated, padded)
        # `policy/vdn_UAV.py` `learn` method:
        #   - "From ReplayBuffer receives sampling data (by Runner)"
        #   - Calculates Q values using eval_mlp and target_mlp.
        #   - Calculates total Q using eval_vdn_net and target_vdn_net.
        #   - Calculates TD Target and loss.
        
        # The epsilon here might be for exploration if the policy uses it during training updates,
        # but typically TD learning uses greedy actions from target network.
        # The VDN.learn method itself should manage any necessary epsilon for its internal calculations if any.
        
        loss = self.policy.learn(batch, train_step)
        return loss

    def save_model(self, path, train_step):
        """Saves the policy model."""
        self.policy.save_model(path)
        print(f"Agent's policy model saved (identifier: {train_step}) to directory: {path}")

    def load_model(self, path):
        """Loads the policy model."""
        self.policy.load_model(path)
        print(f"Loaded model from {path}")

    # Utility for epsilon annealing if managed by runner
    def get_epsilon(self):
        return self.epsilon

    def set_epsilon(self, epsilon_val):
        self.epsilon = epsilon_val

# Example usage (conceptual, assumes args and policy are set up)
if __name__ == '__main__':
    # This is a placeholder for testing; real usage would be within the MARL framework (e.g., Runner)
    class DummyArgs:
        def __init__(self):
            self.n_actions = 4 # e.g., 2 speeds * 2 channels
            self.n_agents = 3
            self.state_shape = 12 # Example state size
            self.obs_shape = 10   # Example observation size
            self.alg = 'vdn'
            self.epsilon = 0.5
            self.min_epsilon = 0.01
            self.anneal_epsilon = (self.epsilon - self.min_epsilon) / 5000 # Example anneal
            self.use_cuda = False # torch.cuda.is_available()

            # Args for VDN policy (these would normally come from arguments.py)
            self.rnn_hidden_dim = 64
            self.lr = 0.0005
            self.gamma = 0.99
            self.target_update_cycle = 200
            self.grad_norm_clip = 10
            self.device = torch.device("cuda" if self.use_cuda else "cpu")


    args = DummyArgs()
    
    # Mock the VDN policy structure for standalone testing of Agents class choose_action
    # In a real run, this is imported from policy.vdn_UAV
    class MockVDNPolicy:
        def __init__(self, args_policy):
            self.n_agents = args_policy.n_agents
            self.n_actions = args_policy.n_actions
            self.device = args_policy.device
            # Mock eval_mlp for each agent (simplified)
            # In real VDN, these are actual torch.nn.Module instances
            self.eval_mlps = [lambda obs_tensor: torch.rand(obs_tensor.size(0), self.n_actions).to(self.device) 
                              for _ in range(self.n_agents)] 

        def get_agent_q_values(self, obs_tensor, agent_num):
            # This is a simplified mock. Real VDN would pass obs through actual network.
            # obs_tensor is expected to be [1, obs_shape]
            if agent_num < len(self.eval_mlps):
                # return self.eval_mlps[agent_num](obs_tensor)
                # For mock, let's just return random Q-values based on the number of actions
                return torch.rand(obs_tensor.shape[0], self.n_actions).to(self.device)
            else:
                raise ValueError(f"Agent number {agent_num} out of range for policy.")

        def learn(self, batch, train_step):
            print(f"MockVDNPolicy: learn called at train_step {train_step} with batch of size {len(batch['o'])}")
            # Simulate some loss calculation
            return np.random.rand() 
            
        def save_model(self, path, train_step):
            print(f"MockVDNPolicy: save_model called for path {path}, step {train_step}")

        def load_model(self, path):
            print(f"MockVDNPolicy: load_model called for path {path}")


    # Replace the actual policy import with the mock for this test script
    import sys
    # This is a bit of a hack for standalone testing. 
    # It simulates that 'policy.vdn_UAV' module has a 'VDN' class which is our MockVDNPolicy.
    mock_policy_module = type(sys)('policy.vdn_UAV')
    mock_policy_module.VDN = MockVDNPolicy
    sys.modules['policy.vdn_UAV'] = mock_policy_module


    print("--- Testing Agents Class Initialization ---")
    agents_manager = Agents(args)
    print(f"Agents manager initialized with policy: {type(agents_manager.policy)}")

    print("\n--- Testing choose_action ---")
    # Dummy observation for one agent
    dummy_obs_shape = (args.obs_shape,) if isinstance(args.obs_shape, int) else args.obs_shape
    obs_agent_0 = np.random.rand(*dummy_obs_shape) 
    last_action_agent_0 = np.zeros(args.n_actions) # Example last action (one-hot)
    agent_id_0 = 0
    avail_actions_agent_0 = np.array([1, 1, 0, 1]) # Example: action 2 is unavailable

    # Test with exploration
    chosen_action_explore = agents_manager.choose_action(obs_agent_0, last_action_agent_0, agent_id_0, avail_actions_agent_0, epsilon=1.0, evaluate=False)
    print(f"Chosen action (explore, epsilon=1.0, agent 0): {chosen_action_explore}")
    assert avail_actions_agent_0[chosen_action_explore] == 1, "Exploratory action chose an unavailable action"

    # Test with greedy (epsilon=0)
    chosen_action_greedy = agents_manager.choose_action(obs_agent_0, last_action_agent_0, agent_id_0, avail_actions_agent_0, epsilon=0.0, evaluate=False)
    print(f"Chosen action (greedy, epsilon=0.0, agent 0): {chosen_action_greedy}")
    assert avail_actions_agent_0[chosen_action_greedy] == 1, "Greedy action chose an unavailable action"

    # Test with evaluate=True (should be greedy)
    chosen_action_eval = agents_manager.choose_action(obs_agent_0, last_action_agent_0, agent_id_0, avail_actions_agent_0, epsilon=1.0, evaluate=True) # epsilon shouldn't matter
    print(f"Chosen action (evaluate=True, agent 0): {chosen_action_eval}")
    assert avail_actions_agent_0[chosen_action_eval] == 1, "Evaluation action chose an unavailable action"
    assert chosen_action_eval == chosen_action_greedy, "Greedy and Evaluate actions should match for same obs if Q-values are deterministic"


    print("\n--- Testing Epsilon Annealing (conceptual) ---")
    initial_eps = agents_manager.get_epsilon()
    for _ in range(5): # Simulate a few steps
        agents_manager.choose_action(obs_agent_0, last_action_agent_0, agent_id_0, avail_actions_agent_0, agents_manager.get_epsilon(), evaluate=False)
    print(f"Epsilon after some steps: {agents_manager.get_epsilon()} (initial: {initial_eps})")
    assert agents_manager.get_epsilon() < initial_eps or agents_manager.get_epsilon() == args.min_epsilon

    print("\n--- Testing train method ---")
    # Create a dummy batch (simplified structure, real batch is more complex)
    batch_size = 2
    dummy_batch = {
        'o': np.random.rand(batch_size, args.n_agents, args.episode_limit if hasattr(args, 'episode_limit') else 50, args.obs_shape), # o: (batch_size, n_agents, episode_len, obs_shape)
        # Add other necessary keys for the VDN.learn method: u, r, s, o_next, s_next, terminated, padded etc.
        # This is just a placeholder to call the method.
    }
    # A more realistic batch structure from replay_buffer.py:
    # keys: 'o', 's', 'u', 'r', 'o_next', 's_next', 'avail_u', 'avail_u_next', 'u_onehot', 'padded', 'terminated'
    # shapes example: o: (batch_size, episode_limit, n_agents, obs_shape)
    # For this mock, the content doesn't matter much, just that `learn` is called.
    
    # For the mock to work, let's pass a simpler batch that `len(batch['o'])` can understand
    simple_dummy_batch = {'o': [None]*batch_size} # List of episodes, each episode is a dict... this is also not quite right.
                                             # The replay buffer usually samples (B, T, N, ...) where B is batch, T is num_transitions
    
    # The replay buffer likely returns a dictionary of tensors/arrays like:
    # batch['o'] shape (BATCH_SIZE, MAX_EPISODE_LEN, N_AGENTS, OBS_DIM)
    # batch['u'] shape (BATCH_SIZE, MAX_EPISODE_LEN, N_AGENTS, 1)
    # etc.
    # The VDN policy's learn method will expect this.
    # For the mock test, let's just pass something that `learn` can be called with.
    mock_episode_len = 5 # Short episode for testing batch
    mock_batch_for_train = {
        'o': np.random.rand(batch_size, mock_episode_len, args.n_agents, args.obs_shape),
        # ... other necessary parts of the batch for policy.learn()
    }

    loss = agents_manager.train(mock_batch_for_train, train_step=100)
    print(f"Loss from dummy train call: {loss}")

    print("\n--- Testing Save/Load Model ---")
    agents_manager.save_model("./dummy_agent_model", 100)
    agents_manager.load_model("./dummy_agent_model")

    # Clean up mock module
    del sys.modules['policy.vdn_UAV']
    print("\nAgent class tests completed (using mock policy).") 