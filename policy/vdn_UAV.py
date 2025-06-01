import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np # Added numpy import

from network.base_net import D3QN # Or MLP, RNN depending on choice
from network.vdn_net import VDNNet

class VDN:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape_agent = self.obs_shape # Individual agent observation shape

        # Initialize networks
        # Individual Q-networks for each agent
        # According to DETAILED_DESIGN.md, eval_mlp is an instance of D3QN or MLP
        # Using D3QN as specified in base_net.py
        self.eval_mlp = D3QN(input_shape_agent, args) 
        self.target_mlp = D3QN(input_shape_agent, args)

        # VDN mixing network
        self.eval_vdn_net = VDNNet()
        self.target_vdn_net = VDNNet()

        if self.args.cuda:
            self.eval_mlp.cuda()
            self.target_mlp.cuda()
            self.eval_vdn_net.cuda()
            self.target_vdn_net.cuda()

        # Load target network parameters from evaluation network
        self.target_mlp.load_state_dict(self.eval_mlp.state_dict())
        # VDNNet typically has no parameters, but if it did, we'd copy them too.
        # self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())


        # Optimizer
        # Parameters to be optimized are those of the individual Q-networks
        # VDNNet has no parameters by default. If it did, they might also be included.
        self.optimizer = optim.Adam(self.eval_mlp.parameters(), lr=self.args.lr)

        # For saving/loading models
        # Ensure model_dir, alg, and map are present in args
        self.model_dir = os.path.join(args.model_dir, args.alg, args.map)


    def learn(self, batch, max_episode_len, train_step, epsilon=None): # train_step is used for target net update
        """
        Core learning step for VDN.
        Args:
            batch: A batch of experiences.
                   Contains: 'o', 's', 'u', 'r', 'o_next', 's_next', 'terminated', 'padded'.
            max_episode_len (int): Maximum length of an episode.
            train_step (int): Current training step.
            epsilon (float, optional): Epsilon for exploration, not directly used in Q-value calculation
                                       but could be relevant for action selection strategy if integrated here.
        Returns:
            loss (float): The mean TD error for the batch.
        """
        episode_num = batch['o'].shape[0] # Number of episodes in batch
        self.init_hidden(episode_num) # For RNNs, not D3QN/MLP. Keep for compatibility.
        
        # Convert batch data to tensors
        for key in batch.keys():
            if key == 'u': # Actions (u) are long type
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        
        # Extract necessary components from the batch
        # Dimensions:
        # s, s_next: (episode_num, max_episode_len, state_shape)
        # u: (episode_num, max_episode_len, n_agents, 1)  (action index for each agent)
        # r: (episode_num, max_episode_len, 1) (global reward)
        # terminated: (episode_num, max_episode_len, 1)
        # o, o_next: (episode_num, max_episode_len, n_agents, obs_shape)
        # padded: (episode_num, max_episode_len, 1)

        # We need to select data for valid transitions, typically up to max_episode_len - 1
        s = batch['s'][:, :-1]
        r = batch['r'][:, :-1]
        u = batch['u'][:, :-1]
        terminated = batch['terminated'][:, :-1].float()
        obs = batch['o'][:, :-1]
        obs_next = batch['o'][:, 1:]
        mask = (1 - batch["padded"][:, :-1]).float() # Mask for valid transitions


        if self.args.cuda:
            s = s.cuda()
            r = r.cuda()
            u = u.cuda()
            terminated = terminated.cuda()
            obs = obs.cuda()
            obs_next = obs_next.cuda()
            mask = mask.cuda()

        # 1. Calculate Q-values for current observations and actions: Q_i(o_i, a_i)
        # These are the Q-values for the actions *taken* by the agents.
        q_evals_agents = []
        for i in range(self.n_agents):
            agent_obs = obs[:, :, i, :] # Shape: (episode_num, max_episode_len-1, obs_shape)
            # Reshape for network: (episode_num * (max_episode_len-1), obs_shape)
            agent_obs_reshaped = agent_obs.reshape(-1, self.obs_shape) 
            
            # Get Q-values for ALL actions for this agent
            # Shape: (episode_num * (max_episode_len-1), n_actions)
            q_values_all_actions_agent = self.eval_mlp(agent_obs_reshaped)
            
            # Reshape back: (episode_num, max_episode_len-1, n_actions)
            q_values_all_actions_agent = q_values_all_actions_agent.reshape(episode_num, max_episode_len -1, self.n_actions)
            
            # Gather Q-values for the specific actions taken by agent i
            # u[:, :, i] shape: (episode_num, max_episode_len-1, 1)
            action_taken_by_agent_i = u[:, :, i]
            # Shape: (episode_num, max_episode_len-1, 1)
            q_taken_for_agent = torch.gather(q_values_all_actions_agent, dim=2, index=action_taken_by_agent_i)
            q_evals_agents.append(q_taken_for_agent)

        # Concatenate Q-values from all agents along the agent dimension
        # Each element in q_evals_agents is (episode_num, max_episode_len-1, 1)
        # q_evals becomes (episode_num, max_episode_len-1, n_agents)
        q_evals = torch.cat(q_evals_agents, dim=2)

        # Pass these agent-wise Q-values (for chosen actions) to the VDN mixer
        # q_total_eval becomes (episode_num, max_episode_len-1, 1)
        q_total_eval = self.eval_vdn_net(q_evals)


        # 2. Calculate target Q-values for the next state
        # Q'_tot(s', argmax_a' Q'_i(o'_i, a'_i))
        # For VDN, each agent i chooses a_i' to maximize its own Q'_i(o'_i, a_i').
        # These individually maximized Q-values are then summed by the target mixer.
        
        q_target_next_individual_max_values = []
        for i in range(self.n_agents):
            agent_obs_next = obs_next[:, :, i, :] # Shape: (episode_num, max_episode_len-1, obs_shape)
            # Reshape for network: (episode_num * (max_episode_len-1), obs_shape)
            agent_obs_next_reshaped = agent_obs_next.reshape(-1, self.obs_shape)
            
            # Get Q-values for ALL actions for agent i from the TARGET network
            # Shape: (episode_num * (max_episode_len-1), n_actions)
            q_next_agent_target_all_actions = self.target_mlp(agent_obs_next_reshaped)
            
            # Reshape back: (episode_num, max_episode_len-1, n_actions)
            q_next_agent_target_all_actions = q_next_agent_target_all_actions.reshape(episode_num, max_episode_len - 1, self.n_actions)
            
            # Select the max Q-value over actions for this agent
            # Shape: (episode_num, max_episode_len-1, 1)
            q_next_agent_target_max, _ = torch.max(q_next_agent_target_all_actions, dim=2, keepdim=True)
            q_target_next_individual_max_values.append(q_next_agent_target_max)

        # Concatenate the max Q-values from all agents
        # q_target_next_max_per_agent becomes (episode_num, max_episode_len-1, n_agents)
        q_target_next_max_per_agent = torch.cat(q_target_next_individual_max_values, dim=2)

        # Pass these max Q-values to the TARGET VDN mixer
        # q_total_target_next becomes (episode_num, max_episode_len-1, 1)
        q_total_target_next = self.target_vdn_net(q_target_next_max_per_agent)
        
        # 3. Calculate TD Target (y)
        # y = r + gamma * Q'_tot_target(s', a'_max) * (1 - terminated)
        # r shape: (episode_num, max_episode_len-1, 1)
        # q_total_target_next shape: (episode_num, max_episode_len-1, 1)
        # terminated shape: (episode_num, max_episode_len-1, 1)
        targets = r + self.args.gamma * q_total_target_next * (1 - terminated)

        # 4. Calculate Loss
        # Loss = (y - Q_tot_eval(s,a))^2
        # q_total_eval shape: (episode_num, max_episode_len-1, 1)
        # targets shape: (episode_num, max_episode_len-1, 1)
        td_error = (q_total_eval - targets.detach()) # Detach targets to prevent gradients flowing into target nets
        
        # Apply mask for padded steps. Mask has shape (episode_num, max_episode_len-1, 1)
        masked_td_error = td_error * mask
        
        # Mean Squared Error loss
        # Sum of squared errors over all non-padded steps, divided by the number of non-padded steps.
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # 5. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        if hasattr(self.args, 'grad_norm_clip') and self.args.grad_norm_clip is not None:
             torch.nn.utils.clip_grad_norm_(self.eval_mlp.parameters(), self.args.grad_norm_clip)
        self.optimizer.step()

        # 6. Update target network
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_mlp.load_state_dict(self.eval_mlp.state_dict())
            # self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict()) # Only if VDNNet had params

        return loss.item()

    def init_hidden(self, batch_size):
        # For D3QN/MLP, this is a no-op.
        # Kept for API consistency if individual agent networks were RNNs.
        # If self.eval_mlp were an RNN:
        # self.hidden_eval = self.eval_mlp.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1) # (bs, n_agents, rnn_hidden_dim)
        # self.hidden_target = self.target_mlp.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        pass


    def save_model(self, train_step, model_name_suffix=""):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        model_filename = f"{train_step}"
        if model_name_suffix:
            model_filename += f"_{model_name_suffix}"
        
        torch.save(self.eval_mlp.state_dict(), os.path.join(self.model_dir, f"{model_filename}_q_network.pth"))
        # VDNNet usually has no parameters. If it did, save it too.
        # torch.save(self.eval_vdn_net.state_dict(), os.path.join(self.model_dir, f"{model_filename}_vdn_mixer.pth"))
        print(f"Model saved: {os.path.join(self.model_dir, f'{model_filename}_q_network.pth')}")

    def load_model(self, model_path_q_net, model_path_vdn_mixer=None):
        """
        Loads the Q-network and (optionally) VDN mixer parameters.
        Args:
            model_path_q_net (str): Path to the saved Q-network state_dict.
            model_path_vdn_mixer (str, optional): Path to the saved VDN mixer state_dict.
        """
        try:
            self.eval_mlp.load_state_dict(torch.load(model_path_q_net, map_location=lambda storage, loc: storage))
            self.target_mlp.load_state_dict(self.eval_mlp.state_dict()) # Keep target consistent
            print(f"Q-Network loaded from {model_path_q_net}")
        except FileNotFoundError:
            print(f"Error: Q-Network model file not found at {model_path_q_net}")
            return
        except Exception as e:
            print(f"Error loading Q-Network model: {e}")
            return
        
        # VDNNet typically has no learnable parameters.
        # If it did, loading would look like this:
        if model_path_vdn_mixer:
            try:
                # self.eval_vdn_net.load_state_dict(torch.load(model_path_vdn_mixer, map_location=lambda storage, loc: storage))
                # self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())
                # print(f"VDN Mixer loaded from {model_path_vdn_mixer}")
                pass # Placeholder since VDNNet has no params by default
            except FileNotFoundError:
                print(f"Warning: VDN mixer model file {model_path_vdn_mixer} not found (this might be expected).")
            except Exception as e:
                print(f"Error loading VDN Mixer model: {e}")


    def get_q_values_for_actions(self, obs_batch, actions_batch):
        """
        Calculates Q_tot for given observations and actions.
        This is Q_tot( (o_1,...,o_N), (a_1,...,a_N) ) = sum_i Q_i(o_i, a_i)
        Args:
            obs_batch (np.ndarray or torch.Tensor): Batch of observations for all agents.
                                                 Shape: (batch_size, n_agents, obs_shape)
            actions_batch (np.ndarray or torch.Tensor): Batch of actions taken by all agents.
                                                     Shape: (batch_size, n_agents) or (batch_size, n_agents, 1)
        Returns:
            q_total (torch.Tensor): Total Q-value for the joint action. Shape: (batch_size, 1)
        """
        if isinstance(obs_batch, np.ndarray):
            obs_batch = torch.tensor(obs_batch, dtype=torch.float32)
        if isinstance(actions_batch, np.ndarray):
            # Actions should be long for gather
            actions_batch = torch.tensor(actions_batch, dtype=torch.long)

        if actions_batch.ndim == 2: # if (batch_size, n_agents)
            actions_batch = actions_batch.unsqueeze(-1) # to (batch_size, n_agents, 1)

        current_device = next(self.eval_mlp.parameters()).device
        obs_batch = obs_batch.to(current_device)
        actions_batch = actions_batch.to(current_device)
        
        batch_size = obs_batch.shape[0]
        q_selected_actions_agents = []

        for i in range(self.n_agents):
            agent_obs = obs_batch[:, i, :] # Shape: (batch_size, obs_shape)
            
            # Get Q-values for ALL actions for agent i
            # Shape: (batch_size, n_actions)
            q_all_actions_agent_i = self.eval_mlp(agent_obs)
            
            # Gather Q-values for the specific action taken by agent i
            # actions_batch[:, i] shape: (batch_size, 1)
            action_taken_by_agent_i = actions_batch[:, i] # Shape: (batch_size, 1)
            # Shape: (batch_size, 1)
            q_selected_action_agent_i = torch.gather(q_all_actions_agent_i, dim=1, index=action_taken_by_agent_i)
            
            q_selected_actions_agents.append(q_selected_action_agent_i) # List of (batch_size, 1)

        # Concatenate Q-values from all agents.
        # Each element in q_selected_actions_agents is (batch_size, 1).
        # We want to form a tensor (batch_size, n_agents) to pass to VDNNet.
        # Concatenating along dim=1 will give (batch_size, n_agents).
        q_evals_for_mixer = torch.cat(q_selected_actions_agents, dim=1) # Shape: (batch_size, n_agents)

        q_total = self.eval_vdn_net(q_evals_for_mixer) # Shape: (batch_size, 1)
        return q_total
        
    def get_agent_q_values(self, agent_obs_batch):
        """
        Get Q-values for all actions for a batch of a single agent's observations.
        Args:
            agent_obs_batch (np.ndarray or torch.Tensor): Batch of observations for a single agent.
                                                       Shape: (batch_size, obs_shape) or (obs_shape) for single obs.
        Returns:
            q_values (torch.Tensor): Q-values for all actions. Shape: (batch_size, n_actions)
        """
        if not isinstance(agent_obs_batch, torch.Tensor):
            agent_obs_batch = torch.tensor(agent_obs_batch, dtype=torch.float32)
        
        if agent_obs_batch.ndim == 1: # Single observation (obs_shape)
            agent_obs_batch = agent_obs_batch.unsqueeze(0) # Add batch dimension -> (1, obs_shape)

        current_device = next(self.eval_mlp.parameters()).device
        agent_obs_batch = agent_obs_batch.to(current_device)
            
        q_values = self.eval_mlp(agent_obs_batch) # Shape: (batch_size, n_actions)
        return q_values 