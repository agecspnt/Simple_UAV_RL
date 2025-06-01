import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np # Added numpy import
from torch.cuda.amp import GradScaler, autocast

from network.base_net import D3QN # Or MLP, RNN depending on choice
from network.vdn_net import VDNNet

class VDN:
    def __init__(self, args):
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape_agent = self.obs_shape

        # Initialize networks
        # Individual Q-networks for each agent
        # According to DETAILED_DESIGN.md, eval_mlp is an instance of D3QN or MLP
        # Using D3QN as specified in base_net.py
        self.eval_mlp = D3QN(input_shape_agent, args) 
        self.target_mlp = D3QN(input_shape_agent, args)

        # VDN mixing network
        self.eval_vdn_net = VDNNet()
        self.target_vdn_net = VDNNet()

        self.eval_mlp.to(self.args.device)
        self.target_mlp.to(self.args.device)
        self.eval_vdn_net.to(self.args.device)
        self.target_vdn_net.to(self.args.device)

        # Load target network parameters from evaluation network
        self.target_mlp.load_state_dict(self.eval_mlp.state_dict())
        # VDNNet typically has no parameters, but if it did, we'd copy them too.
        # self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())


        # Optimizer
        # Parameters to be optimized are those of the individual Q-networks
        # VDNNet has no parameters by default. If it did, they might also be included.
        self.optimizer = optim.Adam(self.eval_mlp.parameters(), lr=self.args.lr)

        # Initialize GradScaler for mixed precision training if CUDA is available
        self.scaler = GradScaler(enabled=self.args.cuda)

        # self.model_base_dir = os.path.join(args.model_dir, args.alg, args.map) # Base directory for this alg/map
        # Specific model path with train_step will be passed to save/load methods


    def learn(self, batch, train_step, epsilon=None): # train_step is used for target net update. max_episode_len removed.
        """
        Core learning step for VDN.
        Args:
            batch: A batch of experiences.
                   Contains: 'o', 's', 'u', 'r', 'o_next', 's_next', 'terminated', 'padded'.
            train_step (int): Current training step.
            epsilon (float, optional): Epsilon for exploration, not directly used in Q-value calculation
                                       but could be relevant for action selection strategy if integrated here.
        Returns:
            loss (float): The mean TD error for the batch.
        """
        episode_num = batch['o'].shape[0]
        max_episode_len = self.args.episode_limit # Use from args
        self.init_hidden(episode_num)
        
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


        s = s.to(self.args.device)
        r = r.to(self.args.device)
        u = u.to(self.args.device)
        terminated = terminated.to(self.args.device)
        obs = obs.to(self.args.device)
        obs_next = obs_next.to(self.args.device)
        mask = mask.to(self.args.device)

        # 1. Calculate Q-values for current observations and actions: Q_i(o_i, a_i)
        # These are the Q-values for the actions *taken* by the agents.
        
        # Use autocast context manager for forward propagation
        with autocast(enabled=self.args.cuda):
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
            # Q\'_tot(s\', argmax_a\' Q\'_i(o\'_i, a\'_i\'))
            # For VDN, each agent i chooses a_i\' to maximize its own Q\'_i(o\'_i, a_i\').
            # These individually maximized Q-values are then summed by the target mixer.
            
            q_target_next_individual_max_values = []
            for i in range(self.n_agents):
                agent_obs_next = obs_next[:, :, i, :] # Shape: (episode_num, max_episode_len-1, obs_shape)
                # Reshape for network: (episode_num * (max_episode_len-1), obs_shape)
                agent_obs_next_reshaped = agent_obs_next.reshape(-1, self.obs_shape)
                
                # Get Q-values for ALL actions for agent i from the TARGET network
                # Shape: (episode_num * (max_episode_len-1), n_actions)
                # Target network computations are often kept in FP32 for stability,
                # but can also be done in autocast if needed. For now, let's keep it consistent.
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
            # y = r + gamma * Q\'_tot_target(s\', a\'_max) * (1 - terminated)
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
        # End autocast context

        # 5. Optimize
        self.optimizer.zero_grad()
        # Use scaler for backward propagation
        self.scaler.scale(loss).backward()
        
        # Unscales the gradients of optimizer's assigned params in-place
        self.scaler.unscale_(self.optimizer) # Needed before clip_grad_norm_
        
        if hasattr(self.args, 'grad_norm_clip') and self.args.grad_norm_clip is not None:
             torch.nn.utils.clip_grad_norm_(self.eval_mlp.parameters(), self.args.grad_norm_clip)
        
        # Use scaler to update optimizer
        self.scaler.step(self.optimizer)
        # Update scaler
        self.scaler.update()

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


    def save_model(self, path): # path is the directory like model_dir/alg/map/identifier/
        """Saves the policy model to the given path (directory)."""
        if not os.path.exists(path):
            os.makedirs(path)
        
        q_net_path = os.path.join(path, "q_network.pth")
        torch.save(self.eval_mlp.state_dict(), q_net_path)
        # If VDNNet had params: torch.save(self.eval_vdn_net.state_dict(), os.path.join(path, "vdn_mixer.pth"))
        print(f"Model Q-network saved to: {q_net_path}")

    def load_model(self, path): # path is the directory like model_dir/alg/map/identifier/
        """
        Loads the policy model from the given path (directory).
        """
        q_net_path = os.path.join(path, "q_network.pth")
        try:
            self.eval_mlp.load_state_dict(torch.load(q_net_path, map_location=lambda storage, loc: storage))
            self.target_mlp.load_state_dict(self.eval_mlp.state_dict()) 
            print(f"Q-Network loaded from {q_net_path}")
        except FileNotFoundError:
            print(f"Error: Q-Network model file not found at {q_net_path}")
            raise # Re-raise the exception so the caller knows loading failed
        except Exception as e:
            print(f"Error loading Q-Network model from {q_net_path}: {e}")
            raise # Re-raise
        
        # If VDNNet had params and was saved as "vdn_mixer.pth":
        # vdn_mixer_path = os.path.join(path, "vdn_mixer.pth")
        # if os.path.exists(vdn_mixer_path):
        #     try:
        #         self.eval_vdn_net.load_state_dict(torch.load(vdn_mixer_path, map_location=lambda storage, loc: storage))
        #         self.target_vdn_net.load_state_dict(self.eval_vdn_net.state_dict())
        #         print(f"VDN Mixer loaded from {vdn_mixer_path}")
        #     except Exception as e:
        #         print(f"Error loading VDN Mixer model from {vdn_mixer_path}: {e}")
        # else:
        #     print(f"Warning: VDN Mixer model file not found at {vdn_mixer_path} (might be expected).")

    def get_q_values_for_actions(self, obs_batch, actions_batch):
        # This method seems to be for getting Q values for specific actions, might be used internally or for analysis.
        # obs_batch shape: (batch_size, n_agents, obs_shape) or (batch_size, obs_shape) if for single agent.
        # actions_batch shape: (batch_size, n_agents, 1) or (batch_size, 1)
        
        # Assuming obs_batch is (num_episodes * episode_len, n_agents, obs_shape) for all agents
        # and actions_batch is (num_episodes * episode_len, n_agents, 1)
        # Or if it is called per agent: (num_episodes * episode_len, obs_shape)
        # and actions_batch is (num_episodes * episode_len, 1)
        
        # For now, let's assume it is called per agent as self.eval_mlp is for one agent obs
        # obs_batch: (total_timesteps, obs_shape)
        # actions_batch: (total_timesteps, 1)

        if not isinstance(obs_batch, torch.Tensor):
            obs_batch = torch.tensor(obs_batch, dtype=torch.float32, device=self.args.device if self.args.cuda else "cpu")
        if not isinstance(actions_batch, torch.Tensor):
            actions_batch = torch.tensor(actions_batch, dtype=torch.long, device=self.args.device if self.args.cuda else "cpu")

        q_all_actions = self.eval_mlp(obs_batch) # (total_timesteps, n_actions)
        q_taken = torch.gather(q_all_actions, dim=1, index=actions_batch) # (total_timesteps, 1)
        return q_taken

    def get_agent_q_values(self, agent_obs_np, agent_num): # agent_num might not be needed if eval_mlp is shared and stateless regarding agent_id
        """
        Gets Q-values for all actions for a single agent's observation.
        Args:
            agent_obs_np (np.ndarray): Observation for a single agent. Shape (obs_shape,)
            agent_num (int): Agent ID (may not be used by a shared MLP/D3QN directly unless part of input).
        Returns:
            torch.Tensor: Q-values for all actions. Shape (1, n_actions).
        """
        # Convert numpy obs to tensor and add batch dimension
        agent_obs_tensor = torch.tensor(agent_obs_np, dtype=torch.float32).unsqueeze(0)
        agent_obs_tensor = agent_obs_tensor.to(self.args.device)
        
        # self.eval_mlp is shared. It takes obs of one agent.
        q_values = self.eval_mlp(agent_obs_tensor) # Shape: (1, n_actions)
        return q_values

# Further considerations:
# - If D3QN is an RNN, init_hidden needs proper implementation and usage in learn and get_agent_q_values.
# - Device handling (cuda/cpu) should be consistent. 