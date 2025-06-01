import torch
import torch.nn as nn

class VDNNet(nn.Module):
    def __init__(self):
        """
        Value Decomposition Network (VDN) mixer.
        It simply sums the Q-values from individual agents.
        This network typically has no learnable parameters.
        """
        super(VDNNet, self).__init__()

    def forward(self, agent_qs):
        """
        Forward pass for the VDN mixer.
        Args:
            agent_qs (torch.Tensor): Tensor containing Q-values for each agent for the selected actions.
                                     Expected shape: (batch_size, n_agents) or (batch_size, episode_len, n_agents)
                                     if processing sequences.
                                     The VDN policy usually passes Q_i(s_i, a_i) for each agent i.
        Returns:
            q_total (torch.Tensor): Summed Q-values. Shape: (batch_size, 1) or (batch_size, episode_len, 1)
        """
        # Sum across the agent dimension. 
        # If agent_qs is (batch_size, n_agents), sum along dim=1.
        # If agent_qs is (batch_size, episode_len, n_agents), sum along dim=2.
        # We expect the input to be shaped such that the last dimension is n_agents.
        q_total = torch.sum(agent_qs, dim=-1, keepdim=True)
        return q_total 