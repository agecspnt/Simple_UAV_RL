import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_units=[64, 64]):
        """
        Simple Multi-Layer Perceptron (MLP)
        Args:
            input_shape (int): Dimension of the input.
            n_actions (int): Dimension of the output (number of actions).
            hidden_units (list of int): List of hidden layer sizes.
        """
        super(MLP, self).__init__()
        layers = []
        prev_units = input_shape
        for units in hidden_units:
            layers.append(nn.Linear(prev_units, units))
            layers.append(nn.ReLU())
            prev_units = units
        layers.append(nn.Linear(prev_units, n_actions))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class RNN(nn.Module):
    def __init__(self, input_shape, args):
        """
        Gated Recurrent Unit (GRU) Cell based RNN.
        Args:
            input_shape (int): Dimension of the input to the GRU cell.
            args: Contains rnn_hidden_dim.
        """
        super(RNN, self).__init__()
        self.args = args
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, args.n_actions) # Output Q-values for each action

    def init_hidden(self):
        # Make hidden states on same device as model parameters
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, obs, hidden_state):
        """
        Forward pass for the RNN.
        Args:
            obs (torch.Tensor): Batch of observations. Shape: (batch_size, input_shape)
            hidden_state (torch.Tensor): Batch of hidden states. Shape: (batch_size, rnn_hidden_dim)
        Returns:
            q_values (torch.Tensor): Q-values for each action. Shape: (batch_size, n_actions)
            h_out (torch.Tensor): New hidden states. Shape: (batch_size, rnn_hidden_dim)
        """
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        q_values = self.fc2(h_out)
        return q_values, h_out

class D3QN(nn.Module):
    def __init__(self, input_shape, args):
        """
        Dueling Double Deep Q-Network (D3QN).
        Args:
            input_shape (int): Dimension of the input.
            args: Contains n_actions and potentially other MLP config.
        """
        super(D3QN, self).__init__()
        self.args = args
        
        # Feature layer (can be more complex, e.g. CNN for image input)
        # For vector input, a simple MLP layer
        self.feature_layer_size = args.rnn_hidden_dim if hasattr(args, 'rnn_hidden_dim') else 64 
        # Using rnn_hidden_dim as a common size for MLP layers in this project context
        # Or a fixed size like 64 if rnn_hidden_dim is not specified (e.g. for non-RNN based VDN)

        self.feature_net = MLP(input_shape, self.feature_layer_size, 
                               hidden_units=[self.feature_layer_size]) # One hidden layer for feature extraction

        # Advantage stream
        self.advantage_net = MLP(self.feature_layer_size, args.n_actions, 
                                 hidden_units=[self.feature_layer_size // 2]) # Smaller MLP for advantage

        # Value stream
        self.value_net = MLP(self.feature_layer_size, 1, 
                             hidden_units=[self.feature_layer_size // 2]) # Smaller MLP for value

    def forward(self, x):
        """
        Forward pass for D3QN.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            q_values (torch.Tensor): Q-values.
        """
        features = self.feature_net(x)
        
        advantage = self.advantage_net(features)
        value = self.value_net(features)
        
        # Combine value and advantage streams
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values 