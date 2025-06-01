import numpy as np
import threading

class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit

        # Memory entries
        self.buffer = {
            'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
            's': np.empty([self.size, self.episode_limit, self.state_shape]),
            'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]), # Actions
            'r': np.empty([self.size, self.episode_limit, 1]), # Rewards
            'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
            's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
            'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
            'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
            'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
            'padded': np.empty([self.size, self.episode_limit, 1]), # Mask for padded steps
            'terminated': np.empty([self.size, self.episode_limit, 1]) # Mask for terminated states
        }
        
        # Thread lock
        self.lock = threading.Lock()
        
        # Buffer metadata
        self.current_idx = 0
        self.current_size = 0

    def store_episode(self, episode_batch):
        """
        Stores a completed episode batch into the replay buffer.
        episode_batch: A dictionary containing episode data, typically from RolloutWorker.
                       Keys should match those in self.buffer. 
                       Example: episode_batch['o'] has shape (episode_len, n_agents, obs_shape)
        """
        # batch_size = len(episode_batch['o']) # This was number of timesteps, not episodes
        # We assume episode_batch is a single episode.
        
        with self.lock:
            idx = self._get_storage_idx(1) # inc should be 1 for storing one episode
            for key in self.buffer.keys():
                # Data from episode_batch is [episode_len, num_features...]
                # Buffer expects [idx, episode_len, num_features...]
                # Ensure the episode_batch[key] has the same length as episode_limit by padding if necessary
                # This should ideally be handled by the RolloutWorker before storing.
                # For now, assume episode_batch[key] has length self.episode_limit.
                self.buffer[key][idx] = episode_batch[key]
    
    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.
        batch_size: The number of episodes to sample.
        Returns: A dictionary containing the sampled batch.
        """
        temp_buffer = {}
        with self.lock:
            # Randomly sample episode indices
            episode_idxs = np.random.randint(0, self.current_size, batch_size)
            
            for key in self.buffer.keys():
                temp_buffer[key] = self.buffer[key][episode_idxs]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        """
        Determines the next available storage index or indices.
        inc: Number of episodes to store (usually 1).
        Returns: A single index or a range of indices.
        """
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size: # Handle wrap-around if not enough space at the end
            # Not enough space at the end, fill remaining and then from start
            # This simple version just overwrites from start if current_idx is near end
            # A more robust version would handle partial fills if inc > 1 and wraps around.
            # For inc=1 (typical for store_episode), this is fine.
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else: # Buffer is full, overwrite from the beginning
            idx = np.arange(0, inc)
            self.current_idx = inc

        self.current_size = min(self.size, self.current_size + inc)
        
        if inc == 1:
            return idx[0]
        return idx

    def __len__(self):
        return self.current_size


if __name__ == '__main__':
    # Dummy args for testing ReplayBuffer
    class DummyArgs:
        def __init__(self):
            self.n_actions = 4
            self.n_agents = 3
            self.state_shape = 12
            self.obs_shape = 10
            self.buffer_size = 5 # Small buffer for testing
            self.episode_limit = 2 # Short episode for testing

    args = DummyArgs()
    replay_buffer = ReplayBuffer(args)

    print(f"--- Replay Buffer Initialized ---")
    print(f"Size: {replay_buffer.size}, Current Size: {len(replay_buffer)}")
    print(f"Episode limit: {replay_buffer.episode_limit}")

    # Create a dummy episode batch (single episode)
    episode_len_test = args.episode_limit 
    dummy_episode = {
        'o': np.random.rand(episode_len_test, args.n_agents, args.obs_shape),
        's': np.random.rand(episode_len_test, args.state_shape),
        'u': np.random.randint(0, args.n_actions, size=(episode_len_test, args.n_agents, 1)),
        'r': np.random.rand(episode_len_test, 1),
        'o_next': np.random.rand(episode_len_test, args.n_agents, args.obs_shape),
        's_next': np.random.rand(episode_len_test, args.state_shape),
        'avail_u': np.random.randint(0, 2, size=(episode_len_test, args.n_agents, args.n_actions)),
        'avail_u_next': np.random.randint(0, 2, size=(episode_len_test, args.n_agents, args.n_actions)),
        'u_onehot': np.eye(args.n_actions)[np.random.randint(0, args.n_actions, size=(episode_len_test, args.n_agents, 1)).reshape(-1)].reshape(episode_len_test, args.n_agents, args.n_actions),
        'padded': np.zeros((episode_len_test, 1)),
        'terminated': np.zeros((episode_len_test, 1))
    }
    dummy_episode['terminated'][episode_len_test-1,0] = 1 # Last step is terminated

    print("\n--- Storing Episodes ---")
    num_episodes_to_store = 7
    for i in range(num_episodes_to_store):
        # Create a unique episode for each store call if needed, or reuse dummy
        # For this test, reusing dummy_episode structure with potentially different data is fine.
        # Let's make data slightly different by adding `i` to distinguish
        current_episode_data = {k: v + i if v.dtype in [np.float64, np.float32] else v for k,v in dummy_episode.items()}
        
        replay_buffer.store_episode(current_episode_data)
        print(f"Stored episode {i+1}. Buffer current_idx: {replay_buffer.current_idx}, current_size: {len(replay_buffer)}")
        assert len(replay_buffer) == min(i + 1, args.buffer_size), "Buffer size incorrect after store"

    assert len(replay_buffer) == args.buffer_size, "Buffer not full after enough stores."
    assert replay_buffer.current_idx == num_episodes_to_store % args.buffer_size if num_episodes_to_store >= args.buffer_size else num_episodes_to_store , "current_idx incorrect after wrap around"

    print("\n--- Sampling from Buffer ---")
    sample_batch_size = 2
    if len(replay_buffer) >= sample_batch_size:
        sampled_batch = replay_buffer.sample(sample_batch_size)
        print(f"Sampled batch of {sample_batch_size} episodes.")
        for key, value in sampled_batch.items():
            # Expected shape: (sample_batch_size, episode_limit, ...)
            print(f"  Key '{key}', Shape: {value.shape}")
            if key == 'o':
                assert value.shape == (sample_batch_size, args.episode_limit, args.n_agents, args.obs_shape), f"Shape mismatch for {key}"
            elif key == 's':
                assert value.shape == (sample_batch_size, args.episode_limit, args.state_shape), f"Shape mismatch for {key}"
            elif key == 'r' or key == 'padded' or key == 'terminated':
                 assert value.shape == (sample_batch_size, args.episode_limit, 1), f"Shape mismatch for {key}"
        print("Sampling test passed.")
    else:
        print(f"Skipping sampling test as buffer size ({len(replay_buffer)}) is less than sample batch size ({sample_batch_size}).")

    print("ReplayBuffer tests completed.") 