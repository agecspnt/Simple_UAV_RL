import numpy as np
import math
import random # Added import

class Multi_UAV_env_multi_level_practical_multicheck:
    def __init__(self, args):
        # Environment Parameters from DETAILED_DESIGN.md
        self.n_a_agents = 3  # Number of aerial agents (UAVs)
        self.n_bs = 1  # Number of base stations
        self.n_channel = 2  # Number of communication channels

        self.sinr_AV = 4.5  # dB, UAV communication minimum SINR threshold
        self.sinr_AV_real = 10**(self.sinr_AV / 10)  # Linear value

        self.pw_AV = 30  # dBm, UAV transmission power
        self.pw_AV_real = (10**(self.pw_AV / 10)) / 1000  # W

        self.pw_BS = 40  # dBm, Base station transmission power
        self.pw_BS_real = (10**(self.pw_BS / 10)) / 1000  # W

        self.g_main = 10  # Linear value, Antenna main lobe gain (10 dB)
        self.g_side = 1   # Linear value, Antenna side lobe gain (0 dB)

        self.N0 = -120  # dBm, Noise power
        self.N0_real = (10**(self.N0 / 10)) / 1000  # W

        self.reference_AV = -60  # dB, UAV to BS reference path loss
        self.reference_AV_real = 10**(self.reference_AV / 10)  # Linear value (10^-6)

        self.env_PL_exp = 3  # Path loss exponent

        self.height_BS = 15  # m, Base station height
        self.height_AV = 150 # m, UAV flight height

        self.v_max = 50  # m/s, UAV maximum speed
        self.delta_t = 2  # s, Time step duration

        self.n_speed = 2  # Number of speed levels
        self.vel_actions = np.array([0, 50], dtype=float)  # m/s, Corresponding speed values

        self.BS_locations = np.zeros((self.n_bs, 2), dtype=float) # Initialize placeholder
        self.bs_radius = 200 # Radius for random BS placement

        self.init_location = np.array([[-250.0, 400.0], [-30.33, 930.33], [-30.33, -130.33]], dtype=float)
        self.dest_location = np.array([[1250.0, 400.0], [1030.33, -130.33], [1030.33, 930.33]], dtype=float)

        # Trajectory parameters: slope k, init_y, dest_y (calculated or stored)
        self.trajectory = np.zeros((self.n_a_agents, 6)) # k, b, init_x, init_y, dest_x, dest_y
        for i in range(self.n_a_agents):
            delta_x = self.dest_location[i,0] - self.init_location[i,0]
            delta_y = self.dest_location[i,1] - self.init_location[i,1]
            if delta_x == 0: # Vertical movement
                self.trajectory[i, 0] = np.inf if delta_y > 0 else -np.inf # Slope k
            else:
                self.trajectory[i, 0] = delta_y / delta_x # Slope k
            self.trajectory[i, 1] = self.init_location[i,1] - self.trajectory[i,0] * self.init_location[i,0] # Intercept b (y = kx + b)
            self.trajectory[i, 2] = self.init_location[i,0]
            self.trajectory[i, 3] = self.init_location[i,1]
            self.trajectory[i, 4] = self.dest_location[i,0]
            self.trajectory[i, 5] = self.dest_location[i,1]


        self.max_dist = 1500  # m, Maximum distance for normalization
        self.check_num = 5  # Number of intermediate points for collision/conflict check

        self.collision_threshold = 10 # m, Collision detection distance threshold

        # Learning & Rewards
        self.episode_limit = args.episode_limit if hasattr(args, 'episode_limit') else 50
        self.arrive_reward = 10
        self.all_arrived_reward = 2000
        self.conflict_reward = -100
        self.collision_reward = -100
        self.movement_penalty = -1 # Penalty per step for each UAV not arrived

        # Action space
        self.n_actions = self.n_speed * self.n_channel

        # State variables
        self.current_location = np.copy(self.init_location)
        self.is_arrived = np.zeros(self.n_a_agents, dtype=bool)
        self.is_collision = np.zeros(self.n_a_agents, dtype=bool) # For individual UAV collision status
        self.episode_step = 0
        self.recorded_arrive = [] # For all_arrived_reward logic

        # For get_env_info()
        self.args = args


    def get_state_size(self):
        """Returns the size of the global state."""
        # arrive_state: self.n_a_agents
        # onehot_bs_association: self.n_a_agents * self.n_bs (currently 3*1 = 3)
        # location: self.n_a_agents * 2
        return self.n_a_agents + (self.n_a_agents * self.n_bs) + (self.n_a_agents * 2)

    def get_state(self):
        """Returns the global state of the environment."""
        arrive_state = self.is_arrived.astype(float).flatten()

        # onehot_bs_association: For now, all UAVs are associated with the single BS
        # This part is designed for future multi-BS scenarios.
        # For a single BS, it's effectively fixed but kept for structural consistency.
        bs_association_flags = np.zeros(self.n_a_agents * self.n_bs)
        for i in range(self.n_a_agents):
            # Assuming UAV i is associated with BS 0 (the only BS)
            bs_association_flags[i * self.n_bs + 0] = 1.0 
        
        location_norm = self.current_location.flatten() / self.max_dist # Normalized location

        state = np.concatenate((arrive_state, bs_association_flags, location_norm))
        return state

    def get_obs_size(self):
        """Returns the size of the individual agent observation."""
        # 1. Arrived_status / dist_to_dest_norm (1)
        # 2. Agent type (1, fixed for UAV)
        # 3. Other UAVs info: (N_AV-1) * (N_BS_assoc_onehot + dist_to_al_bs_norm + safe_dist_related_to_al_norm)
        #    (self.n_a_agents - 1) * (self.n_bs + 1 + 1)
        # 4. Other UAVs arrived status: (N_AV-1)
        # 5. Current UAV ID (one-hot): N_AV
        
        # Current UAV info:
        # - Own distance to destination (normalized): 1
        # - Own location (normalized, x, y): 2
        # - Own velocity (selected level, normalized by n_speed): 1 (Not directly in obs by design, but good to consider if needed)
        # - Own channel (selected, normalized by n_channel): 1 (Not directly in obs by design)
        
        # Other UAVs' info: For each other UAV (N_AV - 1):
        # - Relative position (normalized, dx, dy): 2
        # - Is it arrived?: 1
        
        # BS Info (for each BS, currently 1):
        # - Distance to BS (normalized): 1
        # - BS Location (normalized, x,y): 2 (can be excluded if BS is static and known)

        # Total:
        # Own UAV:
        #   - dist_to_dest_norm: 1
        #   - current_loc_norm (x,y): 2
        #   - is_arrived_own: 1 (could be combined with dist_to_dest)
        # Other UAVs (N_AV - 1 agents):
        #   - relative_loc_norm (dx, dy): 2
        #   - is_arrived_other: 1
        #   - dist_to_other_uav_norm: 1
        # Agent ID (one-hot): N_AV
        
        # Let's follow the detailed design more closely for now:
        # 1. Arrived status/dist (1)
        # 2. Agent type (1)
        # 3. Other UAVs info: (self.n_a_agents - 1) * (self.n_bs + 1 + 1)
        #    -> (self.n_a_agents - 1) * (self.n_bs + 2)
        # 4. Other UAVs arrived (self.n_a_agents - 1)
        # 5. Own ID (self.n_a_agents)
        
        # Simplified from doc:
        # 1. dist_to_dest_normalized (1)
        # 2. own_location_normalized (2)
        # For each other agent (N_AV - 1):
        #   3. relative_position_to_other_agent_normalized (2)
        #   4. other_agent_is_arrived (1)
        #   5. other_agent_dist_to_its_dest_normalized (1)
        # 6. agent_id_one_hot (N_AV)
        # Total for this simplified version: 1 + 2 + (N_AV-1)*(2+1+1) + N_AV
        # = 3 + (N_AV-1)*4 + N_AV
        # = 3 + 4*N_AV - 4 + N_AV
        # = 5*N_AV - 1
        # For N_AV=3: 15-1 = 14

        # Let's try to stick to the markdown as much as possible:
        # 1. Arrived status/Remaining dist: 1 element
        #    (Let's use normalized remaining distance if not arrived, 0 if arrived)
        # 2. Agent type symbol: 1 element (e.g., 1.0 for UAV)
        # 3. Other UAVs' info ((N_AV-1) * (N_BS + 2)):
        #    For each other UAV `al_id`:
        #    - `bs_association` (N_BS elements, one-hot encoding of al_id's BS). For N_BS=1, this is just [1.0]
        #    - `dist_to_al_id_bs_norm`: 1 element (distance from current agent to al_id's BS)
        #    - `safe_dist_related_to_al_id_norm`: 1 element (distance from current agent to al_id, normalized)
        # 4. Other UAVs' arrived status (N_AV-1 elements)
        # 5. Current UAV ID (one-hot N_AV elements)

        obs_size_val = 0
        obs_size_val += 1  # 1. Arrived status / Remaining dist
        obs_size_val += 1  # 2. Agent type symbol
        obs_size_val += (self.n_a_agents - 1) * (self.n_bs + 1 + 1) # 3. Other UAVs' info
        obs_size_val += (self.n_a_agents - 1) # 4. Other UAVs' arrived status
        obs_size_val += self.n_a_agents # 5. Current UAV ID (one-hot)
        return obs_size_val


    def get_obs_agent(self, agent_id):
        """Returns the observation for a specific agent."""
        obs = []

        # 1. Arrived status / Remaining distance
        my_loc = self.current_location[agent_id]
        my_dest = self.dest_location[agent_id]
        if self.is_arrived[agent_id]:
            obs.append(0.0) # Arrived
        else:
            dist_to_dest = np.linalg.norm(my_dest - my_loc)
            obs.append(dist_to_dest / self.max_dist) # Normalized distance

        # 2. Agent type symbol (1.0 for UAV)
        obs.append(1.0)

        # 3. Other UAVs' info
        for other_id in range(self.n_a_agents):
            if other_id == agent_id:
                continue
            
            other_loc = self.current_location[other_id]
            # other_dest = self.dest_location[other_id] # Not directly needed for this part of obs

            # 3.1 BS association for other_id (one-hot)
            # Assuming other_id is associated with BS 0 (the only BS)
            bs_assoc_other = np.zeros(self.n_bs)
            if self.n_bs > 0 : # Should always be true given self.n_bs = 1
                 bs_assoc_other[0] = 1.0 
            obs.extend(bs_assoc_other.tolist())

            # 3.2 Distance from current agent (agent_id) to other_id's BS
            # Assuming other_id is associated with BS_locations[0]
            other_bs_loc_3d = np.append(self.BS_locations[0], self.height_BS)
            my_loc_3d = np.append(my_loc, self.height_AV)
            dist_to_other_bs = np.linalg.norm(other_bs_loc_3d - my_loc_3d)
            obs.append(dist_to_other_bs / self.max_dist)

            # 3.3 Safe distance related to other_id (distance from current agent to other_id)
            dist_to_other_agent = np.linalg.norm(my_loc_3d - np.append(other_loc, self.height_AV))
            obs.append(dist_to_other_agent / self.max_dist)

        # 4. Other UAVs' arrived status
        for other_id in range(self.n_a_agents):
            if other_id == agent_id:
                continue
            obs.append(1.0 if self.is_arrived[other_id] else 0.0)

        # 5. Current UAV ID (one-hot)
        agent_one_hot = np.zeros(self.n_a_agents)
        agent_one_hot[agent_id] = 1.0
        obs.extend(agent_one_hot.tolist())
        
        return np.array(obs, dtype=float)

    def get_obs(self):
        """Returns observations for all agents."""
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_a_agents)]
        return agents_obs

    def get_env_info(self):
        """Returns environment information."""
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.n_actions,
            "n_agents": self.n_a_agents,
            "episode_limit": self.episode_limit,
            "agent_num": self.n_a_agents, # For compatibility with some runners
            "action_space": [np.arange(self.n_actions)] * self.n_a_agents # Discrete action space
        }
        return env_info

    def _calculate_path_loss(self, loc1_2d, loc2_2d, h1, h2):
        """Calculates 3D distance and path loss."""
        loc1_3d = np.append(loc1_2d, h1)
        loc2_3d = np.append(loc2_2d, h2)
        dist_3d = np.linalg.norm(loc1_3d - loc2_3d)
        if dist_3d < 1e-6: # Avoid division by zero or very small distances
            dist_3d = 1e-6
        path_loss_linear = self.reference_AV_real * (dist_3d ** self.env_PL_exp)
        if path_loss_linear == 0: # Avoid division by zero if path loss is zero
            return np.inf 
        return 1.0 / path_loss_linear # Return gain (1/PL)


    def _calculate_sinr(self, agent_id, channel_idx, current_agent_locations, agent_actions_decoded):
        """
        Calculates SINR for agent_id on channel_idx.
        Assumes agent_id is trying to communicate with the BS.
        current_agent_locations: 2D locations of all UAVs at this step.
        agent_actions_decoded: list of (vel_idx, ch_idx) for all agents.
        """
        # Desired signal: UAV_agent_id <-> BS_0
        # Assuming UAV is transmitting to BS, BS is receiver
        # Or BS is transmitting to UAV, UAV is receiver
        # The problem description implies UAVs select a channel for "communication activities"
        # and conflict is based on this. Let's assume the primary link is UAV <-> BS.

        # For simplicity, let's assume the UAV is the transmitter and BS is the receiver.
        # The logic can be adapted if we need to distinguish uplink/downlink explicitly.
        # The design doc says: "SINR of UAV on its chosen channel"
        # "SINR ... consider UAV as transmitter ... or receiver"
        # "Conflict based on SINR < threshold"

        # Path loss between agent_id and BS_0
        # Using self.current_location for SINR calculation as it's post-move
        uav_loc_2d = current_agent_locations[agent_id]
        bs_loc_2d = self.BS_locations[0]
        
        gain_signal = self._calculate_path_loss(uav_loc_2d, bs_loc_2d, self.height_AV, self.height_BS)
        if np.isinf(gain_signal): # Path loss was zero
             received_signal_power = 0
        else:
             # Assuming UAV transmits, BS receives: P_rx = P_tx_UAV * G_main_UAV * G_main_BS / PL
             # Design doc: P_rx_dest = self.pw_AV_real * (self.g_main^2) / PL_UAV_BS
             # The g_main^2 suggests antenna gains at both ends.
             # Let's use pw_AV_real as the Tx power for the signal part.
             received_signal_power = self.pw_AV_real * (self.g_main**2) * gain_signal


        total_interference_power = 0

        # Interference from other UAVs transmitting on the same channel to the *same BS*
        # (This is one interpretation: UAV_k -> BS_0 is interference for UAV_i -> BS_0 link)
        for interferer_id in range(self.n_a_agents):
            if interferer_id == agent_id:
                continue

            _, interferer_channel_idx = agent_actions_decoded[interferer_id]
            interferer_vel_idx, _ = agent_actions_decoded[interferer_id]
            interferer_is_active = (not self.is_arrived[interferer_id]) and (self.vel_actions[interferer_vel_idx] > 0)


            if interferer_channel_idx == channel_idx and interferer_is_active:
                interferer_loc_2d = current_agent_locations[interferer_id]
                # Path loss from interferer_UAV to the target_BS (BS_0)
                gain_interf_uav_to_bs = self._calculate_path_loss(interferer_loc_2d, bs_loc_2d, self.height_AV, self.height_BS)
                if np.isinf(gain_interf_uav_to_bs):
                    interf_power_contrib = 0
                else:
                    # Design doc: P_interf = self.pw_AV_real * self.g_main * self.g_side / PL_interfUAV_BS
                    # This assumes interferer's main lobe towards its own target, side lobe towards this BS.
                    # Or, if target is this BS, main lobe.
                    # For simplicity, assume pw_AV_real is Tx power, and use (g_main * g_side) for gain product.
                    # This is a common simplification.
                    interf_power_contrib = self.pw_AV_real * self.g_main * self.g_side * gain_interf_uav_to_bs
                total_interference_power += interf_power_contrib
        
        # Note: The design doc also mentions interference "when UAV is receiver" from other BS or other UAVs to *this* UAV.
        # The current SINR calculation focuses on UAV->BS link quality at the BS.
        # If the UAV is a receiver (e.g., BS -> UAV link), then:
        # Signal: BS_0 -> UAV_agent_id. Power_RX = self.pw_BS_real * (self.g_main**2) * gain_signal
        # Interference sources:
        #   a. Other UAV_k -> UAV_agent_id on same channel.
        #      P_interf_UAV = self.pw_AV_real * self.g_main * self.g_side * gain_interfUAV_to_targetUAV
        #   b. Other BS_j -> UAV_agent_id on same channel (if multi-BS). (Not relevant for n_bs=1)

        # For now, let's stick to the "UAV as transmitter, BS as receiver" model for conflict assessment,
        # as it's simpler and captures inter-UAV interference on the uplink to the shared BS.
        # This seems to align with "evaluate its SINR on its chosen single channel".

        denominator = total_interference_power + self.N0_real
        if denominator == 0: # Avoid division by zero
            return np.inf # Effectively, very high SINR if no noise and no interference
        
        sinr_real = received_signal_power / denominator
        return sinr_real


    def delta_location(self, vel_level_idx, agent_id):
        """Calculates the displacement (delta_loc) for an agent."""
        vel = self.vel_actions[vel_level_idx]
        moving_dist = vel * self.delta_t

        if moving_dist == 0:
            return np.array([0.0, 0.0])

        # Trajectory parameters
        slope_param = self.trajectory[agent_id, 0] # k
        # init_y = self.trajectory[agent_id, 3] # Not directly used here for angle
        # dest_y = self.trajectory[agent_id, 5] # Not directly used here for angle
        
        # Determine direction based on current location vs destination
        current_pos = self.current_location[agent_id]
        dest_pos = self.dest_location[agent_id]
        
        direction_vector = dest_pos - current_pos
        dist_to_dest = np.linalg.norm(direction_vector)

        if dist_to_dest < 1e-6 : # Already at destination (should be caught by is_arrived)
             return np.array([0.0,0.0])

        if moving_dist >= dist_to_dest: # Will reach or overshoot destination
            return direction_vector # Move directly to destination

        # Calculate movement along the line connecting current to destination
        # This is more robust than relying purely on initial trajectory slope
        # if the UAV ever deviates (though current model doesn't allow deviation from path type)
        
        unit_direction_vector = direction_vector / dist_to_dest
        delta_x = unit_direction_vector[0] * moving_dist
        delta_y = unit_direction_vector[1] * moving_dist
        
        return np.array([delta_x, delta_y])


    def check_action_collision(self, inter_locations_all_agents):
        """
        Checks for collisions between any pair of UAVs based on their intermediate path locations.
        inter_locations_all_agents: shape (n_a_agents, check_num, 2)
        Returns a boolean array of size n_a_agents, where True indicates the UAV was involved in a collision.
        """
        collision_flags = np.zeros(self.n_a_agents, dtype=bool)
        for i in range(self.n_a_agents):
            for j in range(i + 1, self.n_a_agents):
                # Check collision between UAV i and UAV j
                collided_this_pair = False
                for k_check_point in range(self.check_num):
                    loc_i_2d = inter_locations_all_agents[i, k_check_point, :]
                    loc_j_2d = inter_locations_all_agents[j, k_check_point, :]
                    
                    # 3D distance check (assuming fixed height self.height_AV for all)
                    dist_sq = (loc_i_2d[0] - loc_j_2d[0])**2 + \
                              (loc_i_2d[1] - loc_j_2d[1])**2
                              # (self.height_AV - self.height_AV)**2 == 0
                    
                    if dist_sq < self.collision_threshold**2:
                        collision_flags[i] = True
                        collision_flags[j] = True
                        collided_this_pair = True
                        break # This pair collided, no need to check more points for this pair
                # if collided_this_pair: # If optimization needed, can break outer loops too if any collision is catastrophic
                    # pass
        return collision_flags


    def step(self, actions):
        """
        Executes one time step in the environment.
        actions: List or array of integers, one action for each agent.
        """
        self.episode_step += 1
        rewards = np.zeros(self.n_a_agents) # Individual rewards initially
        
        # Decode actions: (integer action) -> (velocity_idx, channel_idx)
        agent_actions_decoded = []
        for i in range(self.n_a_agents):
            action_int = actions[i]
            vel_level_idx = action_int // self.n_channel
            channel_idx = action_int % self.n_channel
            agent_actions_decoded.append((vel_level_idx, channel_idx))

        # UAV Movement and Intermediate Locations
        # Store pre-move locations for inter_location calculation
        prev_locations = np.copy(self.current_location)
        inter_locations = np.zeros((self.n_a_agents, self.check_num, 2)) # 2D locations

        for i in range(self.n_a_agents):
            delta_loc = np.array([0.0, 0.0])
            if not self.is_arrived[i]:
                vel_idx, _ = agent_actions_decoded[i]
                delta_loc = self.delta_location(vel_idx, i)
                self.current_location[i] += delta_loc
            
            # Generate intermediate points for collision checking
            # Based on design: inter_locations[i, k, :] = (prev_loc) + k * (delta_loc_step / (check_num-1))
            # where delta_loc_step is the total displacement in this step.
            # The formula in doc: (self.current_location[i] - delta_loc) is prev_loc
            for k in range(self.check_num):
                if self.check_num == 1: # Avoid division by zero if check_num is 1
                    inter_locations[i, k, :] = prev_locations[i] + delta_loc # End point
                else:
                    inter_locations[i, k, :] = prev_locations[i] + k * (delta_loc / (self.check_num - 1))
        
        # Arrival Check & Update
        for i in range(self.n_a_agents):
            if not self.is_arrived[i]:
                current_dist_to_dest = np.linalg.norm(self.current_location[i] - self.dest_location[i])
                if current_dist_to_dest < 10.0: # Arrival threshold in meters
                    self.is_arrived[i] = True
                    self.current_location[i] = np.copy(self.dest_location[i]) # Snap to destination
                    if i not in self.recorded_arrive: # First time arrival for this UAV
                        rewards[i] += self.arrive_reward
                        self.recorded_arrive.append(i)

        # Collision Detection
        # self.is_collision is an array [bool, bool, bool] indicating if UAV i was involved in a collision
        self.is_collision = self.check_action_collision(inter_locations)


        # Reward Calculation
        # Initialize step reward (sum of individual agent rewards for this step)
        step_reward_val = 0

        # 1. Movement Penalty (applied per agent not yet arrived)
        for i in range(self.n_a_agents):
            if not self.is_arrived[i]:
                step_reward_val += self.movement_penalty
        
        # 2. Arrival Rewards (already added to individual `rewards`, sum them up here)
        #    The design document implies a single shared reward signal.
        #    So `arrive_reward` contributes to the global reward.
        #    The individual `rewards` array was a temp placeholder.
        #    Let's use a single `current_step_global_reward`.
        
        current_step_global_reward = 0
        
        # Re-evaluate arrival rewards for the global reward signal
        newly_arrived_this_step = []
        for i in range(self.n_a_agents):
            # is_arrived was updated above.
            # We need to check if it *became* arrived in *this* step.
            # This is implicitly handled by `recorded_arrive` logic.
            # If UAV i is in `self.recorded_arrive` and its `is_arrived` is true,
            # it means it was counted.
            # We need to sum up `self.arrive_reward` for those that *just* arrived.
            # The current `rewards` array (which should be renamed) holds this.
            pass # Will be handled more cleanly below.

        # Clear and recalculate global reward components
        # Movement penalty
        for i in range(self.n_a_agents):
            if not self.is_arrived[i]:
                current_step_global_reward += self.movement_penalty
        
        # Individual arrival rewards
        for i in range(self.n_a_agents):
            if self.is_arrived[i] and i not in self.recorded_arrive: # This logic is slightly off, recorded_arrive updated before this block
                # Let's fix recorded_arrive update point.
                # `is_arrived` is final for this step.
                # If `is_arrived[i]` is true, and it wasn't in `recorded_arrive` *before this step's arrival processing*, it's a new arrival.
                # The `rewards` array was intended for this.
                # Let's adjust:
                pass # This is tricky. Simplest: if is_arrived[i] is true AND it wasn't true at start of step.
                     # For VDN, we need one global reward.

        # Let's use the pre-calculated individual rewards from arrival for now
        # The `rewards` array should actually be the global reward components.
        
        # Corrected approach for global reward:
        _current_reward = 0 
        
        # Movement penalty
        for i in range(self.n_a_agents):
            if not self.is_arrived[i]: # If UAV i has not arrived
                 _current_reward += self.movement_penalty

        # Arrival rewards
        # Check if UAVs arrived in *this* step
        for i in range(self.n_a_agents):
            # If UAV i is now marked as arrived, and it wasn't in self.recorded_arrive *before* this step's processing
            # This logic is easier if `self.recorded_arrive` is managed carefully.
            # The `rewards` array which was updated during arrival check is good.
            # `rewards` was reset to zeros.
            # Let's re-do the arrival reward accumulation for the global signal:
            pass # This logic has become convoluted.

        # Simpler reward accumulation:
        _reward_this_step = 0
        
        # Penalties first
        for i in range(self.n_a_agents):
            if not self.is_arrived[i]: # Active UAVs get movement penalty
                _reward_this_step += self.movement_penalty
            
            if self.is_collision[i]: # If UAV i is involved in a collision
                _reward_this_step += self.collision_reward
                # Collision penalty might be applied once per collision event, or per UAV involved.
                # Design: "if self.is_collision[i] ... received this penalty" -> per UAV involved.

        # SINR-based Conflict Penalties
        for i in range(self.n_a_agents):
            vel_idx, ch_idx = agent_actions_decoded[i]
            is_active_for_comms = (not self.is_arrived[i]) and (self.vel_actions[vel_idx] > 0)
            
            if is_active_for_comms:
                # Calculate SINR for UAV i on its chosen channel ch_idx
                # Need current locations (post-move) for SINR
                sinr_val = self._calculate_sinr(i, ch_idx, self.current_location, agent_actions_decoded)
                if sinr_val < self.sinr_AV_real:
                    _reward_this_step += self.conflict_reward
        
        # Arrival Rewards
        # Check for newly arrived agents in this step
        # `self.is_arrived` is updated. `self.recorded_arrive` should only contain agents that have *ever* arrived.
        # A better way:
        # At the start of step, copy `self.is_arrived` to `was_arrived_last_step`.
        # Then `newly_arrived_this_step_mask = self.is_arrived & ~was_arrived_last_step`.
        # For now, using the `self.recorded_arrive` list that was updated during arrival logic.
        
        # Let's refine the arrival reward and recorded_arrive logic.
        # `self.recorded_arrive` stores IDs of agents whose `self.arrive_reward` has been given.
        # It should be updated *after* giving the reward.
        
        # Re-do arrival rewards for global reward:
        # `self.is_arrived` is the status *after* movement and arrival check.
        for i in range(self.n_a_agents):
            if self.is_arrived[i]: # If currently arrived
                if i not in self.recorded_arrive: # And this is the first time it's noted as arrived (for reward purposes)
                    _reward_this_step += self.arrive_reward
                    self.recorded_arrive.append(i) # Mark as rewarded for single arrival

        # All arrived reward
        if len(self.recorded_arrive) == self.n_a_agents: # All UAVs have arrived at some point
            # Check if this is the first step all are simultaneously arrived
            # Or, if this reward is given once when the last UAV arrives.
            # "All UAVs arrived时给予" - when the condition becomes true.
            # This needs a flag like `all_arrived_bonus_given`.
            if not hasattr(self, 'all_arrived_bonus_given_flag') or not self.all_arrived_bonus_given_flag:
                 # Check if ALL are currently arrived
                 all_currently_arrived = True
                 for i in range(self.n_a_agents):
                     if not self.is_arrived[i]:
                         all_currently_arrived = False
                         break
                 if all_currently_arrived:
                    _reward_this_step += self.all_arrived_reward
                    self.all_arrived_bonus_given_flag = True # Give this bonus only once
        
        # Check termination condition
        terminated = False
        if self.episode_step >= self.episode_limit:
            terminated = True
        
        # All UAVs arrived also terminates the episode
        if len(self.recorded_arrive) == self.n_a_agents: # If all have arrived (at any point)
            all_currently_arrived_check = True
            for i in range(self.n_a_agents):
                if not self.is_arrived[i]:
                    all_currently_arrived_check = False; break
            if all_currently_arrived_check:
                 terminated = True


        # Info dict
        info = {
            'is_success': len(self.recorded_arrive) == self.n_a_agents and terminated, # Success if terminated and all arrived
            'collisions': np.sum(self.is_collision), # Number of UAVs involved in collision this step
            # Add other relevant info if needed
        }
        if terminated and not hasattr(self, 'all_arrived_bonus_given_flag'): # Ensure flag exists if episode ends early
            self.all_arrived_bonus_given_flag = False


        # The environment should return a single global reward for VDN
        # return _reward_this_step, terminated, info
        # The runner expects reward to be a scalar.
        # The `_reward` function was specified as `_reward(self, actions)` in design,
        # but it's usually part of `step`. Let's rename _reward_this_step to final_reward.

        final_reward = _reward_this_step
        
        # For MARL, often the obs, state are returned by reset and step.
        # The step usually returns: reward, terminated, info (and obs, state are obtained by get_obs, get_state after step)
        # However, many frameworks expect: next_obs, reward, done, info
        # The current setup (MAgent) has rollout worker call env.step -> get r, done, env_info
        # then collect o, s, u, r, o_next, s_next, terminated etc.
        # So returning (scalar_reward, terminated, info) is standard for many MARL envs.

        return final_reward, terminated, info

    # Helper method to be called at the beginning of reset to clear flags like all_arrived_bonus
    def _clear_episode_flags(self):
        if hasattr(self, 'all_arrived_bonus_given_flag'):
            del self.all_arrived_bonus_given_flag
        self.recorded_arrive = [] # Also reset here for safety, though reset() does it.

    # Override reset to include this
    def reset(self):
        """Resets the environment to the initial state."""
        self._clear_episode_flags() # Clear flags first
        self.episode_step = 0
        self.current_location = np.copy(self.init_location)
        self.is_arrived = np.zeros(self.n_a_agents, dtype=bool)
        self.is_collision = np.zeros(self.n_a_agents, dtype=bool)
        # self.recorded_arrive = [] # Cleared by _clear_episode_flags

        # Randomize BS locations within a circle of bs_radius
        for i in range(self.n_bs):
            # Generate random angle
            angle = random.uniform(0, 2 * math.pi)
            # Generate random radius (sqrt for uniform distribution within circle)
            r = self.bs_radius * math.sqrt(random.uniform(0, 1))
            # Convert polar to cartesian coordinates
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            self.BS_locations[i] = [x, y]
        # Return obs and state as per common MARL practices
        return self.get_obs(), self.get_state()


# Example of how to use the Padded class wrapper if needed later
# class Multi_UAV_env_multi_level_practical_multicheck_padded(Multi_UAV_env_multi_level_practical_multicheck):
#     def __init__(self, args):
#         super().__init__(args)
#
#     def step(self, actions, step_param_not_used): # The 'step' parameter is not used by base class
#         # This shows that the padded version's step signature is different.
#         # For now, we are implementing the base class.
#         return super().step(actions)


if __name__ == '__main__':
    # Dummy args for testing
    class DummyArgs:
        def __init__(self):
            self.episode_limit = 60
            # Add any other args the environment might expect from self.args if used internally
            # For example, if get_env_info or other parts directly access self.args.param

    args = DummyArgs()
    env = Multi_UAV_env_multi_level_practical_multicheck(args)

    print("--- Environment Test ---")
    print(f"Number of Agents: {env.n_a_agents}")
    print(f"Number of Actions: {env.n_actions}")
    print(f"State Size: {env.get_state_size()}")
    print(f"Observation Size: {env.get_obs_size()}")
    print(f"Max episode steps: {env.episode_limit}")

    obs, state = env.reset()
    print("\nInitial Observations (Agent 0):", obs[0])
    print("Initial State:", state)
    assert len(obs[0]) == env.get_obs_size(), "Obs size mismatch"
    assert len(state) == env.get_state_size(), "State size mismatch"


    total_reward_acc = 0
    done = False
    current_step = 0

    print("\n--- Running a Sample Episode ---")
    while not done and current_step < env.episode_limit :
        current_step += 1
        random_actions = [np.random.randint(0, env.n_actions) for _ in range(env.n_a_agents)]
        
        # Store pre-step state for comparison
        # prev_locs_test = np.copy(env.current_location)
        
        reward, done, info = env.step(random_actions)
        
        next_obs = env.get_obs()
        next_state = env.get_state()
        
        total_reward_acc += reward
        
        print(f"Step: {current_step}, Actions: {random_actions}, Reward: {reward:.2f}, Done: {done}")
        # print(f"  Locations: {env.current_location}")
        # print(f"  Arrived: {env.is_arrived}")
        # print(f"  Collisions: {env.is_collision}")
        if 'collisions' in info and info['collisions'] > 0 :
             print(f"  Collision detected this step for UAVs: {env.is_collision}")

        if done:
            print(f"Episode finished after {current_step} steps.")
            print(f"Final reward: {total_reward_acc:.2f}")
            print(f"Success: {info.get('is_success', False)}")
            break
    
    if not done:
        print(f"Episode timed out after {env.episode_limit} steps.")
        print(f"Final reward: {total_reward_acc:.2f}")

    print("\n--- Testing get_env_info ---")
    env_info_dict = env.get_env_info()
    print(env_info_dict)
    assert env_info_dict["state_shape"] == env.get_state_size()
    assert env_info_dict["obs_shape"] == env.get_obs_size()
    assert env_info_dict["n_actions"] == env.n_actions
    assert env_info_dict["n_agents"] == env.n_a_agents

    print("\n--- Testing specific scenarios (conceptual) ---")
    # Example: Test collision
    env.reset()
    # Force two UAVs to be at the same spot (or very close)
    env.current_location[0] = np.array([100.0, 100.0])
    env.current_location[1] = np.array([100.5, 100.5]) # Within 10m collision_threshold
    env.current_location[2] = np.array([500.0, 500.0]) # Away

    # Simulate a step where they don't move far, so inter_locations will be close
    # Create dummy inter_locations for testing check_action_collision directly
    test_inter_locs = np.zeros((env.n_a_agents, env.check_num, 2))
    for ag_idx in range(env.n_a_agents):
        for ch_idx in range(env.check_num):
            test_inter_locs[ag_idx, ch_idx, :] = env.current_location[ag_idx] 
            # (Simplification: all check points are the same as current_location)
    
    collision_results = env.check_action_collision(test_inter_locs)
    print(f"Manual collision check for UAVs at almost same spot: {collision_results}")
    assert collision_results[0] == True, "Collision not detected for UAV 0"
    assert collision_results[1] == True, "Collision not detected for UAV 1"
    assert collision_results[2] == False, "Collision incorrectly detected for UAV 2"

    # Test SINR calculation (conceptual - needs specific setup)
    env.reset()
    env.current_location[0] = np.array([0.0, 0.0]) # UAV 0
    env.current_location[1] = np.array([10.0, 0.0]) # UAV 1 (potential interferer)
    env.current_location[2] = np.array([500.0, 20.0]) # UAV 2 (far away)
    
    # Actions: UAV0 (0,0), UAV1 (0,0), UAV2 (0,1) -> UAV0 and UAV1 on channel 0
    # Vel_idx=0 (50m/s), Chan_idx=0
    # Action_int = vel_idx * n_channel + chan_idx
    # Action for UAV0: 0*2+0 = 0
    # Action for UAV1: 0*2+0 = 0
    # Action for UAV2: 0*2+1 = 1
    decoded_actions_test = [(0,0), (0,0), (0,1)] # (vel_idx, ch_idx)
    env.is_arrived[:] = False # Ensure they are active

    sinr_uav0 = env._calculate_sinr(agent_id=0, channel_idx=0, 
                                   current_agent_locations=env.current_location, 
                                   agent_actions_decoded=decoded_actions_test)
    print(f"SINR for UAV0 (with UAV1 interfering on same channel 0): {sinr_uav0:.4e} (linear), dB: {10*np.log10(sinr_uav0) if sinr_uav0 > 0 else -np.inf:.2f}")

    # Test SINR for UAV2 (on different channel, should be higher SINR if no other interferers on ch1)
    sinr_uav2 = env._calculate_sinr(agent_id=2, channel_idx=1,
                                   current_agent_locations=env.current_location,
                                   agent_actions_decoded=decoded_actions_test)
    print(f"SINR for UAV2 (on channel 1, no other same-channel interferers): {sinr_uav2:.4e} (linear), dB: {10*np.log10(sinr_uav2) if sinr_uav2 > 0 else -np.inf:.2f}")
    
    # Check if SINR for UAV0 is below threshold
    if sinr_uav0 < env.sinr_AV_real:
        print(f"UAV0 SINR ({10*np.log10(sinr_uav0):.2f} dB) is BELOW threshold ({env.sinr_AV} dB). Conflict penalty should apply.")
    else:
        print(f"UAV0 SINR ({10*np.log10(sinr_uav0):.2f} dB) is ABOVE threshold ({env.sinr_AV} dB). No conflict penalty for UAV0.")

    print("\nBasic tests completed.") 