# 基于 VDN 的多无人机速度与信道联合优化强化学习环境设计

## 1. 项目简介与目标

本项目旨在使用值分解网络 (Value-Decomposition Networks, VDN) 算法，设计一个多智能体强化学习环境。该环境用于解决一个特定的优化问题：**控制3架无人机（UAV）从各自的起点飞行至指定目标点，并在每一步联合选择飞行速度和通信信道，以最小化所有无人机的总飞行时间**。

**核心挑战与优化目标**：
-   **多智能体协调**：3架UAV需要独立决策但共同对整体目标做出贡献。
-   **离散动作空间**：每个UAV每一步需要同时决定离散的速度（2档）和离散的通信信道（2条）。
-   **优化目标**：最小化三架UAV到达各自目标点的总计飞行时间（即，每个UAV未到达目标点时的累积时间步数之和）。

VDN 是一种适用于合作型多智能体任务的算法，它学习每个智能体的独立效用函数 \(Q_i\)，然后将它们相加得到联合动作的总效用 \(Q_{tot} = \sum_i Q_i\)。这使得它适合处理此类问题，其中个体贡献可以汇总为团队目标。

## 2. 算法与项目结构

根据您提供的项目结构，各个模块在VDN算法框架下的作用如下：

*   **`main.py`**: 程序主入口。负责：
    *   解析和设置通用参数及特定算法的参数 (通过调用 `common/arguments.py` 中的函数)。
    *   初始化多无人机飞行环境 (`Multi_UAV_env.py` 中的 `Multi_UAV_env_multi_level_practical_multicheck`)。
    *   从环境中获取动作空间、状态空间、观测空间等信息，并配置给参数对象。
    *   实例化训练执行器 `Runner` (`runner.py`)。
    *   根据参数启动训练流程 (`runner.run()`) 或评估流程 (`runner.evaluate()`)。

*   **`Multi_UAV_env.py`**: （本次设计的重点）实现多无人机飞行环境的逻辑。主要包含：
    *   环境的初始化 (`__init__`)，定义无人机数量、动作空间、物理参数、目标位置、基站位置等。
    *   状态表示 (`get_state`) 和局部观测表示 (`get_obs`, `get_obs_agent`) 的生成。
    *   动作执行 (`step`)，根据智能体选择的动作更新环境状态。
    *   状态转移逻辑，模拟无人机的移动和交互。
    *   奖励计算 (`_reward`)，根据无人机的行为（如到达目标、发生冲突/碰撞等）给出奖励信号。
    *   环境重置 (`reset`)，用于开始新的 episode。
    *   提供环境信息 (`get_env_info`) 给其他模块。

*   **`runner.py`**: 训练执行器。负责：
    *   初始化智能体集合 (`agent/agent_UAV.py` 中的 `Agents` 类)。
    *   初始化经验产生器 `RolloutWorker` (`common/rollout.py`)。
    *   初始化经验回放缓冲区 `ReplayBuffer` (`common/replay_buffer.py`)。
    *   在主训练循环 (`run` 方法) 中：
        *   调用 `RolloutWorker` 生成 episodes，收集智能体与环境的交互数据。
        *   将收集到的经验数据存入 `ReplayBuffer`。
        *   从 `ReplayBuffer` 中采样数据。
        *   调用智能体集合的训练方法 (`agents.train()`) 来更新策略网络。
        *   定期进行评估 (`evaluate` 方法)，并记录性能指标。

*   **`agent/agent_UAV.py`**: 定义 `Agents` 类，作为所有智能体的管理器。主要负责：
    *   根据配置参数 (如 `args.alg`) 初始化并持有一个具体的策略对象 (例如，`policy.vdn_UAV.VDN`)。
    *   提供 `choose_action` 方法，供 `RolloutWorker` 调用。此方法会利用其内部策略对象的网络 (如 `VDN` 策略中的 `eval_mlp`) 为单个智能体基于其局部观测选择动作。
    *   提供 `train` 方法，供 `Runner` 调用。此方法会调用其内部策略对象的学习方法 (如 `VDN` 策略中的 `learn` 方法) 来更新网络参数。

*   **`network/base_net.py`**: 包含构成单个智能体 Q 网络的基础神经网络模块。例如：
    *   `MLP` (多层感知机)
    *   `RNN` (循环神经网络，如 GRUCell)
    *   `D3QN` (Dueling Double Deep Q-Network)
    *   这些网络通常被 `policy` 目录下的具体策略类 (如 `policy/vdn_UAV.py` 中的 `VDN` 类) 实例化，作为每个智能体独立的 Q 网络 (或其一部分)。输入通常是该智能体的局部观测（可能还会拼接其他信息如 agent ID）。

*   **`network/vdn_net.py`**: 实现 VDN (Value Decomposition Network) 的混合网络 (`VDNNet` 类)。
    *   其核心功能是将各个独立智能体 Q 网络输出的、对应于所选动作的 Q 值 (Q_i(s_i, a_i)) 相加，得到总的联合 Q 值 (Q_tot(mathbf{s},mathbf{a}))。
    *   在标准的 VDN 中，这个混合网络本身没有可学习的参数，仅执行求和操作。

*   **`policy/vdn_UAV.py`**: 实现 VDN 策略 (`VDN` 类)。主要负责：
    *   初始化：
        *   每个智能体的评估 Q 网络 (`self.eval_mlp`，通常实例化自 `network/base_net.py` 中的 `D3QN` 或 `MLP`) 和对应的目标 Q 网络 (`self.target_mlp`)。
        *   VDN 混合网络 (`self.eval_vdn_net`，实例化自 `network/vdn_net.py` 中的 `VDNNet`) 和对应的目标混合网络 (`self.target_vdn_net`)。
        *   优化器，用于更新 `eval_mlp` 和 `eval_vdn_net` (如果 `vdn_net` 设计为有参数的话) 的参数。
    *   学习 (`learn` 方法)：
        *   从 `ReplayBuffer` 接收采样数据 (由 `Runner` 传递)。
        *   使用 `eval_mlp` 和 `target_mlp` 计算每个智能体的当前 Q 值和目标 Q 值。
        *   使用 `eval_vdn_net` 和 `target_vdn_net` 将各智能体的 Q 值合并为总 Q 值 (`q_total_eval` 和 `q_total_target`)。
        *   计算 TD Target: (y = r + \gamma Q'_{tot}(\mathbf{s}', \mathbf{a}'))。
        *   计算损失函数 (例如，TD Target 和 (Q_{tot}) 之间的均方误差)。
        *   执行反向传播和参数更新。
        *   定期更新目标网络参数。
    *   动作选择的Q值计算：虽然实际的 (\epsilon)-greedy 等选择逻辑在 `agent/agent_UAV.py` 中，但获取 Q 值的计算 (`self.eval_mlp(inputs)`) 是由这个策略类执行的。

*   **`common/arguments.py`**: 使用 `argparse` 存储和管理项目的所有超参数。
    *   提供 `get_common_args()` 函数获取通用参数。
    *   提供针对不同算法的特定参数设置函数 (如 `get_mixer_args()` 用于 VDN)。
    *   这些参数包括学习率、折扣因子 (\gamma)、网络结构参数、epsilon-greedy 参数、缓冲区大小、训练步数、评估周期等。

*   **`common/replay_buffer.py`**: 实现经验回放缓冲区 (`ReplayBuffer` 类等)。
    *   定义存储结构，用于存放智能体与环境交互产生的经验元组，如：(联合状态 `s`, 各智能体观测 `o`, 各智能体动作 `u` (及one-hot `u_onehot`), 联合奖励 `r`, 下一联合状态 `s_next`, 下一各智能体观测 `o_next`, 是否终止 `terminated`, 填充位 `padded`)。
    *   提供 `store_episode` 方法，将 `RolloutWorker` 产生的整个 episode 数据存入缓冲区。
    *   提供 `sample` 方法，供 `Runner` 调用以从中随机采样一批经验数据用于训练。

*   **`common/rollout.py`**: 实现 `RolloutWorker` 类，作为辅助模块，用于执行一个完整的 rollout (即一个 episode 的交互序列)。
    *   在 `generate_episode` 方法中：
        *   与环境 (`Multi_UAV_env.py`) 交互。
        *   调用 `agent/agent_UAV.py` 中 `Agents` 类的 `choose_action` 方法为每个智能体获取动作。
        *   收集 (o, s, u, r, terminated, ...) 等经验数据。
        *   处理 episode 结束和数据填充。
    *   供 `runner.py` 调用以生成训练或评估所需的经验数据。
    *   包含 `write_log` 方法，在评估过程中，将详细的交互信息（如动作、冲突、碰撞、GUE位置等）写入日志文件。这些日志文件通常保存在动态创建的 `./log/...` 目录下。

*   **`log/`**: 此目录**不是项目预设的静态目录**。它通常是在 `common/rollout.py` 中的 `write_log` 函数执行评估任务时，根据需要**动态创建**的。用于存放训练过程中的评估日志、性能指标、每个评估 episode 的详细交互步骤等。具体的子目录结构可能还会包含算法名称和评估轮次等信息。

## 3. 环境

这是一个多智能体强化学习环境，模拟了多个无人机 (AVs - Aerial Vehicles) 在共享无线信道环境下的操作。无人机需要从初始点到达目标点，并选择合适的信道进行通信。智能体需要学习移动策略和信道选择策略以最大化累积奖励。

---

#### 1. 环境参数 (Environment Parameters)

这些参数定义了模拟的基础：

*   **智能体数量 (Agent Numbers)**:
    *   `self.n_a_agents = 3`: 空中智能体 (无人机) 数量。
*   **网络与通信 (Network & Communication)**:
    *   `self.n_bs = 1`: 基站数量。
    *   `self.n_channel = 2`: 通信信道数量 (不区分上下行，无人机从中选择用于通信的信道)。
    *   `self.sinr_AV = 4.5` (dB): 无人机通信的最小SINR阈值。对应的线性值为 `self.sinr_AV_real = 10**(self.sinr_AV / 10) approx 2.818`。
    *   `self.pw_AV = 30` (dBm): 无人机发射功率。线性值为 `self.pw_AV_real = (10**(self.pw_AV / 10)) / 1000` W。
    *   `self.pw_BS = 40` (dBm): 基站发射功率。线性值为 `self.pw_BS_real = (10**(self.pw_BS / 10)) / 1000` W。
    *   `self.g_main = 10` (线性值): 天线主瓣增益 (对应 10 dB)。
    *   `self.g_side = 1` (线性值): 天线旁瓣增益 (对应 0 dB)。
    *   `self.N0 = -120` (dBm): 噪声功率。代码中转换为线性瓦特 `self.N0_real = (10**(self.N0 / 10)) / 1000` W。此值代表总噪声功率，在SINR计算中直接使用，无需额外乘以带宽。
    *   `self.reference_AV = -60` (dB): 无人机到基站的参考路径损耗。其线性值为 `self.reference_AV_real = 10**(self.reference_AV / 10) = 10**(-6)`。
    *   **路径损耗模型**: \\(PL(d) = K \\cdot d^{\\alpha}\\)，其中：
        *   \\(K = self.reference_{AV\_real} = 10^{-6}\\)。
        *   \\(d\\) 是无人机与基站（或另一无人机）之间的三维欧氏距离。
        *   路径损耗指数 \\(\\alpha = self.env\\_PL\\_exp = 3\\) (环境参数，固定为3)。
    *   **物理环境与移动 (Physical Environment & Movement)**:
        *   `self.height_BS = 15` (m): 基站高度。
        *   `self.height_AV = 150` (m): 无人机飞行高度。
        *   `self.v_max = 50` (m/s): 无人机最大速度。
        *   `self.delta_t = 2` (s): 每个时间步长。
        *   `self.n_speed = 2`: 速度等级数量。对应的速度值 `self.vel_actions = np.array([0, 50])` (m/s)。
        *   `self.BS_locations = np.array([[500, 600]], dtype=float)`: 基站位置 (x,y坐标)。
        *   `self.init_location = np.array([[-250.0, 400.0], [-30.33, 930.33], [-30.33, -130.33]], dtype=float)`: 3个无人机的初始位置。
        *   `self.dest_location = np.array([[1250.0, 400.0], [1030.33, -130.33], [1030.33, 930.33]], dtype=float)`: 3个无人机的目标位置。
        *   无人机轨迹参数 (`self.trajectory`): 包含每架无人机的预定义轨迹信息，主要用于 `delta_location` 函数中计算移动：
            *   `self.trajectory[i, 0]`: 轨迹的斜率 `k`。
            *   `self.trajectory[i, 3]`: 无人机初始位置的 y 坐标 (`self.init_location[i,1]`)。`delta_location` 中用于与目标y坐标比较，以确定y轴方向。
            *   `self.trajectory[i, 5]`: 无人机目标位置的 y 坐标 (`self.dest_location[i,1]`)。`delta_location` 中用于与初始y坐标比较，以确定y轴方向，并在 `is_arrived_cal` 中用于判断是否到达。
        *   `self.max_dist = 1500` (m): 用于归一化的最大距离。
        *   `self.check_num = 5`: 在一个移动步内检查冲突和碰撞的中间点数量 (固定为5)。
        *   **碰撞检测距离阈值**: 10 米。无人机之间的距离小于此阈值时，判定为发生碰撞。
*   **学习与奖励 (Learning & Rewards)**:
    *   `self.episode_limit = 50`: 每个 episode 的最大步数 (可以调整)。
    *   `self.arrive_reward = 10`: 单个无人机到达奖励 (可以调整)。
    *   `self.all_arrived_reward = 2000`: 所有无人机到达的额外奖励 (可以调整)。
    *   `self.conflict_reward = -100`: 发生通信干扰的惩罚 (可以调整)。
    *   `self.collision_reward = -100`: 发生碰撞的惩罚 (可调整)。环境设计确保碰撞和通信冲突是独立评估并分别施加相应奖励/惩罚的。

---

#### 2. 状态空间 (State Space)

环境定义了全局状态和每个无人机的个体观测。

*   **全局状态 (Global State)** (维度会根据新参数调整):
    *   主要构成:
        *   `arrive_state`: `self.n_a_agents` (3) 个元素，表示每个无人机是否到达目标点。
        *   `onehot_bs_association`: `self.n_a_agents * self.n_bs` (3 * 1 = 3) 个元素，表示无人机关联的基站。在当前单基站配置下，此部分状态的实际信息承载是固定的（所有无人机关联唯一的基站）。其设计旨在保持状态向量结构的一致性，并支持未来平滑扩展至多基站场景。
        *   `location`: `self.n_a_agents * 2` (3 * 2 = 6) 个元素，表示每个无人机的二维坐标 (x,y)。
    *   具体实现细节及完整维度请参考 `get_state_size()` 和 `get_state()` 函数。

*   **个体观测 (Agent Observation)** (维度会根据新参数调整):
    每个无人机的观测向量。其构成如下 (N_AV = `self.n_a_agents` = 3, N_BS = `self.n_bs` = 1):
    1.  **到达状态/剩余距离指示 (1 element)**: 表明当前无人机是否已到达其目标点，或相关的距离信息。
    2.  **智能体类型符号 (1 element)**: 始终为代表无人机的值。
    3.  **其他无人体的信息 ( (N_AV-1) * (N_BS + 2) elements )**: 对于其他每个无人机 `al_id`:
        *   `bs_association`: (N_BS) 个元素，`al_id` 所关联基站的编码。
        *   `dist_to_al_id_bs`: 1 个元素，到 `al_id` 关联基站的距离。
        *   `safe_dist_related_to_al_id`: 1 个元素，与 `al_id` 相关的安全距离。
    4.  **其他无人机的到达状态 (N_AV-1 elements)**.
    5.  **当前无人机ID (N_AV elements)**: one-hot 编码。
    *   具体实现需要参考原 `get_obs_agent()` 和 `get_obs_size()` 并适配新参数。

---

#### 3. 动作空间 (Action Space)

每个无人机有一个离散的动作空间。
`self.n_actions = self.n_speed * self.n_channel` (例如, 对于2个速度等级和2个信道，则为 `2 * 2 = 4` 个动作)。

一个动作 `a` (例如，整数 0-3) 被解码为：

1.  **速度等级 (Velocity Level)**:
    *   `vel_level_idx = a // self.n_channel` (例如, `a // 2`)
    *   从 `self.vel_actions` (例如, `[0, 50]`) 中选择速度。
2.  **信道选择 (Channel Selection)**:
    *   `channel_idx = a % self.n_channel` (例如, `a % 2`)
    *   从 `self.n_channel` (例如, 2) 个可用信道中选择 (索引 0 至 `self.n_channel - 1`)。此选择的信道将用于所有相关的通信活动评估。

---

#### 4. 环境动态 (Environment Dynamics)

描述环境如何根据无人机的动作进行演化。

*   **重置 (`reset`)**:
    1.  `self.episode_step` 重置为 0。
    2.  无人机的 `self.current_location` 重置为 `self.init_location`。
    3.  无人机的 `self.is_arrived` 重置为未到达。
    4.  `self.recorded_arrive` 清空。

*   **步进 (`step`)**:
    *   在 `Multi_UAV_env_multi_level_practical_multicheck` 类中，`step` 函数的定义是 `step(self, actions)`。
    *   在 `Multi_UAV_env_multi_level_practical_multicheck_padded` 类中，`step` 函数定义为 `step(self, actions, step)`。传入的第二个参数 `step` (代表当前 episode 步数) 在当前函数实现中未被使用；函数内部依赖 `self.episode_step` 来获取当前步数信息。
    1.  **无人机移动**: 对于每个未到达的无人机 `i`:
        *   解码动作 `a = actions[i]` (包含每个智能体单个整数动作的列表/数组)，得到速度选择 `vel_level_idx` 和信道选择 `channel_idx`。
        *   如果 `self.is_arrived[i] == 1`，则位移 `delta_loc = np.array([0,0])` (已到达则不移动)。
        *   否则，调用 `delta_location(vel_level_idx, i)` 计算位移 `delta_loc`。该函数内部逻辑如下：
            1.  `vel = self.vel_actions[vel_level_idx]`
            2.  `moving_dist = vel * self.delta_t`
            3.  获取无人机 `i` 的轨迹参数: 斜率 `slope_param = self.trajectory[i, 0]`，初始y坐标 `init_y = self.trajectory[i, 3]`，目标y坐标 `dest_y = self.trajectory[i, 5]`。
            4.  根据 `slope_param` 和 y 轴移动方向 (`dest_y - init_y`)，结合三角函数 (`math.cos(math.atan(abs(slope_param)))`, `math.sin(math.atan(abs(slope_param)))`)，将 `moving_dist` 分解为 `delta_x` 和 `delta_y`。
            5.  `delta_loc = np.array([delta_x, delta_y])`。
        *   更新无人机位置: `self.current_location[i] += delta_loc`。
        *   **到达处理**：计算无人机当前位置与目标点的欧氏距离 `current_dist`。如果 `current_dist < 10` (米) 并且之前未到达，则判定该无人机到达，其位置被强制设为目标点 `self.current_location[i] = self.dest_location[i]`。
        *   生成路径上的 `self.check_num` (即5个) 中间点 `inter_locations`，用于后续的冲突和碰撞检测。
            `inter_locations[i, k, :] = (self.current_location[i] - delta_loc) + k * (delta_loc / (self.check_num -1 ))` (此公式通过线性插值计算无人机 `i` 在当前时间步移动路径上的 `self.check_num` 个等间隔检查点。这些点从该无人机移动前的位置 (`self.current_location[i] - delta_loc`) 开始，到移动后的新位置 (`self.current_location[i]`) 结束。 `k` 的取值范围是 `0` 到 `self.check_num - 1`。)
    2.  `self.episode_step` 增加 1。
    3.  **碰撞检测**: 调用 `self.is_collision = self.check_action_collision(inter_locations)`。
        *   `check_action_collision` 逻辑：该函数负责检测无人机之间的碰撞。它会遍历所有无人机对，在它们各自路径上的 `inter_locations` (中间检查点序列) 检查它们之间的三维欧氏距离。如果任意一对无人机在对应的中间检查点之间的距离小于碰撞检测距离阈值 (10米)，则判定为发生碰撞，并相应地更新 `self.is_collision` 状态数组中对应无人机的标志位。
    4.  **奖励计算**: 调用 `self._reward(actions)`。`_reward` 函数直接使用从主 `actions` 参数中解码出的信道选择信息进行通信质量评估。
    5.  **到达状态更新**: 在 `_reward` 中或 `step` 中（当前主要在 `step` 中，第329-338行）通过判断 `current_dist < 10` 更新 `self.is_arrived`。
    6.  返回 `reward`, `terminated`, `info`。

---

#### 5. 奖励函数 (Reward Function - `_reward`)

引导无人机完成任务并优化通信。`_reward` 函数直接使用从主 `actions` 参数中解码出的信道选择信息来评估通信质量。

*   **基础奖励**: 初始化 `reward = 0`。

*   **到达奖励 (Arrival Rewards)**:
    *   `self.all_arrived_reward`: 所有无人机到达时给予。
    *   `self.arrive_reward`: 每个无人机首次到达时给予。

*   **冲突惩罚 (Conflict Penalties)**: `self.conflict_reward`
    *   对每个活跃 (未到达且速度非零) 的无人机，评估其在其选择的**单一信道 `channel_idx`** 上的通信质量 (SINR)。
    *   SINR的评估会考虑该无人机是作为发射方（例如，向基站发送数据）还是接收方（例如，从基站接收数据）。冲突惩罚基于此SINR是否低于 `self.sinr_AV_real`。
    *   **SINR 计算公式**:
        \\[ SINR = \\frac{P_{received\_signal\_linear}}{ (\\sum P_{interfering\_signals\_linear}) + N_{0\_linear} } \\]
        *   \\(N_{0\_linear} = self.N0_{real}\\) (总噪声功率，线性瓦特)。
    *   **期望信号功率 \\(P_{received\_signal\_linear}\\)** (模型假设发射方和接收方天线主瓣对准):
        *   当UAV作为发射方 (例如，到BS): \\(P_{rx\_dest} = self.pw_{AV\_real} \\cdot (self.g_{main}^2) / PL_{UAV\_BS}\\) (注意：此为理论公式。在代码实现中，天线增益 `g_main` 和 `g_side` 的影响已整合到路径损耗参考值 `self.reference_AV_real` 或直接计入信号功率计算逻辑中。因此，在将此公式与代码对应时，需注意其实际计算方式。)
        *   当UAV作为接收方 (例如，从BS): \\(P_{rx\_dest} = self.pw_{BS\_real} \\cdot (self.g_{main}^2) / PL_{BS\_UAV}\\) (注意：此为理论公式。在代码实现中，天线增益 `g_main` 和 `g_side` 的影响已整合到路径损耗参考值 `self.reference_AV_real` 或直接计入信号功率计算逻辑中。因此，在将此公式与代码对应时，需注意其实际计算方式。)
        *   路径损耗 \\(PL(d) = self.reference_{AV\_real} \\cdot d^{self.env\_PL\_exp}\\)，其中 \\(d\\) 为3D距离， \\(self.reference_{AV\_real} = 10^{-6}\\)，路径损耗指数 \\(self.env\_PL\_exp = 3\\)。
    *   **干扰信号功率 \\(\\sum P_{interfering\_signals\_linear}\\)** (线性累加所有在所选信道 `channel_idx` 上的干扰源功率):
        *   **当UAV作为发射方 (目标接收端为BS)**: 干扰源定义为其他UAV在同一信道 `channel_idx` 上向此BS或其他BS的发射。
            单个干扰UAV到目标BS的接收功率: \\(P_{interf} = self.pw_{AV\_real} \\cdot self.g_{main} \\cdot self.g_{side} / PL_{interfUAV\_BS}\\) (此处的路径损耗 \\(PL_{interfUAV\_BS}\\) 指干扰UAV到目标BS的路径损耗；模型假设干扰信号通过一个主瓣和一个旁瓣的天线增益组合进行传播)。
        *   **当UAV作为接收方 (目标发射端为BS)**: 干扰源定义为：
            a.  其他UAV在同一信道 `channel_idx` 上的发射。
                单个干扰UAV到目标UAV的接收功率: \\(P_{interf\_UAV} = self.pw_{AV\_real} \\cdot self.g_{main} \\cdot self.g_{side} / PL_{interfUAV\_targetUAV}\\) (模型假设干扰信号通过一个主瓣和一个旁瓣的天线增益组合进行传播)。
            b.  BS向其他同信道 `channel_idx` UAV的发射，其信号泄漏到此目标UAV。
                单个此类干扰路径的接收功率: \\(P_{interf\_BS} = self.pw_{BS\_real} \\cdot self.g_{main} \\cdot self.g_{side} / PL_{BS\_targetUAV}\\) (模型假设干扰信号通过一个主瓣和一个旁瓣的天线增益组合进行传播)。
    *   如果计算出的 `sinr_real < self.sinr_AV_real` (其线性值为 `10**(4.5/10) approx 2.818`)，则施加 `self.conflict_reward`。

*   **碰撞惩罚 (Collision Penalty)**: `self.collision_reward = -100`
    *   如果无人机 `i` 的 `self.is_collision[i]` 状态 (在 `step` 函数中通过 `check_action_collision` 更新) 为 `True`，表明该无人机发生了碰撞，则会受到此惩罚。该机制旨在促使智能体学习避免互相过于接近。

*   **移动惩罚 (Movement Penalty)**:
    *   每个未到达的无人机每步受到一个小额负奖励 (例如 `reward -= 1`)。

---

**关于信道选择的说明**：
在此项目的设计中，每个无人机在其动作中选择一个**速度等级**以及一个**单一的通信信道** (`channel_idx`)。此单一选择的信道将用于所有相关的通信活动评估，**不区分上行或下行链路的固有属性**。

该信道选择直接影响奖励，主要通过冲突惩罚机制：
-   当无人机需要在其选择的信道 `channel_idx` 上进行通信（无论是作为发射方还是接收方）时，会评估其SINR。
-   如果由于其他无人机或基站在同一信道 `channel_idx` 上的活动导致SINR低于阈值 (`self.sinr_AV_real`)，则会受到惩罚。

通过引入基于SINR模型的冲突惩罚，智能体被激励去学习协同地选择一个既能满足自身通信需求，又能最小化对其他智能体干扰（反之亦然）的信道。这促使无人机避免使用过于拥挤的信道，从而优化整体的通信性能。已到达目标或选择速度为0的无人机在此模型中不被视为会产生干扰或评估SINR。
这种采用单一信道选择并据此评估双向通信质量的机制是当前设计的核心。

## 6. 项目任务分解 (Task Directory)

为了更好地管理和推进项目，我们将整个开发过程分解为以下主要任务模块和子任务。每个任务模块对应项目中的一个或多个核心组件。

### 任务模块一：强化学习环境 (`Multi_UAV_env.py`)
*   **目标**：构建稳定、准确的多无人机飞行与通信仿真环境。
*   **子任务**：
    1.  **环境参数定义与初始化**:
        *   [ ] 实现 `__init__` 方法，加载所有物理参数、智能体参数、网络参数等。
        *   [ ] 验证参数的正确性和单位一致性。
    2.  **状态空间与观测空间实现**:
        *   [ ] 实现 `get_state()` 函数，生成全局状态。
        *   [ ] 实现 `get_obs_agent()` 和 `get_obs()` 函数，生成每个智能体的局部观测。
        *   [ ] 实现 `get_state_size()` 和 `get_obs_size()` 以提供维度信息。
    3.  **核心动态逻辑 - `step()` 函数**:
        *   [ ] 实现动作解码，将整数动作映射到速度和信道选择。
        *   [ ] 实现无人机移动逻辑 (`delta_location`)，包括基于轨迹的位移计算。
        *   [ ] 实现到达目标点的判断与处理逻辑 (强制位置更新)。
        *   [ ] 实现路径中间检查点的生成逻辑。
    4.  **碰撞检测机制**:
        *   [ ] 实现 `check_action_collision()` 函数，检测无人机之间的碰撞。
    5.  **奖励函数 (`_reward`) 设计与实现**:
        *   [ ] 实现到达奖励 (单个及全部)。
        *   [ ] 实现冲突惩罚，包括：
            *   [ ] SINR 计算逻辑 (区分UAV为发射方/接收方)。
            *   [ ] 期望信号功率计算。
            *   [ ] 干扰信号功率计算。
            *   [ ] 根据SINR阈值施加惩罚。
        *   [ ] 实现碰撞惩罚。
        *   [ ] 实现移动惩罚 (或时间惩罚)。
    6.  **环境重置 (`reset`)**:
        *   [ ] 实现 `reset()` 函数，将环境恢复到初始状态。
    7.  **环境信息接口**:
        *   [ ] 实现 `get_env_info()` 函数，提供必要的环境配置信息。
    8.  **单元测试与调试**:
        *   [ ] 针对移动、奖励、碰撞、状态观测等关键部分编写测试用例。
        *   [ ] 进行环境的整体交互测试。

### 任务模块二：智能体 (`agent/agent_UAV.py`)
*   **目标**：实现智能体的统一管理接口，连接策略与环境交互。
*   **子任务**：
    1.  **`Agents` 类实现**:
        *   [ ] 初始化，根据参数加载并持有具体的策略对象 (如 `VDN` 策略)。
        *   [ ] 实现 `choose_action()` 方法，供 `RolloutWorker` 调用，基于局部观测和策略网络选择动作 (包括epsilon-greedy探索)。
        *   [ ] 实现 `train()` 方法，供 `Runner` 调用，触发策略网络的学习更新。
    2.  **单元测试**:
        *   [ ] 测试动作选择逻辑。
        *   [ ] 测试训练接口的调用。

### 任务模块三：神经网络 (`network/`)
*   **目标**：构建VDN算法所需的神经网络组件。
*   **子任务**：
    1.  **基础Q网络 (`network/base_net.py`)**:
        *   [ ] 实现 `MLP` 类。
        *   [ ] (可选，根据需要) 实现 `RNN` (如 `GRUCell`) 类。
        *   [ ] (可选，根据需要) 实现 `D3QN` 类或其他高级Q网络结构。
    2.  **VDN混合网络 (`network/vdn_net.py`)**:
        *   [ ] 实现 `VDNNet` 类，执行各智能体Q值的加和。
    3.  **单元测试**:
        *   [ ] 测试各网络模块的前向传播。

### 任务模块四：策略 (`policy/vdn_UAV.py`)
*   **目标**：实现VDN算法的核心逻辑。
*   **子任务**：
    1.  **`VDN` 类实现**:
        *   [ ] 初始化：
            *   [ ] 创建评估Q网络 (`eval_mlp`) 和目标Q网络 (`target_mlp`) 实例。
            *   [ ] 创建VDN混合网络 (`eval_vdn_net` 和 `target_vdn_net`) 实例。
            *   [ ] 初始化优化器。
        *   [ ] 实现 `learn()` 方法:
            *   [ ] 从经验回放中获取数据。
            *   [ ] 计算当前Q值和目标Q值。
            *   [ ] 计算TD Target。
            *   [ ] 计算损失函数 (MSE)。
            *   [ ] 执行反向传播和参数更新。
            *   [ ] 实现目标网络的定期更新。
        *   [ ] 提供Q值计算接口，供 `Agents` 类调用。
    2.  **单元测试**:
        *   [ ] 测试 `learn` 方法的计算流程。
        *   [ ] 验证损失计算和参数更新。

### 任务模块五：通用组件 (`common/`)
*   **目标**：开发训练流程中可复用的辅助模块。
*   **子任务**：
    1.  **参数管理 (`common/arguments.py`)**:
        *   [ ] 定义并提供获取通用参数和算法特定参数的函数。
        *   [ ] 确保所有超参数均可通过命令行配置。
    2.  **经验回放缓冲区 (`common/replay_buffer.py`)**:
        *   [ ] 实现 `ReplayBuffer` 类，包括数据存储结构。
        *   [ ] 实现 `store_episode()` 方法，存储完整episode数据。
        *   [ ] 实现 `sample()` 方法，随机采样批次数据。
        *   [ ] 单元测试缓冲区存取功能。
    3.  **Rollout Worker (`common/rollout.py`)**:
        *   [ ] 实现 `RolloutWorker` 类。
        *   [ ] 实现 `generate_episode()` 方法：
            *   [ ] 与环境交互循环。
            *   [ ] 调用 `Agents.choose_action()` 获取动作。
            *   [ ] 收集经验元组。
            *   [ ] 处理episode结束和数据填充。
        *   [ ] 实现 `write_log()` 方法，用于评估时记录详细交互日志。
        *   [ ] 单元测试episode生成和日志记录。

### 任务模块六：主程序与训练执行器 (`main.py`, `runner.py`)
*   **目标**：整合所有模块，实现完整的训练和评估流程。
*   **子任务**：
    1.  **`main.py`**:
        *   [ ] 实现命令行参数解析。
        *   [ ] 初始化环境、`Runner`。
        *   [ ] 根据参数调用 `runner.run()` 或 `runner.evaluate()`。
    2.  **`runner.py` (`Runner` 类)**:
        *   [ ] 初始化 `Agents`, `RolloutWorker`, `ReplayBuffer`。
        *   [ ] 实现 `run()` 方法 (主训练循环):
            *   [ ] 调用 `RolloutWorker` 生成数据。
            *   [ ] 将数据存入 `ReplayBuffer`。
            *   [ ] 从 `ReplayBuffer` 采样。
            *   [ ] 调用 `Agents.train()` 更新策略。
            *   [ ] 定期评估并保存模型/日志。
        *   [ ] 实现 `evaluate()` 方法，执行评估流程并记录结果。
    3.  **集成测试**:
        *   [ ] 测试完整的训练流程。
        *   [ ] 测试评估流程。

### 任务模块七：集成、测试与调优
*   **目标**：确保整个系统按预期工作，并对性能进行优化。
*   **子任务**：
    1.  [ ] **端到端测试**: 运行完整的训练和评估流程，检查是否存在错误和不一致。
    2.  [ ] **性能调优**: 根据训练曲线和评估结果，调整超参数（学习率、折扣因子、网络结构、epsilon参数等）。
    3.  [ ] **结果分析**: 分析训练日志和评估指标，理解智能体的学习行为。
    4.  [ ] **(可选) 结果可视化**: 开发或使用工具可视化无人机轨迹、奖励变化等。

### 任务模块八：文档与报告
*   **目标**：记录项目细节和成果。
*   **子任务**：
    1.  [ ] **完善 `DETAILED_DESIGN.md`**: 根据开发过程中的实际情况更新设计文档。
    2.  [ ] **撰写最终项目报告**: 总结项目目标、方法、过程、结果和遇到的挑战。
