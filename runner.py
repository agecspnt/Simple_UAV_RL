import numpy as np
import os
from common.replay_buffer import ReplayBuffer
from common.rollout import RolloutWorker
from agent.agent_UAV import Agents # Assuming agent_UAV.py contains Agents class
import torch # For saving/loading models
from tqdm import tqdm # Import tqdm
import collections # For deque
import subprocess # <-- Add this for calling the visualizer script
import time # For rollout worker log filename
import matplotlib.pyplot as plt # For convergence curve plotting directly in runner or passed to visualizer
import visualizer # Import the visualizer module to call its functions
from io import StringIO # <<< Ensure this is imported for pstats stream
import pstats # <<< Ensure pstats is imported for profiling

class Runner:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        # Initialize Agents
        self.agents = Agents(args) # Agents class will internally create the policy (e.g., VDN)

        # Initialize Replay Buffer
        self.buffer = ReplayBuffer(args)

        # Initialize Rollout Worker
        # The RolloutWorker needs the environment, agents, and args
        self.rolloutWorker = RolloutWorker(self.env, self.agents, self.args)

        # Logging and Saving
        self.log_dir = args.log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # For epsilon annealing (used by RolloutWorker perhaps, or directly in agent action selection)
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon

        # Ensure model directory exists (also created in main.py, but good to have here for robustness if Runner is used standalone)
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        # Data for convergence curve
        self.convergence_timesteps = []
        self.convergence_avg_rewards = []
        self.convergence_avg_losses = []

        print(f"Runner initialized with algorithm: {args.alg}")
        print(f"Number of agents: {args.n_agents}, Number of actions: {args.n_actions}")
        print(f"Episode limit: {args.episode_limit}, Total training steps: {args.n_steps}")
        print(f"Buffer size: {args.buffer_size}, Batch size: {args.batch_size}")
        print(f"Epsilon: init={self.epsilon:.2f}, min={self.min_epsilon:.2f}, anneal_steps={args.anneal_steps}")
        print(f"Target update cycle: {args.target_update_cycle}, Evaluate cycle: {args.evaluate_cycle}, Save cycle: {args.save_cycle}")
        print(f"Device: {args.device}")

    def run(self):
        """
        Main training loop.
        """
        print("\nStarting training run...")
        time_steps = 0
        episode_num = 0
        last_evaluate_T = 0
        last_save_T = 0

        # For logging recent training performance
        train_rewards = collections.deque(maxlen=100) # Store rewards of last 100 training episodes
        train_episode_lengths = collections.deque(maxlen=100)
        train_losses = collections.deque(maxlen=1000) # Store last 1000 training losses

        first_eval_done_for_profiling = False # <<< New flag for profiling control

        with tqdm(total=self.args.n_steps, desc="Training Progress", mininterval=self.args.tqdm_mininterval, unit="step") as pbar:
            while time_steps < self.args.n_steps:
                episode_num += 1
                current_epsilon_for_rollout = max(self.min_epsilon, self.epsilon)
                
                episode_batch, episode_stats, _ = self.rolloutWorker.generate_episode(
                    episode_num=episode_num, 
                    epsilon=current_epsilon_for_rollout,
                    evaluate=False
                )
                episode_reward = episode_stats["episode_reward"]
                episode_len = episode_stats["episode_length"]
                
                train_rewards.append(episode_reward)
                train_episode_lengths.append(episode_len)
                self.buffer.store_episode(episode_batch)
                
                num_train_steps_this_episode = 0
                if self.buffer.current_size >= self.args.batch_size:
                    for _ in range(episode_len): # Train once per collected step in the episode
                        batch = self.buffer.sample(self.args.batch_size)
                        loss = self.agents.train(batch, time_steps + num_train_steps_this_episode)
                        train_losses.append(loss)
                        num_train_steps_this_episode +=1
                
                pbar.update(episode_len)
                time_steps += episode_len
                
                if self.epsilon > self.min_epsilon:
                    if self.args.anneal_steps > 0:
                         self.epsilon = self.args.epsilon - (self.args.epsilon - self.min_epsilon) * (min(time_steps, self.args.anneal_steps) / self.args.anneal_steps)
                    else: 
                        self.epsilon = self.min_epsilon
                
                # Log descriptive stats to tqdm
                log_stats = {
                    "Episode": episode_num,
                    "Epsilon": f"{current_epsilon_for_rollout:.3f}",
                    "Avg Reward (Tr)~": f"{np.mean(train_rewards):.2f}" if train_rewards else "N/A",
                    "Avg Length (Tr)~": f"{np.mean(train_episode_lengths):.2f}" if train_episode_lengths else "N/A",
                    "Avg Loss~": f"{np.mean(train_losses):.4f}" if train_losses else "N/A",
                    "Buffer": f"{self.buffer.current_size}/{self.buffer.size}"
                }
                pbar.set_postfix(log_stats)

                if (time_steps - last_evaluate_T) >= self.args.evaluate_cycle:
                    pbar.set_description_str(f"Running evaluation at T={time_steps}")
                    
                    if train_rewards: 
                        self.convergence_timesteps.append(time_steps)
                        self.convergence_avg_rewards.append(np.mean(train_rewards))
                    if train_losses: 
                        self.convergence_avg_losses.append(np.mean(train_losses)) 
                    else: 
                        if self.convergence_timesteps and len(self.convergence_avg_losses) < len(self.convergence_timesteps):
                             self.convergence_avg_losses.append(np.nan) 

                    eval_start_time = time.time() 
                    self.evaluate(episode_num, time_steps)
                    eval_duration = time.time() - eval_start_time 
                    pbar.write(f"Evaluation at T={time_steps} (Episode {episode_num}) finished in {eval_duration:.2f} seconds.") 
                    
                    last_evaluate_T = time_steps
                    pbar.set_description_str("Training Progress") 

                    # Profiling logic after first evaluation
                    if not first_eval_done_for_profiling and hasattr(self.args, 'runner_should_print_profile') and self.args.runner_should_print_profile:
                        if hasattr(self.args, 'profiler_instance'):
                            pbar.write("Disabling profiler and printing results after first evaluation...")
                            self.args.profiler_instance.disable() 
                            
                            s_io_runner = StringIO()
                            ps_runner = pstats.Stats(self.args.profiler_instance, stream=s_io_runner).sort_stats('cumulative')
                            # Explicitly print stats to the stream before getting value
                            ps_runner.print_stats(100) # Print top 100, or choose another number
                            profile_output_runner = s_io_runner.getvalue()

                            # --- Temporary Debugging --- (Optional: keep or remove after testing)
                            # pbar.write(f"DEBUG: Length of profile_output_runner: {len(profile_output_runner)}")
                            # if not profile_output_runner or profile_output_runner.isspace():
                            #    pbar.write("DEBUG: profile_output_runner is empty or whitespace.")
                            # elif "0 function calls" in profile_output_runner:
                            #    pbar.write("DEBUG: profile_output_runner indicates 0 function calls.")
                            # --- End Temporary Debugging ---

                            if profile_output_runner and not profile_output_runner.isspace() and "0 function calls" not in profile_output_runner:
                                pbar.write("\n" + "="*50 + "\nPROFILE RESULTS (After First Evaluation in Training):" + "="*50)
                                pbar.write(profile_output_runner)
                            else:
                                pbar.write("DEBUG: Profiler output was considered empty/trivial or had 0 calls. Raw output (if any) was:")
                                pbar.write(profile_output_runner) # Print it anyway to see content
                            
                            if hasattr(self.args, 'profiler_printed_by_runner_flag_ref') and 'profiler_printed_by_runner' in self.args.profiler_printed_by_runner_flag_ref:
                                self.args.profiler_printed_by_runner_flag_ref['profiler_printed_by_runner'] = True
                            
                            self.args.runner_should_print_profile = False # Prevent this block from running again
                        first_eval_done_for_profiling = True

                if (time_steps - last_save_T) >= self.args.save_cycle and time_steps > 0:
                    pbar.set_description_str(f"Saving model at T={time_steps}")
                    self.save_models(time_steps) 
                    last_save_T = time_steps
                    pbar.set_description_str("Training Progress") # Reset description
        
        print(f"\nTraining finished after {time_steps} timesteps and {episode_num} episodes.")
        
        # Fallback profiling print if training ends before first eval cycle that would trigger profiling print
        if hasattr(self.args, 'runner_should_print_profile') and self.args.runner_should_print_profile:
            if hasattr(self.args, 'profiler_instance'):
                self.args.profiler_instance.disable() 
                
                s_io_fallback = StringIO()
                ps_fallback = pstats.Stats(self.args.profiler_instance, stream=s_io_fallback).sort_stats('cumulative')
                ps_fallback.print_stats(100) # Explicitly print to stream
                profile_output_fallback = s_io_fallback.getvalue()

                # --- Temporary Debugging --- (Optional)
                # print(f"DEBUG: Fallback - Length of profile_output_fallback: {len(profile_output_fallback)}")
                # if not profile_output_fallback or profile_output_fallback.isspace():
                #    print("DEBUG: Fallback - profile_output_fallback is empty or whitespace.")
                # elif "0 function calls" in profile_output_fallback:
                #    print("DEBUG: Fallback - profile_output_fallback indicates 0 function calls.")
                # --- End Temporary Debugging ---

                if profile_output_fallback and not profile_output_fallback.isspace() and "0 function calls" not in profile_output_fallback:
                    print("\n" + "="*50 + "\nPROFILE RESULTS (Training Ended Before First Eval Triggered Profile Print):" + "="*50)
                    print(profile_output_fallback)
                else:
                    print("DEBUG: Fallback profiler output was considered empty/trivial or had 0 calls. Raw output (if any) was:")
                    print(profile_output_fallback)
                
                if hasattr(self.args, 'profiler_printed_by_runner_flag_ref') and 'profiler_printed_by_runner' in self.args.profiler_printed_by_runner_flag_ref:
                     self.args.profiler_printed_by_runner_flag_ref['profiler_printed_by_runner'] = True
                self.args.runner_should_print_profile = False # Mark as handled

        self.evaluate(episode_num, time_steps) 
        self.save_models(time_steps) 

    def evaluate(self, current_episode_num=0, current_time_steps=0):
        """
        Evaluates the learned policy.
        """
        print(f"\nEvaluating model... Training Episode: {current_episode_num}, Timesteps: {current_time_steps}")
        total_rewards = []
        total_steps = []

        # The self.args.log_dir is now the run-specific timestamped directory set in main.py.
        # Visualizations will go into a 'visualizations' subfolder within this run-specific log_dir.
        visualization_save_dir = os.path.join(self.args.log_dir, "visualizations")
        if not os.path.exists(visualization_save_dir):
            os.makedirs(visualization_save_dir)

        first_log_file_path = None

        # Detailed evaluation logs (ep0) will be saved directly into self.args.log_dir (the run-specific timestamped folder).
        # No need to create another timestamped sub-folder here for each evaluation cycle.
        # The log_output_dir passed to generate_episode will be self.args.log_dir.
        run_specific_log_dir = self.args.log_dir 
        if not run_specific_log_dir : # Should not happen if main.py sets it up
            print("Warning: self.args.log_dir is not set. Cannot save detailed episode logs.")
            # Optionally, disable logging or use a default path

        for eval_ep_num in tqdm(range(self.args.evaluate_episodes), desc="Evaluation Episodes", leave=False, mininterval=self.args.tqdm_mininterval):
            log_episode_identifier = f"eval_T{current_time_steps}_ep{eval_ep_num}"
            
            episode_data, episode_stats, log_file_path_this_ep = self.rolloutWorker.generate_episode(
                episode_num=log_episode_identifier, 
                evaluate=True, 
                epsilon=0.0,
                log_output_dir=run_specific_log_dir # Pass the run-specific (already timestamped) log_dir
            )
            total_rewards.append(episode_stats["episode_reward"])
            total_steps.append(episode_stats["episode_length"])

            if eval_ep_num == 0 and log_file_path_this_ep:
                first_log_file_path = log_file_path_this_ep

        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        avg_steps = np.mean(total_steps)
        
        print(f"Evaluation Results (over {self.args.evaluate_episodes} episodes):")
        print(f"  Avg Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
        print(f"  Avg Steps: {avg_steps:.2f}")
        print("-"*40)
        
        # --- Visualize Trajectories (GIF) ---
        if first_log_file_path and hasattr(self.args, 'visualize_latest_eval') and self.args.visualize_latest_eval:
            print(f"Attempting to visualize trajectories: {first_log_file_path}")
            try:
                visualizer_args_list = ["python", "visualizer.py", first_log_file_path]
                # Default to saving GIF as per user request.
                # The visualizer.py itself now handles --save_dir for GIF.
                # Runner controls where that save_dir is.
                # runner.py ensures visualization_save_dir is passed if self.args.save_visualization_plot is True (which it is by default)
                
                # If --save_visualization_plot is true (default), pass the save_dir
                if hasattr(self.args, 'save_visualization_plot') and self.args.save_visualization_plot:
                    visualizer_args_list.extend(["--save_dir", visualization_save_dir])
                # else: visualizer will show the GIF (if imageio supports it, or save to current dir by default)

                subprocess.run(visualizer_args_list, check=True)
                # print(f"Trajectory visualization generated. If saving, check: {visualization_save_dir}")
            except subprocess.CalledProcessError as e:
                print(f"Error running visualizer.py for trajectories: {e}")
            except FileNotFoundError:
                print(f"Error: visualizer.py not found or python not in PATH.")
        
        # --- Plot and Save Convergence Curve ---
        if hasattr(self.args, 'visualize_latest_eval') and self.args.visualize_latest_eval: # Reuse same flag or add a new one
            convergence_plot_path = os.path.join(visualization_save_dir, "convergence_curve.png")
            print(f"Attempting to generate convergence curve: {convergence_plot_path}")
            try:
                # Ensure that convergence_avg_losses has the same length as convergence_timesteps if it's plotted
                # This handles the case where losses might not be recorded if buffer isn't full at a collection point
                current_losses_len = len(self.convergence_avg_losses)
                expected_len = len(self.convergence_timesteps)
                if current_losses_len < expected_len:
                    # Pad with NaNs or a sensible placeholder for plotting
                    padded_losses = list(self.convergence_avg_losses) + [np.nan] * (expected_len - current_losses_len)
                else:
                    padded_losses = self.convergence_avg_losses

                if self.convergence_timesteps: # Only plot if there's data
                    visualizer.plot_convergence_curve(
                        self.convergence_timesteps,
                        self.convergence_avg_rewards, # Rewards should align with timesteps
                        padded_losses, # Use padded losses
                        convergence_plot_path
                    )
                    print(f"Convergence curve saved to {convergence_plot_path}")
                else:
                    print("No data yet to plot convergence curve.")

            except Exception as e:
                print(f"Error generating convergence curve: {e}")

        # TODO: Log these stats to a central logger (e.g., TensorBoard, WandB, or CSV file)

    def save_models(self, identifier): # identifier could be episode_count or time_steps
        model_path = os.path.join(self.args.model_dir, str(identifier))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # print(f"Saving models to {model_path}") # tqdm might make this noisy
        self.agents.save_model(model_path, identifier) 

    def load_models(self, path):
        print(f"Loading models from {path}")
        self.agents.load_model(path)
        self.epsilon = self.min_epsilon 
        # Or self.epsilon = 0.0 if strictly greedy evaluation is desired.