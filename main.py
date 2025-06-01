import argparse
import os
import torch
import numpy as np
from runner import Runner
from common.arguments import get_common_args, get_mixer_args
from Multi_UAV_env import Multi_UAV_env_multi_level_practical_multicheck

# <<< 开始：添加 cProfile 相关的导入 >>>
import cProfile
import pstats
from io import StringIO # 用于在内存中捕获输出，如果需要的话
import time # <<< Added import for time
# <<< 结束：添加 cProfile 相关的导入 >>>

# Global profiler instance so Runner can access it
profiler = cProfile.Profile()
profiler_printed_by_runner = False # New global flag to track if runner printed results

def main_function_to_profile(prof):
    global profiler_printed_by_runner # Allow modification of the global flag
    # Get arguments
    args = get_common_args()
    if args.alg.lower() == 'vdn': # VDN uses a specific mixer type
        args = get_mixer_args(args) 
    
    # Create a run-specific timestamped log directory
    if args.log_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # The original args.log_dir (e.g., './log') becomes the parent directory
        parent_log_dir = args.log_dir 
        args.log_dir = os.path.join(parent_log_dir, timestamp) # Update args.log_dir to the new timestamped path
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        print(f"All logs for this run will be saved in: {args.log_dir}")
    else:
        # Fallback if args.log_dir is not set by arguments, though get_common_args usually provides a default.
        # Create a timestamped log directory in the current working directory.
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.log_dir = os.path.join(".", timestamp + "_logs") # Example: ./20231027-160000_logs
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        print(f"Warning: Original args.log_dir not specified. All logs for this run will be saved in: {args.log_dir}")
    
    # Save all arguments (hyperparameters) to a file in the run-specific log directory
    hyperparameters_file_path = os.path.join(args.log_dir, "hyperparameters.txt")
    with open(hyperparameters_file_path, 'w') as f:
        f.write(f"Hyperparameters for run: {args.log_dir}\n")
        f.write("-"*40 + "\n")
        for arg_name, arg_value in sorted(vars(args).items()): # Sort for consistent order
            f.write(f"{arg_name}: {arg_value}\n")
    print(f"Hyperparameters saved to: {hyperparameters_file_path}")

    # Pass profiler and set flag for profiling after first eval (if training)
    args.profiler_instance = prof
    # This flag tells the runner it *should* try to print after first eval.
    args.runner_should_print_profile = not args.evaluate 
    # Pass a way for the runner to set the global flag in this module
    args.profiler_printed_by_runner_flag_ref = globals()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    # Create model directory if it doesn't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    # Initialize environment
    env = Multi_UAV_env_multi_level_practical_multicheck(args) 

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"] 

    # Initialize Runner
    runner = Runner(env, args) # Runner gets args with profiler_instance and flag

    # Run training or evaluation
    if args.evaluate:
        print("Starting evaluation...")
        prof.enable() # Enable specifically for this evaluation block
        runner.evaluate()
        prof.disable() # Disable after this evaluation block
        
        if prof.stats: # Check if profiler has stats before printing
            print("\n" + "="*50 + "\nPROFILE RESULTS (Evaluation):" + "="*50)
            s_io = StringIO()
            ps = pstats.Stats(prof, stream=s_io).sort_stats('cumulative')
            ps.print_stats(50)
            print(s_io.getvalue())
            profiler_printed_by_runner = True # Mark as printed to prevent finally block from re-printing
    else:
        print("Starting training...")
        # Profiler is enabled globally. Runner will disable if args.runner_should_print_profile is True.
        runner.run()
        # If runner.run() completes and didn't print (e.g., finished before first eval),
        # the finally block in __main__ will catch it and print if not already printed.

if __name__ == '__main__':
    profiler.enable() # Enable profiler at the very start
    try:
        main_function_to_profile(profiler)
    finally:
        profiler.disable() # Ensure profiler is always disabled at the end
        # Print results only if they haven't been printed by the runner or eval block.
        # After disable(), pstats.Stats can be created. If no data, it will be an empty report.
        if not profiler_printed_by_runner:
            # Create a pstats.Stats object. This doesn't error if profiler is empty.
            s_io = StringIO()
            ps = pstats.Stats(profiler, stream=s_io).sort_stats('cumulative')
            
            # Check if any stats were actually recorded by checking the length of the string output
            # or by inspecting ps.total_tt or ps.top_level_stats, though direct output check is simpler.
            output = s_io.getvalue()
            if output and not output.isspace() and "0 function calls" not in output: # Check if output is non-empty and not just whitespace or an empty report
                print("\n" + "="*50 + "\nPROFILE RESULTS (Main Fallback - End of Script):" + "="*50)
                print(output)
            # else: No relevant stats to print from fallback 