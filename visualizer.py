import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re
import imageio.v2 as imageio
import shutil

def parse_log_file(log_path):
    """
    Parses the log file to extract UAV trajectories, collisions, and other relevant info.
    Returns:
        A dictionary containing:
        - 'trajectories': List of lists, where each inner list contains (x,y) tuples for a UAV.
        - 'collisions': List of (step, agent_idx, x, y) tuples indicating collision events.
        - 'initial_locations': List of (x,y) for initial UAV positions.
        - 'dest_locations': List of (x,y) for destination UAV positions.
        - 'bs_locations': List of (x,y) for base station positions (if available in env or args).
        - 'episode_reward': Total reward for the episode.
        - 'episode_length': Total length of the episode.
    """
    trajectories = {}  # Dict to store trajectories, keyed by agent_id
    collision_events = [] # Store (step, agent_idx, x, y)
    
    initial_locations = []
    dest_locations = [] # These might need to be hardcoded or passed if not in log
    # For now, let's assume some defaults based on DETAILED_DESIGN.md if not found
    # Or better, extract from the first step's locations if they represent initial.

    bs_locations = [] # Initialize as empty, to be read from log or defaulted
    default_bs_location_if_not_in_log = [[0.0, 0.0]] # Default if not found in log

    episode_reward = 0
    episode_length = 0
    
    # Regex to parse locations: e.g., (x1,y1) or [x1,y1]
    # Original location_pattern for UAV locations:
    location_pattern = re.compile(r"\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)") # UAV locs like (x,y)
    # More general coordinate pattern for BS locations, allowing brackets or parentheses
    # And also allowing for integers or floats
    bs_coord_pattern = re.compile(r"\[(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*)\]") # BS locs like [x,y]


    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        header_parsed = False
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("BS Locations:"):
                try:
                    locations_str_part = line.split(":", 1)[1].strip()
                    parsed_bs_locs = []
                    for match in bs_coord_pattern.finditer(locations_str_part):
                        x, y = float(match.group(2)), float(match.group(3)) # Corrected group indices
                        parsed_bs_locs.append([x, y])
                    
                    if parsed_bs_locs:
                        bs_locations = parsed_bs_locs
                except Exception as e:
                    print(f"Could not parse BS Locations line: '{line}'. Error: {e}")
                continue

            if line.startswith("Total Reward:"):
                match_rew = re.search(r"Total Reward: (-?[\d\.]+)", line)
                if match_rew:
                    episode_reward = float(match_rew.group(1))
                match_len = re.search(r"Length: (\d+)", line)
                if match_len:
                    episode_length = int(match_len.group(1))
                continue

            if line.startswith("Step |"): # Header for step data
                header_parsed = True
                continue
            
            if not header_parsed or "|" not in line:
                continue

            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 5: # Step, Actions, Reward, Terminated, Locations, Collisions
                continue
            
            try:
                step = int(parts[0])
                locations_str = parts[4]
                collisions_str = parts[5] if len(parts) > 5 else "N/A"

                current_step_locations = []
                for match in location_pattern.finditer(locations_str):
                    x, y = float(match.group(1)), float(match.group(2)) # Corrected: UAV locations use group 1 and 2
                    current_step_locations.append((x, y))
                
                if not current_step_locations:
                    continue

                # Initialize trajectories and capture initial locations at step 0
                if step == 0:
                    initial_locations = list(current_step_locations)
                    for i in range(len(current_step_locations)):
                        trajectories[i] = []
                
                for i, loc in enumerate(current_step_locations):
                    if i in trajectories:
                        trajectories[i].append(loc)
                    else: # Should not happen if initialized at step 0
                        trajectories[i] = [loc]

                # Parse collisions
                # Example: "[C, -, C]" or "True" or "[True, False, True]"
                if collisions_str != "N/A":
                    # Check for the list format like [C, -, -]
                    collision_list_match = re.search(r"\[([C,\s,-]+)\]", collisions_str)
                    if collision_list_match:
                        collision_states = collision_list_match.group(1).split(',')
                        for agent_idx, state in enumerate(collision_states):
                            if 'C' in state.strip().upper() or 'TRUE' in state.strip().upper(): # 'C' for collision
                                if agent_idx < len(current_step_locations):
                                    collision_events.append((step, agent_idx, current_step_locations[agent_idx][0], current_step_locations[agent_idx][1]))
                    elif 'TRUE' in collisions_str.upper() or 'C' in collisions_str.upper(): # Simpler global collision flag, less ideal
                        # This case is harder to pinpoint to a specific UAV without more info
                        # For now, we might mark all UAVs or a generic point
                        pass


            except ValueError as e:
                print(f"Skipping malformed line: {line} - Error: {e}")
                continue
        
        # Infer destination locations from the last point of each trajectory if episode completed
        # Or use hardcoded values if available (more reliable)
        # From DETAILED_DESIGN.md
        # self.dest_location = np.array([[1250.0, 400.0], [1030.33, -130.33], [1030.33, 930.33]], dtype=float)
        dest_locations_hardcoded = [[1250.0, 400.0], [1030.33, -130.33], [1030.33, 930.33]]
        if not dest_locations and initial_locations: # Ensure we have a consistent number of agents
             dest_locations = dest_locations_hardcoded[:len(initial_locations)]

        # If bs_locations were not found in the log, use a default
        if not bs_locations:
            print(f"Warning: BS locations not found in log '{os.path.basename(log_path)}'. Using default: {default_bs_location_if_not_in_log}")
            bs_locations = default_bs_location_if_not_in_log

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return None
    except Exception as e:
        print(f"Error parsing log file {log_path}: {e}")
        return None

    return {
        'trajectories': [trajectories[i] for i in sorted(trajectories.keys()) if i in trajectories and trajectories[i]],
        'collisions': collision_events,
        'initial_locations': initial_locations,
        'dest_locations': dest_locations,
        'bs_locations': bs_locations,
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'log_file_name': os.path.basename(log_path)
    }

def plot_frame(frame_data, current_step, total_steps, episode_reward, log_file_name):
    """
    Plots a single frame for the GIF.
    frame_data contains trajectories up to current_step, and all static points.
    """
    plt.figure(figsize=(12, 10))
    num_agents = len(frame_data['trajectories_up_to_step'])
    color_map = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    colors = [color_map[i % len(color_map)] for i in range(num_agents)]

    all_x = []
    all_y = []

    # Plot trajectories up to current step
    for i, trajectory_segment in enumerate(frame_data['trajectories_up_to_step']):
        if not trajectory_segment: continue
        x_coords, y_coords = zip(*trajectory_segment)
        all_x.extend(x_coords)
        all_y.extend(y_coords)
        plt.plot(x_coords, y_coords, marker='.', linestyle='-', label=f'UAV {i+1}', color=colors[i])
        # Plot current position of UAV as a larger dot
        if x_coords and y_coords:
            plt.scatter(x_coords[-1], y_coords[-1], s=80, color=colors[i], edgecolor='black', zorder=6)

    # Plot initial locations
    if frame_data['initial_locations']:
        init_x, init_y = zip(*frame_data['initial_locations'])
        all_x.extend(init_x); all_y.extend(init_y)
        plt.scatter(init_x, init_y, s=100, c=[colors[i] for i in range(len(frame_data['initial_locations']))], marker='o', edgecolor='black', label='Start Points', zorder=5)

    # Plot destination locations
    if frame_data['dest_locations']:
        dest_x, dest_y = zip(*frame_data['dest_locations'])
        all_x.extend(dest_x); all_y.extend(dest_y)
        plt.scatter(dest_x, dest_y, s=100, c=[colors[i] for i in range(len(frame_data['dest_locations']))], marker='X', edgecolor='black', label='End Points', zorder=5)

    # Plot Base Station locations
    if frame_data['bs_locations']:
        bs_x, bs_y = zip(*frame_data['bs_locations'])
        all_x.extend(bs_x); all_y.extend(bs_y)
        plt.scatter(bs_x, bs_y, s=150, c='darkgrey', marker='s', edgecolor='black', label='Base Station', zorder=5)

    # Plot collision events up to current step
    if frame_data['collisions_up_to_step']:
        coll_x = [c[2] for c in frame_data['collisions_up_to_step']]
        coll_y = [c[3] for c in frame_data['collisions_up_to_step']]
        all_x.extend(coll_x); all_y.extend(coll_y)
        plt.scatter(coll_x, coll_y, s=80, c='red', marker='*', label='Collision', zorder=10)

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        # Fixed world view can be set here if desired, e.g. based on all initial/dest points or a predefined area.
        # For dynamic padding based on current frame content:
        x_padding = (max_x - min_x) * 0.1 if (max_x - min_x) > 0 else 10
        y_padding = (max_y - min_y) * 0.1 if (max_y - min_y) > 0 else 10
        plt.xlim(min_x - x_padding, max_x + x_padding)
        plt.ylim(min_y - y_padding, max_y + y_padding)
    
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.title(f"UAV Trajectories - {log_file_name}\nStep: {current_step + 1}/{total_steps} | Reward: {episode_reward:.2f}")
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside

def create_trajectory_gif(data, gif_save_path, frame_duration=0.2):
    """
    Creates a GIF from the parsed log data.
    """
    if not data or not data['trajectories']:
        print("No trajectory data to create GIF.")
        return

    episode_length = data['episode_length']
    full_trajectories = data['trajectories']
    all_collisions = data['collisions']
    
    temp_frame_dir = "./temp_gif_frames/"
    if os.path.exists(temp_frame_dir):
        shutil.rmtree(temp_frame_dir) # Clean up old frames
    os.makedirs(temp_frame_dir)

    frame_filenames = []

    # Determine overall bounds for a consistent view (optional, but can be good for GIFs)
    all_xs_overall = list(data['initial_locations'][i][0] for i in range(len(data['initial_locations'])))
    all_ys_overall = list(data['initial_locations'][i][1] for i in range(len(data['initial_locations'])))
    if data['dest_locations']:
        all_xs_overall.extend(data['dest_locations'][i][0] for i in range(len(data['dest_locations'])))
        all_ys_overall.extend(data['dest_locations'][i][1] for i in range(len(data['dest_locations'])))
    for traj in full_trajectories:
        if traj:
            all_xs_overall.extend([p[0] for p in traj])
            all_ys_overall.extend([p[1] for p in traj])
    
    overall_min_x, overall_max_x = min(all_xs_overall) if all_xs_overall else 0, max(all_xs_overall) if all_xs_overall else 1
    overall_min_y, overall_max_y = min(all_ys_overall) if all_ys_overall else 0, max(all_ys_overall) if all_ys_overall else 1
    x_padding_overall = (overall_max_x - overall_min_x) * 0.1 if (overall_max_x - overall_min_x) > 0 else 10
    y_padding_overall = (overall_max_y - overall_min_y) * 0.1 if (overall_max_y - overall_min_y) > 0 else 10
    fixed_xlim = (overall_min_x - x_padding_overall, overall_max_x + x_padding_overall)
    fixed_ylim = (overall_min_y - y_padding_overall, overall_max_y + y_padding_overall)

    print(f"Generating {episode_length} frames for GIF...")
    for step in range(episode_length):
        frame_plot_data = {
            'trajectories_up_to_step': [traj[:step+1] for traj in full_trajectories if traj],
            'initial_locations': data['initial_locations'],
            'dest_locations': data['dest_locations'],
            'bs_locations': data['bs_locations'],
            'collisions_up_to_step': [coll for coll in all_collisions if coll[0] <= step]
        }
        
        # Call plot_frame with fixed_xlim and fixed_ylim for consistent view
        plt.figure(figsize=(12, 10))
        num_agents = len(frame_plot_data['trajectories_up_to_step'])
        color_map = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        colors = [color_map[i % len(color_map)] for i in range(num_agents)]

        # Plot trajectories up to current step
        for i, trajectory_segment in enumerate(frame_plot_data['trajectories_up_to_step']):
            if not trajectory_segment: continue
            x_coords, y_coords = zip(*trajectory_segment)
            plt.plot(x_coords, y_coords, marker='.', linestyle='-', label=f'UAV {i+1}', color=colors[i])
            if x_coords and y_coords:
                plt.scatter(x_coords[-1], y_coords[-1], s=80, color=colors[i], edgecolor='black', zorder=6)

        if frame_plot_data['initial_locations']:
            init_x, init_y = zip(*frame_plot_data['initial_locations'])
            plt.scatter(init_x, init_y, s=100, c=[colors[i] for i in range(len(frame_plot_data['initial_locations']))], marker='o', edgecolor='black', label='Start', zorder=5)
        if frame_plot_data['dest_locations']:
            dest_x, dest_y = zip(*frame_plot_data['dest_locations'])
            plt.scatter(dest_x, dest_y, s=100, c=[colors[i] for i in range(len(frame_plot_data['dest_locations']))], marker='X', edgecolor='black', label='End', zorder=5)
        if frame_plot_data['bs_locations']:
            bs_x, bs_y = zip(*frame_plot_data['bs_locations'])
            plt.scatter(bs_x, bs_y, s=150, c='darkgrey', marker='s', edgecolor='black', label='BS', zorder=5)
        if frame_plot_data['collisions_up_to_step']:
            coll_x = [c[2] for c in frame_plot_data['collisions_up_to_step']]
            coll_y = [c[3] for c in frame_plot_data['collisions_up_to_step']]
            plt.scatter(coll_x, coll_y, s=80, c='red', marker='*', label='Collision', zorder=10)

        plt.xlim(fixed_xlim)
        plt.ylim(fixed_ylim)
        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.title(f"Trajectories - {data.get('log_file_name', 'Log')}\nStep: {step + 1}/{episode_length} | Reward: {data['episode_reward']:.2f}")
        # Create a concise legend if many UAVs, or place it carefully
        handles, labels = plt.gca().get_legend_handles_labels()
        # Remove duplicate labels for Start/End/BS/Collision if they appear per agent
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust layout further for external legend

        frame_filename = os.path.join(temp_frame_dir, f"frame_{step:04d}.png")
        plt.savefig(frame_filename)
        plt.close() # Close plot to free memory
        frame_filenames.append(frame_filename)

    # Create GIF
    print(f"Compiling GIF from {len(frame_filenames)} frames...")
    with imageio.get_writer(gif_save_path, mode='I', duration=frame_duration) as writer:
        for filename in frame_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    print(f"GIF saved to {gif_save_path}")

    # Clean up temporary frames
    shutil.rmtree(temp_frame_dir)
    print(f"Temporary frame directory {temp_frame_dir} removed.")

def plot_convergence_curve(timesteps, avg_rewards, avg_losses, save_path):
    """
    Plots the training convergence curves (average reward and loss over time) and saves it.
    Deletes the previous curve if it exists to ensure it's an update.
    """
    if not timesteps:
        print("No data provided for convergence curve.")
        return

    # Delete the old plot if it exists to simulate an update
    if os.path.exists(save_path):
        try:
            os.remove(save_path)
        except OSError as e:
            print(f"Error deleting old convergence plot {save_path}: {e}")
            # Continue to try plotting anyway

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Average Reward
    color = 'tab:blue'
    ax1.set_xlabel('Training Timesteps')
    ax1.set_ylabel('Average Reward (Smoothed)', color=color)
    ax1.plot(timesteps, avg_rewards, color=color, linestyle='-', marker='.', label='Avg Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Create a second y-axis for Average Loss if loss data is available
    if avg_losses and not all(np.isnan(loss) for loss in avg_losses): # Check if there are non-NaN losses
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Average Loss (Smoothed)', color=color)  # we already handled the x-label with ax1
        ax2.plot(timesteps, avg_losses, color=color, linestyle=':', marker='x', label='Avg Loss')
        ax2.tick_params(axis='y', labelcolor=color)
        # To make sure both y-axes are visible if they overlap significantly, you might need to adjust limits or scales manually.
    
    fig.suptitle('Training Convergence Over Time', fontsize=16)
    # Add legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    if avg_losses and not all(np.isnan(loss) for loss in avg_losses):
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')
    else:
        ax1.legend(lines, labels, loc='best')

    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    
    try:
        # Ensure the directory for save_path exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)
        plt.close(fig) # Close the figure to free memory
        # print(f"Convergence curve saved to {save_path}") # Runner already prints this
    except Exception as e:
        print(f"Error saving convergence plot to {save_path}: {e}")
        if fig:
            plt.close(fig) # Ensure figure is closed on error too

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize UAV trajectories from log files as a GIF.")
    parser.add_argument("log_file", type=str, help="Path to the evaluation log file.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the GIF. If None, saves in current dir.")
    parser.add_argument("--duration", type=float, default=0.2, help="Duration (in seconds) for each frame in the GIF.")
    
    args = parser.parse_args()

    log_data = parse_log_file(args.log_file)
    
    if log_data:
        gif_filename_base = os.path.splitext(os.path.basename(args.log_file))[0] + "_trajectory.gif"
        
        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            gif_save_path = os.path.join(args.save_dir, gif_filename_base)
        else:
            gif_save_path = gif_filename_base # Save in current directory if no save_dir
            
        create_trajectory_gif(log_data, gif_save_path, frame_duration=args.duration)