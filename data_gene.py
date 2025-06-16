import re
import json
import os

def parse_log_file(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()

    data = []
    bs_pattern = re.compile(r"BS Locations:\s*(\[\[.*?\]\])")
    step_pattern = re.compile(r"^\s*\d+\s*\|\s*\[([^\]]+)\]")

    current_bs = None
    current_actions = []

    for line in lines:
        bs_match = bs_pattern.search(line)
        if bs_match:
            if current_bs is not None and current_actions:
                output_str = ", ".join([f"[{a.strip()}]" for a in current_actions])
                data.append({
                    "input": f"BS Locations: {current_bs}",
                    "output": output_str
                })
                current_actions = []
            current_bs = bs_match.group(1)

        step_match = step_pattern.search(line)
        if step_match:
            action = step_match.group(1)
            current_actions.append(action)

    if current_bs is not None and current_actions:
        output_str = ", ".join([f"[{a.strip()}]" for a in current_actions])
        data.append({
            "input": f"BS Locations: {current_bs}",
            "output": output_str
        })
    
    return data

# 示例调用
if __name__ == "__main__":
    log_folder_path = "for_fine_turing"      # TODO: 修改为你的log文件夹路径
    output_path = "converted_dataset.jsonl"
    
    all_episodes_data = []
    processed_files_count = 0

    if not os.path.isdir(log_folder_path):
        print(f"❌ Error: Log folder not found at {log_folder_path}")
    else:
        for filename in os.listdir(log_folder_path):
            if filename.endswith(".log"):
                file_path = os.path.join(log_folder_path, filename)
                print(f"📄 Processing {file_path}...")
                try:
                    episode_data = parse_log_file(file_path)
                    all_episodes_data.extend(episode_data)
                    processed_files_count += 1
                    print(f"  extracted {len(episode_data)} episodes.")
                except Exception as e:
                    print(f"  ❌ Error processing {file_path}: {e}")

        if all_episodes_data:
            # 写入 JSONL 文件
            with open(output_path, 'w') as f_out:
                for item in all_episodes_data:
                    f_out.write(json.dumps(item) + "\n")
            print(f"✅ Finished! {len(all_episodes_data)} total episodes from {processed_files_count} log files written to {output_path}")
        else:
            print(f"🤷 No log files processed or no data extracted from log files in {log_folder_path}")
