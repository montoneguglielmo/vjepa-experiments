import os
import csv
import random
from pathlib import Path

def create_video_index(dataset_path, output_file):
    # Get all video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = []
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))
    
    # Sort files for reproducibility
    video_files.sort()
    
    # Create CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(['path', 'label'])  # Header
        
        # Write each video with a random label
        for video_path in video_files:
            # Use absolute path
            abs_path = os.path.abspath(video_path)
            # Assign random label (0-999)
            label = random.randint(0, 999)
            writer.writerow([abs_path, label])
    
    print(f"Created video index with {len(video_files)} videos at {output_file}")

if __name__ == "__main__":
    dataset_path = "../datasets/drawing-agent"
    output_file = "configs/pretrain/drawing_agent_index.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    create_video_index(dataset_path, output_file) 