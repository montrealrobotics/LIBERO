import h5py
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import argparse


def extract_videos_from_hdf5(file_path, output_dir):
    """
    Extract image sequences from HDF5 files and save them as videos.
    
    Args:
        file_path (str): Path to the HDF5 file
        output_dir (str): Directory to save the extracted videos
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(file_path, 'r') as f:
        # Iterate through each trajectory in the file
        for traj_key in tqdm(f["data"].keys(), desc="Processing trajectories"):
            traj = f["data"][traj_key]
            
            # Get images from the trajectory
            if 'obs' in traj:
                images = traj["obs"]["agentview_rgb"][:]  # Shape should be (T, H, W, C)
                
                # Create video writer
                video_path = os.path.join(output_dir, f"{Path(file_path).stem}_{traj_key}.mp4")
                
                # Get first image to determine dimensions
                first_image = images[0]
                height, width = first_image.shape[:2]
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
                
                # Write each frame to video
                for frame in images:
                    # Convert to BGR if necessary (HDF5 might store in RGB)
                    if frame.shape[-1] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = frame[::-1, ::-1]
                    out.write(frame)
                
                out.release()

def main():
    parser = argparse.ArgumentParser(description='Extract videos from HDF5 dataset files.')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory containing the HDF5 dataset files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where the extracted videos will be saved')
    
    args = parser.parse_args()
    
    # Process all HDF5 files in the dataset directory
    for hdf5_file in Path(args.dataset_dir).glob("*.hdf5"):
        print(f"Processing file: {hdf5_file}")
        extract_videos_from_hdf5(str(hdf5_file), args.output_dir)

if __name__ == "__main__":
    main()