import os
import sys
import argparse
import time
from datetime import datetime, timedelta
import requests
import torch
import complete_stroke_experiment_predicted_fMRI_ROIs
sys.path.append('/home/matanyaw/DIP_decoder/voxel_embeddings_ROIs')
from ROI_actions import *


def argparse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the complete stroke experiment on ROIs.')

    parser.add_argument('--save_path', type=str, default=None,
                        help='Root path to save results.')
    parser.add_argument('--run', type=str, default=None, help='Name of the run for saving results.')
    parser.add_argument('--steps_to_do', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='List of steps to perform in the experiment.')
    parser.add_argument('--image_type', type=str, default='shared',
                        help='Type of images to use in the experiment (e.g., "shared", "excluded").')
    parser.add_argument('--images_indices', type=int, nargs='+', default=[1, 4],
                        help='List of image indices to process in the experiment.')
    parser.add_argument('--modify_roi', action='store_true',
                        help='Whether to modify the ROI during the experiment.')
    parser.add_argument('--save_throughout', action='store_true',
                        help='Whether to save intermediate results throughout the experiment.')
    parser.add_argument('--roi_to_process', type=str, nargs='+', default=None,
                        help='List of specific ROIs to process. If None, all predefined ROIs will be processed.')
    parser.add_argument('--create_montage', action='store_true',
                        help='Whether to create a montage of the results for each image.')

    return parser.parse_args()



def send_notification(message):
    """Send a notification using ntfy."""
    try:
        requests.post(
            "https://ntfy.sh/complete_stroke_experiment_predicted_fMRI_ROIs",
            data=message.encode("utf-8")
        )
    except Exception as e:
        print(f"Failed to send ntfy notification: {e}")

def small_script():
    a = 1 + 1
    a = a / 2
    print(f"Small script executed, result: {a}")
    return a

if __name__ == "__main__":

    start_time = time.time()
    args = argparse_args()
    current_date = datetime.now().strftime("%y_%m_%d")
    default_save_path = f'/home/matanyaw/DIP_decoder/matanya_results/results_{current_date}'
    if args.save_path is None:
        save_path = default_save_path
    else:
        save_path = args.save_path
    
    if args.run:
        save_path = os.path.join(save_path, f'run_{args.run}')
        os.makedirs(save_path, exist_ok=True)

    args_config = {
        'images_indices': args.images_indices,  # List of image indices to process in the experiment
        'save_path': save_path,  # Root path to save results
        'steps_to_do': args.steps_to_do,  # List of steps to perform in the experiment
        'image_type': args.image_type,  # Type of images to use in the experiment
        'modify_roi': args.modify_roi,  # Whether to modify the ROI during the experiment
        'run': args.run,  # Name of the run for saving results
        'save_throughout': args.save_throughout,
        'roi_to_process': args.roi_to_process,  # List of specific ROIs to process
        'create_montage': args.create_montage,  # Whether to create a montage of the results for each image
        
        # 'save_throughout': true,
        # 'n_saves': 5,
        # 'num_iter_no_stroke': 4001,
        # 'num_iter_stroke': 601,
        # 'learning_rate': 0.001,
    }

    # Just to see which GPU is being used
    if torch.cuda.is_available():
        dev_id = torch.cuda.current_device()
        print(f"Using CUDA device #{dev_id}: {torch.cuda.get_device_name(dev_id)}")
        print(f"Total GPUs available: {torch.cuda.device_count()}")
    else:
        print("No GPU detected!")

    try: 
        complete_stroke_experiment_predicted_fMRI_ROIs.run_experiment(args_config)
        # small_script()
        end_time = time.time()
        time_taken = timedelta(seconds=end_time - start_time)
        print(f'Total time taken: {time_taken}')
        send_notification(f"✅ Experiment completed successfully\nRun: {args_config['run']} Time: {time_taken}\n")
    except Exception as e:
        end_time = time.time()
        time_taken = timedelta(seconds=end_time - start_time)
        print(f'Total time taken: {time_taken}')
        send_notification(f"❌ Experiment failed\nRun: {args_config['run']} with error: {e}\n Time: {time_taken}\n")
        raise
    

# python complete_stroke_experiment_predicted_fMRI_ROIs.py --images_indices 1 4  --steps_to_do 1 2 3 --run 12
