#!/bin/bash
#SBATCH --job-name=test_deletes_2
#SBATCH --output=logs/%j_test_deletes_2.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=irani_run.q 

# Load Conda environment
echo "Activating conda environment"
eval "$(conda shell.bash hook)"
conda activate amit-env

# Create log directory if it doesn't exist
mkdir -p logs

# Choose a name for this run (can be anything)
RUN_NAME="test_deletes_2"

# Launch the script with required arguments
echo "Running python script"
srun python -u stroke_experimet_CLI.py \
    --run "$RUN_NAME" \
    --image_type shared \
    --images_indices 29 51 65 69 99 \
    --modify_roi \
    --create_montage \
    --steps_to_do 1 2 4 \
    # --roi_to_process EBA \
    # --save_path

# Creating image indices

# images_indices = np.sort(np.array([1, 5, 6, 9, 12, 13, 14, 22, 28, 27, 66, 119, 39, 109, 56, 44, 69]))
# images_indices = [69,109,119]

# images_indices = np.sort(np.array([1, 5, 6, 9, 12, 13, 14, 22, 28, 27, 66, 69, 39, 109, 56, 44])) # For excluded
# images_indices = np.sort(np.array([1, 4, 7, 9, 15, 16, 18, 20, 21, 29, 51, 65, 69, 96, 99])) # For shared