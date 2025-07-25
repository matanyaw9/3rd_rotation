import subprocess
import os

# List of image indices you want to process
# image_indices = [1, 4, 7, 9, 15]  # Change as needed
image_indices = [1]  # Change as needed
steps_to_do = [1, 2]  # Steps to perform
run = 11

# Common Slurm parameters
partition = "irani.q"
mem = "64G"
cpus = 8
gpus = 1
time = "08:00:00"
save_dir = f'/home/matanyaw/DIP_decoder/results_12_07_25/run_{run}'

# Notification URL
notify_url = "https://ntfy.sh/complete_stroke_experiment_predicted_fMRI_ROIs"

# Submit one job per image index
for idx in image_indices:
    # save_path = f"{save_dir}/run_{idx}"
    save_path = save_dir
    python_cmd = (
        f"python /home/matanyaw/DIP_decoder/complete_stroke_experiment_predicted_fMRI_ROIs.py "
        f"--images_indices {idx} --steps_to_do {steps_to_do} --save_path {save_path} && "
        f"curl -d '✅ Finished image {idx} run {run}' {notify_url}"
    )
    logs_dir = f"{save_dir}/logs"
    os.makedirs(logs_dir, exist_ok=True)
    sbatch_cmd = [
        "sbatch",
        "--job-name", f"run{run}_img_{idx}",
        "--output", f"{logs_dir}/run{run}_img_{idx}.out",
        # "--error", f"{logs_dir}/run{run}_img_{idx}.err",
        "--partition", partition,
        "--mem", mem,
        "--cpus-per-task", str(cpus),
        "--gres", f"gpu:{gpus}",
        "--time", time,
        "--wrap", python_cmd
    ]

    print("Submitting job:", " ".join(sbatch_cmd))
    subprocess.run(sbatch_cmd)
