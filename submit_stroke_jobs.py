#!/usr/bin/env python3
import argparse
import math
import shlex
import subprocess
from pathlib import Path
from itertools import accumulate

# --- Use env's Python directly (no conda activate needed) ---
PYTHON_EXE = "/home/matanyaw/miniconda3/envs/amit-env/bin/python"

PROJECT_ROOT = Path("/home/matanyaw/DIP_decoder")
CLI_PATH = PROJECT_ROOT / "stroke_experimet_CLI.py"
LOG_DIR = PROJECT_ROOT / "logs"
# ROI_COVERAGE_DIR = '/home/matanyaw/DIP_decoder/data/roi_coverages'
ROI_COVERAGE_DIR = '/home/matanyaw/DIP_decoder/data/one_hemi_roi_coverages'



# Presets
face_images = [16, 18, 20, 21, 88, 116, 118, 124, 158, 135, 188, 218, 254, 275, 364, 366]
no_face_images = [3, 5, 6, 8, 10, 17, 30, 33, 37, 60, 61, 63, 80, 81, 82, 271, 280, 338, 403, 404]
jonathans_images = [1, 4, 7, 9, 15, 16, 18, 20, 21, 29, 51, 65, 69, 96, 99]

def build_cli_args(
    run: str,
    roi_cov_dir: str,
    image_type: str,
    images_indices,
    create_montage: bool,
    steps_to_do,
    roi_to_process=None,
    save_path=None,
):
    """Turn Python values into CLI flags for stroke_experimet_CLI.py."""
    args = [
        "--run", run,
        "--roi_cov_dir", roi_cov_dir, 
        "--image_type", image_type,
        "--steps_to_do", *map(str, steps_to_do),
        "--images_indices", *map(str, images_indices),
    ]
    if create_montage:
        args.append("--create_montage")
    if roi_to_process:
        args += ["--roi_to_process", *roi_to_process]
    if save_path:
        args += ["--save_path", save_path]
    return args

def submit_job(
    job_name: str,
    partition: str,
    gpus: int,
    cpus_per_task: int,
    mem: str,
    hours: int,
    ntasks: int,
    run: str,
    roi_cov_dir: str,
    image_type: str,
    images_indices,
    steps_to_do,
    create_montage: bool,
    roi_to_process=None,
    save_path=None,
    dry_run: bool = False,
):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_pattern = str(LOG_DIR / f"%j_{job_name}.out")

    cli_args = build_cli_args(
        run=run,
        roi_cov_dir=roi_cov_dir,
        image_type=image_type,
        images_indices=images_indices,
        create_montage=create_montage,
        steps_to_do=steps_to_do,
        roi_to_process=roi_to_process,
        save_path=save_path,
    )
    run_cmd = [PYTHON_EXE, "-u", str(CLI_PATH), *cli_args]
    wrap_str = shlex.join(run_cmd)

    sbatch_cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--output={out_pattern}",
        f"--ntasks={ntasks}",
        f"--cpus-per-task={cpus_per_task}",
        f"--mem={mem}",
        f"--gres=gpu:{gpus}",
        f"--time={hours:02d}:00:00",
        f"--partition={partition}",
        "--wrap", wrap_str,
    ]

    print("Submitting with:")
    print(" ", " ".join(shlex.quote(x) for x in sbatch_cmd))

    if dry_run:
        print("[dry-run] Not submitting.")
        return None, out_pattern

    res = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("sbatch failed:")
        print(res.stderr)
        raise SystemExit(res.returncode)

    job_id = res.stdout.strip()
    log_file = out_pattern.replace("%j", job_id)
    print(f"Submitted JobID: {job_id}")
    print(f"Log: {log_file}")
    return job_id, log_file

def parse_args():
    p = argparse.ArgumentParser(description="Submit stroke experiment job(s) via Slurm (no bash editing).")

    # Meta (how to split image list)
    p.add_argument("--imgs", choices=["faces", "no_faces", "jonathans", "all"], default="all",
                   help="Preset image set to use if --images_indices not provided.")
    p.add_argument("--njobs", type=int, default=None, help="Number of jobs to split across (ignored if --nimgs set).")
    p.add_argument("--nimgs", type=int, default=4, help="Images per job (overrides --njobs).")

    # Slurm resources
    p.add_argument("--job_name", default="stroke_run", help="Base job name; chunk index is appended.")
    p.add_argument("--partition", default="irani_run.q")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--cpus_per_task", type=int, default=8)
    p.add_argument("--mem", default="80G")
    p.add_argument("--hours", type=int, default=8)
    p.add_argument("--ntasks", type=int, default=1)

    # CLI params to your script
    p.add_argument("--run", type=str, default="running_script", help="Base run name; chunk index is appended.")
    p.add_argument("--roi_cov_dir", type=str, default=ROI_COVERAGE_DIR, help="Directory with inferred ROI coverages stored")
    p.add_argument("--image_type", default="shared")
    p.add_argument("--images_indices", nargs="+", type=int, default=None,
                   help="Explicit indices override --imgs preset.")
    p.add_argument("--steps_to_do", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--create_montage", action="store_true", default=True)
    p.add_argument("--no-create_montage", dest="create_montage", action="store_false")
    p.add_argument("--roi_to_process", nargs="+")
    p.add_argument("--save_path")

    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()

def choose_images(args):
    if args.images_indices is not None and len(args.images_indices) > 0:
        return list(dict.fromkeys(args.images_indices))  # dedupe, keep order

    if args.imgs == "faces":
        return face_images[:]
    if args.imgs == "no_faces":
        return no_face_images[:]
    if args.imgs == "jonathans":
        return jonathans_images[:]
    # "all"
    return list(dict.fromkeys(face_images + no_face_images + jonathans_images))


def chunck_images(args, images):
    n = len(images)
    if n == 0:
        print("Total images: 0 | imgs/job: 0 | jobs: 0")
        return
    if getattr(args, "njobs", None) and args.njobs > 0:
        njobs = min(args.njobs, n)  # don't create more jobs than images
        base, extra = divmod(n, njobs)
        sizes = [base + (1 if i < extra else 0) for i in range(njobs)]

        # Mode 2: maximum images per job
    elif getattr(args, "nimgs", None) and args.nimgs > 0:
        imgs_per_job = max(1, args.nimgs)
        njobs = math.ceil(n / imgs_per_job)
        # All full-size chunks except possibly the last
        sizes = [imgs_per_job] * (njobs - 1) + [n - imgs_per_job * (njobs - 1)]

    # Mode 3: single job
    else:
        njobs = 1
        sizes = [n]
        # Compute chunk boundaries
    starts = [0] + list(accumulate(sizes))[:-1]
    ends = list(accumulate(sizes))

    return starts, ends, sizes


def main():
    args = parse_args()
    images = choose_images(args)
    if not images:
        raise SystemExit("No images selected. Provide --images_indices or a valid --imgs preset.")

    # Decide chunking

    if args.njobs:
        imgs_per_job = max(1, math.ceil(len(images) / max(1, args.njobs)))
    elif args.nimgs and args.nimgs > 0:
        imgs_per_job = args.nimgs
    else:
        # ceil to avoid zero images per job
        imgs_per_job = max(1, math.ceil(len(images) / max(1, args.njobs)))

    num_jobs = math.ceil(len(images) / imgs_per_job)

    starts, ends, sizes = chunck_images(args, images)
    print(f"Total images: {len(images)}  |  imgs/job: {sum(sizes)/len(sizes):.2f}  |  jobs: {len(starts)}")

    for j in range(num_jobs):
        start = starts[j]
        end = ends[j]
        chunk = images[start:end]

        job_name = f"{args.job_name}_{j+1}of{num_jobs}"
        run_name = args.run


        print(f"\n--> Submitting chunk {j+1}/{num_jobs}: indices[{start}:{end}] = {chunk}")

        submit_job(
            job_name=job_name,
            partition=args.partition,
            gpus=args.gpus,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            hours=args.hours,
            ntasks=args.ntasks,
            run=run_name,
            roi_cov_dir=args.roi_cov_dir,
            image_type=args.image_type,
            images_indices=chunk,
            steps_to_do=args.steps_to_do,
            create_montage=args.create_montage,
            roi_to_process=args.roi_to_process,
            save_path=args.save_path,
            dry_run=args.dry_run,
        )

        # MANIFEST = PROJECT_ROOT / "logs" / f"submit_manifest_{args.run}.csv"
        # first_write = not MANIFEST.exists()
        # with MANIFEST.open("a", newline="") as f:
        #     w = csv.writer(f)
        #     if first_write:
        #         w.writerow(["job_id","job_name","run","chunk_idx","total_chunks","images"])
        #     w.writerow([job_id, job_name, run_name, j+1, num_jobs, " ".join(map(str, chunk))])
        # print(f"Manifest updated: {MANIFEST}")


if __name__ == "__main__":
    main()
