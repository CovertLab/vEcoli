import os
import argparse
import subprocess
import tempfile
import time
import random


def configure_slurm_workers(
    num_workers: int,
    cores_per_worker: int,
    ram_per_worker_mb: int,
    partition: str,
    idle_timeout: int,
    server_dir: str,
) -> None:
    """
    Configure and submit user-defined Slurm workers for HyperQueue.

    Args:
        num_workers: Number of worker nodes to start
        cores_per_worker: CPU cores per worker
        ram_per_worker_mb: RAM per worker in megabytes
        partition: Slurm partition(s) to use
        idle_timeout: Idle timeout for workers in minutes
        server_dir: HyperQueue server directory
    """
    # Make sure server directory exists
    assert os.path.exists(server_dir)

    # Create the worker submission script
    script_content = f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cores_per_worker}
#SBATCH --mem={ram_per_worker_mb}MB
#SBATCH --time=24:00:00
#SBATCH --output={server_dir}/hq-worker-%j.out
#SBATCH --error={server_dir}/hq-worker-%j.err
#SBATCH --signal=B:SIGUSR1@90
#SBATCH --requeue

_resubmit() {{
    # Resubmit job if we reach time limit
    echo "$(date): job $SLURM_JOBID received SIGUSR1, re-submitting"
    sbatch $0
}}
trap _resubmit SIGUSR1

# Start HyperQueue worker with specified options
hq worker start --manager slurm \\
    --server-dir {server_dir} \\
    --cpus {cores_per_worker} \\
    --resource "mem={ram_per_worker_mb}" \\
    --idle-timeout {idle_timeout}m &
wait $!
"""

    # Write script to temporary file
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".sh"
    ) as temp_script:
        temp_script.write(script_content)
        script_file = temp_script.name

    subprocess.run(["cat", script_file], check=True)  # Print script for debugging

    # Submit workers
    print(f"Submitting {num_workers} HyperQueue workers to Slurm...")

    max_retries = 5
    submitted_workers = 0

    for i in range(num_workers):
        worker_submitted = False
        attempt = 0

        while not worker_submitted and attempt < max_retries:
            try:
                # Exponential backoff with jitter to avoid thundering herd
                if attempt > 0:
                    backoff_time = min(60, (2**attempt) + random.uniform(0, 1))
                    print(
                        f"Retry attempt {attempt} for worker {i + 1}, waiting {backoff_time:.2f}s"
                    )
                    time.sleep(backoff_time)

                result = subprocess.run(
                    ["sbatch", script_file],
                    capture_output=True,
                    text=True,
                    check=True,  # Raise exception on error
                )

                job_id = result.stdout.strip().split()[-1]
                print(f"Submitted worker {i + 1}/{num_workers}: job ID {job_id}")
                worker_submitted = True
                submitted_workers += 1

            except subprocess.CalledProcessError as e:
                attempt += 1
                if attempt >= max_retries:
                    print(
                        f"Failed to submit worker {i + 1}/{num_workers} after {max_retries} attempts"
                    )
                    print(f"Final error: {e.stderr}")
                else:
                    print(f"Attempt {attempt} failed: {e.stderr}")

    # Clean up
    os.unlink(script_file)
    print(
        f"Submission complete: {submitted_workers}/{num_workers} workers submitted successfully."
    )


def main():
    parser = argparse.ArgumentParser(description="Submit HyperQueue workers to Slurm")
    parser.add_argument(
        "--num-workers", type=int, required=True, help="Number of worker nodes to start"
    )
    parser.add_argument(
        "--cores-per-worker", type=int, required=True, help="CPU cores per worker"
    )
    parser.add_argument(
        "--ram-per-worker-mb",
        type=int,
        required=True,
        help="RAM per worker in megabytes",
    )
    parser.add_argument(
        "--partition", type=str, required=True, help="Slurm partition(s) to use"
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        required=True,
        help="Idle timeout for workers in minutes",
    )
    parser.add_argument(
        "--server-dir", type=str, required=True, help="HyperQueue server directory"
    )

    args = parser.parse_args()

    configure_slurm_workers(
        num_workers=args.num_workers,
        cores_per_worker=args.cores_per_worker,
        ram_per_worker_mb=args.ram_per_worker_mb,
        partition=args.partition,
        idle_timeout=args.idle_timeout,
        server_dir=args.server_dir,
    )


if __name__ == "__main__":
    main()
