# Initial Setup

## Request a Sherlock Account

If you’ve never had a Sherlock account: Go to <https://www.sherlock.stanford.edu/> and click on `Request an Account`

Note: Markus will have to approve this.

If you’ve had a Sherlock account for a previous group: Email srcc-support@stanford.edu and ask them to move your account to mcovert, and CC Markus on the email and in the email body ask for Markus to give approval

## Additional Resources: Sherlock Documentation from Stanford

https://srcc.stanford.edu/workshops/sherlock-boarding-session

https://www.sherlock.stanford.edu/docs/

—- STOP HERE AND WAIT FOR APPROVAL —-

# Login on to Sherlock

```bash
ssh <YOUR SU NET ID>@login.sherlock.stanford.edu
# Type in Stanford Password
# Do the Duo authentication
# The following setup steps should be done using the Sherlock terminal 
# NOTE that this is a LOGIN node, so no major computing should be done here

# It is best to use a compute node for things like cloning the repo, running code, resetting lpad, etc

srun -p mcovert --time=4:00:00 --cpus-per-task=2 --pty bash

# srun is the command for launching a job step under Slurm
# -p or --partition specifies which partition (queue) to use, choose covert :D
# --time: sets the job’s wall‑clock time limit
# --cpus-per-task specifies # CPU cores for each task in this job step
# --pty: allocates a pseudo‑terminal (TTY) to run an interactive session
# bash: launching a Bash shell
# When it finished, usually you can see your JOB ID in your shell

# You can use scancel to abort your job step
scancel <YOUR JOB ID>
```
You can also refer to the Sherlock Document: (https://www.sherlock.stanford.edu/docs/getting-started/connecting/)


# Clone the vEcoli repo to Sherlock

1. Git clone the vEcoli repo to your Sherlock account
```bash
git clone https://github.com/CovertLab/vEcoli.git
```

If you have already create your branch, you can use:
```bash
# View all the branch (including remote branch)
git branch -a

# Checkout to your own branch
git checkout <your_branch_name>

# Validate your current branch
git branch
```

2. Set up your `vEcoli` based on the detailed tutorial https://covertlab.github.io/vEcoli/hpc.html#sherlock

tips:
- You can use `nano` as text editor:
```bash
nano ~/.bash_profile
# After writing, you can use Ctrl+O to write out, Enter to confirm, and Ctrl+X to exit
# If you choose to use vim, press i for insert, and press Esc, then type :wq and Enter for writing out
```
- Before running the `python3` to set up the env, ensure you are in the vEcoli repo
- It usually takes time to run first job


# Hand in your job with python3

0. For the time we login Sherlock, first we can use `module load` to load crucial tools for experiments:

```bash
# Load newer Git, Java (for nextflow), and Python
module load system git java/21.0.4 python/3.12.1
# Include shared Nextflow and HyperQueue installations on PATH
export PATH=$PATH:$GROUP_HOME/vEcoli_env
```

and it only need to do once.

1. Before running your job, you should refer to the turtorial to construct the config for Sherlock
   https://covertlab.github.io/vEcoli/hpc.html#configuration

Notice:
Since `$HOME` only has a pretty small storage limit (run `sh_quota` to view), it is **highly recommended** to use `$SCRATCH` as your `emitter_arg` instead (like: "out_dir": "/scratch/users/<User_name>/out").

2. With configuration files, a workflow for vEcoli can be started with:
```bash
python3 runscripts/workflow.py --config <Your_config_file>
```
If `build_image` is true in your config JSON, the terminal will report that a **SLURM job** was submitted to build the container image. When the image build job starts, the terminal will report the build progress.

Notice:
- Remember to use `python3` instead of `python`
- This command is supposed to run on **login node**, which means there is no need to use `srun` to request a **compute node**
- If there is trouble with permission denied for nextflow (you can use `nextflow -version` to check out), you can try `chmod a+rwx`

3. Once the build has finished, the terminal will report that a **SLURM job** was submitted for the Nextflow workflow orchestrator before exiting back to the shell. At this point, you are free to close your connection, start additional workflows, etc. 
   
   Unlike workflows run locally, Sherlock’s containerized workflows mean any changes made to the repository after the container image has been built will not affect the running workflow.

4. You can use `squeue` to view the status of your job:
```bash
# View by job
squeue -j <Your_Job_ID>
# View by user
squeue -u <Your_user_name>
```

5. You can start additional, concurrent workflows that each build a new image with different modifications to the cloned repo. 
   By setting `build_image` to `false` and `container_image` to the path of previously saved image, you can save time by reusing a previously built image.

# Debug or Perform Analysis on Sherlock

It's recommended to use `Interactive Container` (you can view more details at next para. or in turtorial at https://covertlab.github.io/vEcoli/hpc.html#interactive-container).

0. First, you should have a basic `Image` for vEcoli. For example, we can use `test_sherlock.json` as config for both checking the project and building up a basic image:

```bash
# At head node:
python3 runscripts/workflow.py --config configs/test_sherlock.json
```

and by default, the image file is at `vEcoli/test_sherlock/test_image`.

1. Then, you can use:

```bash
runscripts/container/interactive.sh -i <Your_image_path> -a
```

to build an interactive image. 

Note: You can view this as a snap shoot for current vEcoli repo., and once it's built up, `~` and environment variables like `$SCRATCH` do not work inside the container, so you should treat them properly **before** building up the image. Also be careful that any changes that you make to /vEcoli inside the container are **discarded** when the container terminates.

2. Inside the image, you can just use `python` commands rather than `uv`. For example, you can do further analysis for your simulation results:

```bash
python3 runscripts/analysis.py --config <Your_config_file_path>
```

Moreover, if you want to exit the image, just use `exit` command.


# Interactive container and Non-Interactive Container

1. Start an interactive container with your full image path:

```bash
runscripts/container/interactive.sh -i <Your_image_path> -a
```

Note: In this way any changes that you make to `/vEcoli` inside the container are **discarded** when the container terminates, and `~` and any environment variables like `$SCRATCH` do not work inside the container. If you want to start an interactive container that **reflects the current state** of your cloned repository, navigate to your cloned repository and run the above command with the -d flag to start a “development” container:

```bash
runscripts/container/interactive.sh -i <Your_image_path> -a -d
```

In this mode, instead of editing source files in `/vEcoli`, you can directly edit the source files in your cloned repository and have those changes immediately reflected when running those scripts inside the container. 

Note: Any changes you make will **persist** after the container terminates and can be tracked using Git version control.

For more detailed information, please refer to the turtorial:
https://covertlab.github.io/vEcoli/hpc.html#interactive-container

2. To run any script inside a container with an non-interactive session, use the same command as **Interactive Container** but specify a command using the `-c `flag, for example:
```bash
runscripts/container/interactive.sh -i <Your_image_path> -c "python /vEcoli/runscripts/parca.py --config <Your_config_file>"
```

If you want to manually run scripts, you can use `sbatch`. 
- First, you should write your own Batch scripts:
  https://www.sherlock.stanford.edu/docs/getting-started/submitting/#batch-scripts

  And following is a sample for sbatch scripts:

```bash
#!/usr/bin/bash
#SBATCH --job-name=analysis_job
#SBATCH --output=analysis_job.%j.out
#SBATCH --error=analysis_job.%j.err
#SBATCH --time=20:00
#SBATCH --ntasks=1
#SBATCH --partition=owners,normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB

srun runscripts/container/interactive.sh -i /path/to/your_image.sif -a -c "python3 runscripts/analysis.py --config <Your_config_path>"
```

- second, use `sbatch` to submit the job:
```bash
sbatch <Your_Job_Name>.sh
```
and you can use `squeue` to check your job status. Moreover, you can list the contents of the output file with the following commands:
```bash
cat slurm-<Your_Job_ID>.out
```

# Download results to local
`SCP` is convenient for downloading files from the cluster. You can simply execute followings on your **local terminal**:

```bash
# -r for recursively deplicate the whole repo:
scp -r <Your_SU_ID>@login.sherlock.stanford.edu:/path/to/remote/folder  /path/to/local/destination

# If you only want to download single file:
scp  <Your_SU_ID>@login.sherlock.stanford.edu:/path/to/remote/file  /path/to/local/destination/
```

and it will require your password and Duo validation.

In practice, usually we want to get the analytical results for our simulation. Due to the report files are HTML files typically, we can turn to shell wildcard and use `rsync` with `include/exclude` filters

```bash
# Recursively downloads all .html files under the specific directory on Sherlock to your local machine while preserving the subdirectory structure:

rsync -av --prune-empty-dirs \
  --include='*/' --include='*.html' --exclude='*' \
  <Your_SU_ID>@login.sherlock.stanford.edu:/path/to/remote/folder  /path/to/local/destination

# --include='*/': Keeps all directories, allowing rsync to traverse into subdirectories
# --include='*.html': Includes only .html files
# --exclude='*': Excludes everything else
# -a: Archive mode (preserves metadata)
# -v: Verbose output
# --prune-empty-dirs: Avoids creating empty directories on the local machine
```

and it will require your password and Duo validation as well.