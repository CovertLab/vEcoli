#!/bin/bash
# Start an interactive Docker or Apptainer container from an image
# built with runscripts/container/build-image.sh.
# Supports optional bind mounts and Cloud Storage bucket mounting

set -eu # Exit on any error or unset variable

# Keep track of resources to clean up
TMP_OVERLAY_DIR=""

cleanup() {
  # Unmount bucket if mounted
  if [ -d "$(pwd)/bucket_mnt" ]; then
    fusermount -u $(pwd)/bucket_mnt &>/dev/null || true
    rm -rf "$(pwd)/bucket_mnt" &>/dev/null || true
    echo "Unmounted Cloud Storage bucket"
  fi

  # Remove the temporary overlay directory if it exists
  if [ -n "$TMP_OVERLAY_DIR" ] && [ -d "$TMP_OVERLAY_DIR" ]; then
    rm -rf "$TMP_OVERLAY_DIR" &>/dev/null || true
    echo "Cleaned up temporary overlay directory"
  fi
}

# Ensure resources are cleaned up on exit
trap cleanup EXIT INT TERM

# Default configuration variables
IMAGE_NAME="${USER}-image"
USE_APPTAINER=0
RUN_LOCAL=0
DEV_MODE=0
REGISTRY="gcp" # Default registry: "gcp" for Artifact Registry, "aws" for ECR
BIND_MOUNTS=()
BIND_STR=""
ENV_STR=""
BUCKET=""
OVERLAY_SIZE=1024
COMMAND="" # Default is empty, will start interactive shell if not specified

# Help message string
usage_str="Usage: interactive.sh [-i IMAGE_NAME] [-d] [-a] [-s OVERLAY_SIZE] [-l] [-r REGISTRY] [-b BUCKET] [-p PATH] [-c \"COMMAND\"]\n\
Options:\n\
    -i: Path to image to run if -a or -l are passed, otherwise name of Docker \
image inside the specified registry; defaults to \"$IMAGE_NAME\".\n\
    -d: Create editable install of current directory in container virtual environment; \
useful for making and testing code changes that, unlike changes to
the code in the container at /vEcoli, are persistent and work with git.\n\
    -a: Load Apptainer image (cannot use with -l).\n\
    -s: Size of sparse temporary Apptainer overlay image in MB; \
defaults to \"$OVERLAY_SIZE\" (only used if -a is passed).\n
    -l: Load local Docker image (cannot use with -a).\n\
    -r: Registry to pull image from: \"gcp\" for Artifact Registry or \"aws\" for ECR; \
defaults to \"$REGISTRY\" (ignored if -l or -a is used).\n\
    -b: Name of Cloud Storage bucket to mount inside container; first mounts
bucket at $(pwd)/bucket_mnt using gcsfuse (does not work with -a).\n\
    -p: Path(s) to mount inside container; can specify multiple with \
\"-p path1 -p path2\"\n\
    -c: Command to run inside container (non-interactive mode); if not provided, \
an interactive bash shell will be started.\n"

# Function to print usage instructions
print_usage() {
  printf "$usage_str"
}

# Parse command-line options
while getopts 'i:das:lr:b:p:c:' flag; do
  case "${flag}" in
  i) IMAGE_NAME="${OPTARG}" ;; # Set custom image name
  d) DEV_MODE=1 ;;             # Enable development mode
  a)
    # Make sure -a and -l are not both specified
    if [ "$RUN_LOCAL" -eq 1 ]; then
      echo "ERROR: Options -a (Apptainer) and -l (local Docker) cannot be used together."
      print_usage
      exit 1
    fi
    USE_APPTAINER=1
    ;;                           # Enable Apptainer mode
  s) OVERLAY_SIZE="${OPTARG}" ;; # Set the size of the sparse overlay
  l)
    # Make sure -l and -a are not both specified
    if [ "$USE_APPTAINER" -eq 1 ]; then
      echo "ERROR: Options -l (local Docker) and -a (Apptainer) cannot be used together."
      print_usage
      exit 1
    fi
    RUN_LOCAL=1
    ;;                           # Enable local Docker mode
  r)
    # Validate registry option
    if [ "${OPTARG}" != "gcp" ] && [ "${OPTARG}" != "aws" ]; then
      echo "ERROR: Invalid registry option '${OPTARG}'. Must be 'gcp' or 'aws'."
      print_usage
      exit 1
    fi
    REGISTRY="${OPTARG}"
    ;;                                         # Set registry (gcp or aws)
  b) BUCKET="${OPTARG}" ;;                     # Set Cloud Storage bucket to mount
  p) BIND_MOUNTS+=($(realpath "${OPTARG}")) ;; # Collect absolute mount path(s)
  c) COMMAND="${OPTARG}" ;;                    # Set the command to run (non-interactive mode)
  *)
    print_usage # Print usage for unknown flags
    exit 1
    ;;
  esac
done

# Validate that bucket mounting is not used with Apptainer
if [ -n "$BUCKET" ] && [ "$USE_APPTAINER" -eq 1 ]; then
  echo "ERROR: Bucket mounting (-b) is not supported with Apptainer mode (-a)."
  print_usage
  exit 1
fi

# Apptainer-specific logic
if (($USE_APPTAINER)); then
  # If there are bind mounts, format them for Apptainer
  if [ ${#BIND_MOUNTS[@]} -ne 0 ]; then
    BIND_STR=$(printf " -B %s" "${BIND_MOUNTS[@]}")
  fi

  echo "=== Launching Apptainer container from ${IMAGE_NAME} ==="

  # Create a temporary overlay directory
  TMP_OVERLAY_DIR=$(mktemp -d)
  echo "Creating ${OVERLAY_SIZE}MB sparse temporary overlay at ${TMP_OVERLAY_DIR}/overlay.img"
  # Create a sparse file (only allocates blocks as needed)
  dd if=/dev/zero of=${TMP_OVERLAY_DIR}/overlay.img bs=1M count=0 seek=${OVERLAY_SIZE}
  # Format the file as ext3 filesystem
  mkfs.ext3 -F ${TMP_OVERLAY_DIR}/overlay.img

  # Include SLURM resource environment variables to limit DuckDB default
  # number of threads on import (reduces baseline memory usage)
  SLURM_ENV="--env SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-} --env SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-} --env SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-}"

  if (($DEV_MODE)); then
    echo "Starting container in development mode..."
    # Fakeroot is necessary for overlay to work
    #
    # UV_PROJECT_ENVIRONMENT is set to the virtual environment inside
    # the container with all dependencies installed. This way uv does
    # not try to create a new one and waste time installing dependencies.
    #
    # UV_COMPILE_BYTECODE=0 skips byte code compilation which would
    # otherwise add dozens of seconds to the start time for development
    # mode. This is because we are doing an editable install of the
    # repository on the host machine to the container .venv.
    
    if [ -z "$COMMAND" ]; then
      # Interactive mode (default)
      apptainer exec -e --overlay ${TMP_OVERLAY_DIR}/overlay.img \
        --fakeroot ${BIND_STR} ${SLURM_ENV} ${IMAGE_NAME} \
        bash -c "export UV_PROJECT_ENVIRONMENT=/vEcoli/.venv && \
        export UV_COMPILE_BYTECODE=0 && uv sync --frozen && exec bash"
    else
      # Non-interactive mode with custom command
      echo "Running command: $COMMAND"
      apptainer exec -e --overlay ${TMP_OVERLAY_DIR}/overlay.img \
        --fakeroot ${BIND_STR} ${SLURM_ENV} ${IMAGE_NAME} \
        bash -c "export UV_PROJECT_ENVIRONMENT=/vEcoli/.venv && \
        export UV_COMPILE_BYTECODE=0 && uv sync --frozen && $COMMAND"
    fi
  else
    if [ -z "$COMMAND" ]; then
      # Interactive mode (default)
      echo "Starting container in interactive mode..."
      apptainer exec -e --overlay ${TMP_OVERLAY_DIR}/overlay.img \
        --fakeroot ${BIND_STR} ${SLURM_ENV} ${IMAGE_NAME} bash
    else
      # Non-interactive mode with custom command
      echo "Running command: $COMMAND"
      apptainer exec -e --overlay ${TMP_OVERLAY_DIR}/overlay.img \
        --fakeroot ${BIND_STR} ${SLURM_ENV} ${IMAGE_NAME} bash -c "$COMMAND"
    fi
  fi
else
  # Docker-specific logic
  if ((!$RUN_LOCAL)); then
    # Pull Docker images from the specified registry
    if [ "$REGISTRY" = "gcp" ]; then
      # Pull from Google Artifact Registry
      PROJECT=$(gcloud config get project)
      REGION=$(gcloud config get compute/region)
      IMAGE_NAME="--pull=always ${REGION}-docker.pkg.dev/${PROJECT}/vecoli/${IMAGE_NAME}"
    elif [ "$REGISTRY" = "aws" ]; then
      # Pull from AWS ECR - get repository URI directly
      REPO_URI=$(aws ecr describe-repositories --repository-names ${IMAGE_NAME} --query 'repositories[0].repositoryUri' --output text)
      # Extract region from repository URI (format: account.dkr.ecr.region.amazonaws.com/repo)
      AWS_REGION=$(echo $REPO_URI | cut -d'.' -f4)
      # Extract registry URL (everything before the repository name)
      REGISTRY_URL=$(echo $REPO_URI | cut -d'/' -f1)
      # Authenticate Docker with ECR
      echo "Authenticating Docker with AWS ECR..."
      aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${REGISTRY_URL}
      IMAGE_NAME="--pull=always ${REPO_URI}"
      # Set AWS region environment variable for S3 access inside container
      ENV_STR="--env AWS_DEFAULT_REGION=${AWS_REGION}"
    fi
  fi
  echo "=== Launching Docker container from ${IMAGE_NAME} ==="

  # If there are bind mounts, format them for Docker
  if [ ${#BIND_MOUNTS[@]} -ne 0 ]; then
    BIND_STR=$(printf " -v %s:%s" "${BIND_MOUNTS[@]}" "${BIND_MOUNTS[@]}")
  fi

  # Mount the cloud storage bucket using gcsfuse if provided
  if [ -n "$BUCKET" ]; then
    echo "Mounting bucket ${BUCKET} inside container at /mnt/disks/${BUCKET}"
    # Create mount point and mount bucket with gcsfuse
    mkdir -p $(pwd)/bucket_mnt
    gcsfuse -o allow_other --implicit-dirs $BUCKET $(pwd)/bucket_mnt
    # Nextflow mounts bucket to /mnt/disks so we need to copy that for
    # symlinks to work properly
    BIND_STR="${BIND_STR} -v $(pwd)/bucket_mnt:/mnt/disks/${BUCKET}"
  fi

  if (($DEV_MODE)); then
    if [ -z "$COMMAND" ]; then
      # Interactive mode (default)
      echo "Starting container in development mode (interactive)..."
      docker container run -it \
        --env UV_PROJECT_ENVIRONMENT=/vEcoli/.venv \
        --env UV_COMPILE_BYTECODE=0 \
        -v $(pwd):$(pwd) --workdir $(pwd) \
        ${BIND_STR} ${ENV_STR} ${IMAGE_NAME} \
        bash -c "uv sync --frozen && exec bash"
    else
      # Non-interactive mode with custom command
      echo "Running command in development mode: $COMMAND"
      docker container run -i \
        --env UV_PROJECT_ENVIRONMENT=/vEcoli/.venv \
        --env UV_COMPILE_BYTECODE=0 \
        -v $(pwd):$(pwd) --workdir $(pwd) \
        ${BIND_STR} ${ENV_STR} ${IMAGE_NAME} \
        bash -c "uv sync --frozen && $COMMAND"
    fi
  else
    if [ -z "$COMMAND" ]; then
      # Interactive mode (default)
      echo "Starting container in interactive mode..."
      docker container run -it ${BIND_STR} ${ENV_STR} ${IMAGE_NAME} bash
    else
      # Non-interactive mode with custom command
      echo "Running command: $COMMAND"
      docker container run -i ${BIND_STR} ${ENV_STR} ${IMAGE_NAME} bash -c "$COMMAND"
    fi
  fi
fi
